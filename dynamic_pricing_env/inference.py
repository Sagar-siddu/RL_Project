"""
Inference Script — Dynamic Pricing Environment
===============================================
MANDATORY REQUIREMENTS (from hackathon rules):
  - Named inference.py, placed in root directory
  - Uses OpenAI Client for all LLM calls
  - Reads API credentials from environment variables
  - Emits structured stdout logs: [START], [STEP], [END]
  - Each task returns score in [0, 1]
  - Runtime < 20 min on 2 vCPU / 8 GB RAM

STDOUT FORMAT:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

ENV VARS:
  API_BASE_URL  — LLM endpoint (default: HuggingFace router)
  MODEL_NAME    — Model identifier
  HF_TOKEN      — HuggingFace / API key
  SPACE_URL     — HuggingFace Space URL for the environment server
  LOCAL_IMAGE_NAME — Docker image name (alternative to SPACE_URL)
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — all from env vars, with safe defaults
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SPACE_URL = os.getenv("SPACE_URL", "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
BENCHMARK = "dynamic_pricing_env"

# Per-task settings
MAX_STEPS_OVERRIDE = {
    "single_sku_stable": 20,
    "multi_sku_competitors": 25,
    "demand_shocks_perishables": 30,
}
SUCCESS_THRESHOLD = 0.35   # score >= this → success
TEMPERATURE = 0.2
MAX_TOKENS = 512

TASKS = [
    "single_sku_stable",
    "multi_sku_competitors",
    "demand_shocks_perishables",
]

# ---------------------------------------------------------------------------
# Logging helpers — EXACT format required by the grader
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert e-commerce pricing strategist. You manage SKU prices
to maximise revenue while maintaining healthy inventory levels and
staying competitive against rivals.

RULES:
1. You must set prices for ALL SKUs listed in the observation.
2. Respond with ONLY a valid JSON object: {"prices": [p1, p2, ...]}
3. Each price must be a positive number (float).
4. Do NOT include any explanation — only the JSON object.

STRATEGY TIPS:
- Use price elasticity: if demand_signal > 1.2, you can raise price safely.
- If competitor_price < your current price by >15%, consider undercutting slightly.
- If inventory is very low, raise price to avoid stockout.
- If perishable (days_until_expiry <= 3), lower price aggressively to clear stock.
- Balance: a 10-20% margin above competitor is often optimal.
""").strip()


def build_user_prompt(obs_dict: dict, step: int) -> str:
    skus = obs_dict.get("skus", [])
    lines = []
    for i, sku in enumerate(skus):
        expiry = sku.get("days_until_expiry")
        expiry_str = f", days_until_expiry={expiry}" if expiry is not None else ""
        lines.append(
            f"  SKU {i} ({sku['sku_id']}): "
            f"current_price={sku['current_price']:.2f}, "
            f"competitor_price={sku['competitor_price']:.2f}, "
            f"inventory={sku['inventory']}, "
            f"sales_last_step={sku['sales_last_step']}, "
            f"demand_signal={sku['demand_signal']:.2f}"
            f"{expiry_str}"
        )
    sku_block = "\n".join(lines) if lines else "  (none)"
    return textwrap.dedent(f"""
        Step {step}
        Episode revenue so far: {obs_dict.get('episode_revenue', 0):.2f}
        Episode units sold: {obs_dict.get('episode_units_sold', 0)}
        Last reward: {obs_dict.get('reward', 0):.3f}

        Current market state:
        {sku_block}

        Respond with JSON only: {{"prices": [p1, p2, ...]}}
        (one price per SKU in the order listed above)
    """).strip()


def get_prices_from_model(
    client: OpenAI,
    obs_dict: dict,
    step: int,
    fallback_prices: List[float],
) -> List[float]:
    """Call LLM and parse prices. Returns fallback on any error."""
    user_prompt = build_user_prompt(obs_dict, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if model wraps in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        prices = [float(p) for p in parsed["prices"]]
        if len(prices) == 0:
            return fallback_prices
        return prices
    except Exception as exc:
        print(f"[DEBUG] Model call failed at step {step}: {exc}", flush=True)
        return fallback_prices


def obs_to_dict(observation) -> dict:
    """Convert observation object or dict to plain dict for prompting."""
    if isinstance(observation, dict):
        return observation
    # dataclass / Pydantic-style object
    try:
        import dataclasses
        if dataclasses.is_dataclass(observation):
            return dataclasses.asdict(observation)
    except Exception:
        pass
    return vars(observation)


def get_fallback_prices(obs_dict: dict) -> List[float]:
    """Return current prices as fallback (hold strategy)."""
    return [sku["current_price"] for sku in obs_dict.get("skus", [])]


# ---------------------------------------------------------------------------
# Single-task episode runner
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, task_name: str) -> dict:
    """
    Runs one complete episode for a task.
    Returns {"score": float, "steps": int, "rewards": list, "success": bool}
    """
    from dynamic_pricing_env import DynamicPricingEnv, PricingAction

    max_steps = MAX_STEPS_OVERRIDE.get(task_name, 30)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    # Connect to environment
    if SPACE_URL:
        env = DynamicPricingEnv(base_url=SPACE_URL)
        await env.__aenter__()
    elif LOCAL_IMAGE_NAME:
        env = await DynamicPricingEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        raise RuntimeError(
            "Set SPACE_URL (deployed HF Space URL) or LOCAL_IMAGE_NAME (local Docker image)."
        )

    try:
        # Reset — pass task name via reset params if supported
        result = await env.reset()
        obs_dict = obs_to_dict(result.observation)

        for step in range(1, max_steps + 1):
            if result.done:
                break

            fallback = get_fallback_prices(obs_dict)
            prices = get_prices_from_model(client, obs_dict, step, fallback)

            action = PricingAction(prices=prices)
            action_str = json.dumps({"prices": [round(p, 2) for p in prices]})

            result = await env.step(action)
            obs_dict = obs_to_dict(result.observation)

            reward = result.reward if result.reward is not None else 0.0
            done = result.done
            error = obs_dict.get("last_action_error")

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Final score: mean of per-step rewards (already normalised 0–1)
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error for task {task_name}: {exc}", flush=True)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"score": score, "steps": steps_taken, "rewards": rewards, "success": success}


# ---------------------------------------------------------------------------
# Main — run all 3 tasks sequentially
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[INFO] Running {len(TASKS)} tasks | model={MODEL_NAME} | env={BENCHMARK}",
          flush=True)

    all_scores = []
    for task_name in TASKS:
        result = await run_task(client, task_name)
        all_scores.append(result["score"])
        print(
            f"[INFO] Task '{task_name}' complete | "
            f"score={result['score']:.3f} | success={result['success']}",
            flush=True,
        )

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[INFO] Overall mean score: {overall:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
