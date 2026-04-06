"""
Local smoke test — runs all 3 tasks directly (no Docker, no network).

Run from the repo root:
    cd dynamic_pricing_env
    python test_local.py
"""

import sys
import os

# Ensure we can import from the project root no matter where Python is invoked
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from server.pricing_environment import DynamicPricingEnvironment
from models import PricingAction
from tasks import list_tasks, TASK_REGISTRY

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  OK  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}" + (f" -- {detail}" if detail else ""))


def run_task_test(task_name: str) -> None:
    print(f"\n-- Task: {task_name} --")

    env = DynamicPricingEnvironment(task_name=task_name)

    obs = env.reset(seed=42)
    check("reset() returns observation", obs is not None)
    check("observation has skus list", hasattr(obs, "skus") and isinstance(obs.skus, list))
    check("at least one SKU", len(obs.skus) > 0)
    check("done=False after reset", obs.done is False)
    check("reward=0.0 after reset", obs.reward == 0.0)
    check("step_number=0 after reset", obs.step_number == 0)

    state = env.state
    check("state.task_name matches", state.task_name == task_name)
    check("state.step_count=0", state.step_count == 0)

    num_skus = len(obs.skus)
    max_steps = env._task.max_steps

    all_rewards = []
    done = False
    last_obs = obs

    for step in range(1, max_steps + 2):
        if done:
            break
        prices = [sku.current_price for sku in last_obs.skus]
        result = env.step(PricingAction(prices=prices))
        last_obs = result

        reward = result.reward
        check(
            f"step {step}: reward in [0,1]",
            0.0 <= reward <= 1.0,
            f"got {reward}",
        )
        check(
            f"step {step}: correct SKU count",
            len(result.skus) == num_skus,
            f"expected {num_skus}, got {len(result.skus)}",
        )
        all_rewards.append(reward)
        done = result.done

    check("episode ends with done=True", done)
    check(
        f"ran exactly {max_steps} steps",
        len(all_rewards) == max_steps,
        f"ran {len(all_rewards)}",
    )

    score = env.grade_episode()
    check("grade_episode() returns float", isinstance(score, float))
    check("grade in [0, 1]", 0.0 <= score <= 1.0, f"got {score}")
    print(f"     Episode score: {score:.4f}")


def test_invalid_task() -> None:
    print("\n-- Invalid task name --")
    try:
        DynamicPricingEnvironment(task_name="nonexistent_task")
        check("raises ValueError for unknown task", False)
    except ValueError:
        check("raises ValueError for unknown task", True)


def test_task_registry() -> None:
    print("\n-- Task registry --")
    tasks = list_tasks()
    check("3 tasks registered", len(tasks) == 3, f"got {len(tasks)}")
    for name in ["single_sku_stable", "multi_sku_competitors", "demand_shocks_perishables"]:
        check(f"'{name}' in registry", name in tasks)


def main() -> None:
    print("=" * 52)
    print("  Dynamic Pricing Env -- Local Smoke Test")
    print("=" * 52)

    test_task_registry()
    test_invalid_task()

    for task_name in list_tasks():
        run_task_test(task_name)

    print("\n" + "=" * 52)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("=" * 52)

    if FAIL > 0:
        print("\n[FAIL] Fix the above issues before submitting.")
        sys.exit(1)
    else:
        print("\n[PASS] All checks passed. Ready to deploy.")
        sys.exit(0)


if __name__ == "__main__":
    main()