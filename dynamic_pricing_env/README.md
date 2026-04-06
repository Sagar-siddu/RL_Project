# Dynamic Pricing Environment

> **OpenEnv environment for e-commerce SKU pricing optimisation.**
> An AI agent sets prices across a simulated market, balancing revenue,
> inventory health, and competitor response across multi-step episodes.

---

## Why This Exists

9.7 million Amazon third-party sellers and 2 million+ Shopify merchants face the same daily problem: *how do I price my SKUs without racing to the bottom or leaving money on the table?* Existing repricers (Feedvisor, Profit Cyclops) are rule-based and do not learn. Analytics tools (Sellerboard, Helium 10) show numbers but take no action.

This environment fills the gap: a realistic, multi-step RL environment where an agent can learn to jointly optimise revenue, inventory health, and competitive positioning — across a live simulated market.

---

## Environment Description

Each episode simulates a trading window (20–30 steps). At every step:

1. The agent observes the current market snapshot (prices, inventory, competitor prices, demand signals).
2. The agent sets a new price for each SKU.
3. The market responds: demand is computed from a price-elasticity model, units are sold, inventory is updated, and competitors reprice reactively.
4. A shaped reward signal is returned (not just end-of-episode).

The environment wraps a FastAPI server and communicates over WebSocket, following the OpenEnv spec exactly.

---

## Action Space

```python
@dataclass
class PricingAction(Action):
    prices: List[float]   # one price per SKU, in observation order, > 0
```

**Example action (Task 1):**
```json
{"prices": [52.50]}
```

**Example action (Task 2, 5 SKUs):**
```json
{"prices": [78.00, 33.50, 115.00, 23.99, 58.00]}
```

---

## Observation Space

```python
@dataclass
class PricingObservation(Observation):
    skus: List[SKUInfo]          # per-SKU market snapshot (see below)
    step_number: int             # current step within episode
    episode_revenue: float       # cumulative revenue so far
    episode_units_sold: int      # cumulative units sold so far
    last_action_prices: List[float]
    last_action_error: Optional[str]   # None if last action was valid
    task_name: str
    done: bool
    reward: float                # normalised reward for last step [0, 1]
```

Each `SKUInfo` contains:

| Field | Type | Description |
|---|---|---|
| `sku_id` | `str` | Unique SKU identifier |
| `current_price` | `float` | Agent's last set price |
| `competitor_price` | `float` | Competing seller's current price |
| `inventory` | `int` | Units remaining in stock |
| `sales_last_step` | `int` | Units sold in the previous step |
| `demand_signal` | `float` | Demand multiplier (1.0 = normal, >1 = high demand) |
| `days_until_expiry` | `int \| None` | Steps until perishable stock expires (Task 3 only) |

---

## Reward Function

The reward is shaped across the full trajectory — not just binary end-of-episode.

```
reward = (revenue + inventory_health_bonus
          - stockout_penalty - overstock_penalty - expiry_loss_penalty)
         / max_possible_revenue_per_step
```

Clamped to **[0.0, 1.0]** per step.

| Component | Signal | Effect |
|---|---|---|
| `revenue` | `units_sold × price` | Core incentive |
| `inventory_health_bonus` | `+0.1 × base_price` when stock in 20–80% of max | Rewards healthy stock levels |
| `stockout_penalty` | `−0.5 × base_price` per SKU at zero stock | Penalises lost-sale events |
| `overstock_penalty` | `−0.2 × base_price × excess_ratio` | Penalises holding cost |
| `expiry_loss_penalty` | `−0.8 × base_price × expired_units` | Penalises wasted perishable stock |

---

## Tasks

### Task 1 — Easy: `single_sku_stable`

| Property | Value |
|---|---|
| SKUs | 1 |
| Competitors | 0 |
| Demand shocks | No |
| Max steps | 20 |
| Baseline score | ~0.45 |

Single SKU with moderate price elasticity (`e = 1.2`). Stable demand. No competitors. The agent must find the revenue-maximising price by exploring the elasticity curve.

**Grader:** `0.7 × (revenue / theoretical_max) + 0.3 × (1 − stockout_rate)`

---

### Task 2 — Medium: `multi_sku_competitors`

| Property | Value |
|---|---|
| SKUs | 5 |
| Competitors | 2 reactive bots |
| Demand shocks | No |
| Max steps | 25 |
| Baseline score | ~0.38 |

Five SKUs with varying elasticities. Two competitor bots that undercut the agent when it raises prices. The agent must balance revenue across the catalogue while managing competitive pressure.

**Grader:** `0.6 × revenue_score + 0.2 × inventory_health + 0.2 × competitive_score`

---

### Task 3 — Hard: `demand_shocks_perishables`

| Property | Value |
|---|---|
| SKUs | 10 (5 standard + 5 perishable) |
| Competitors | 4 aggressive bots |
| Demand shocks | Yes (~15% chance/step) |
| Max steps | 30 |
| Baseline score | ~0.28 |

The hardest task. Perishable items expire if not sold before `days_until_expiry` reaches zero. Random demand shocks (flash spikes or slumps) require dynamic repricing. Four aggressive competitors react quickly.

**Grader:** `0.40 × revenue + 0.25 × perishable_sellthrough + 0.20 × inventory_health + 0.15 × shock_resilience`

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker
- `pip install openenv-core`

### Run locally (without Docker)

```bash
git clone <your-repo>
cd dynamic_pricing_env

pip install openenv-core fastapi uvicorn pydantic

# Start the server (default task: single_sku_stable)
PYTHONPATH=. PRICING_TASK=single_sku_stable uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal — quick smoke test
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'
```

### Run with Docker

```bash
# Build
docker build -t dynamic-pricing-env .

# Run (Task 1 by default)
docker run -p 7860:7860 -e PRICING_TASK=single_sku_stable dynamic-pricing-env

# Run Task 3
docker run -p 7860:7860 -e PRICING_TASK=demand_shocks_perishables dynamic-pricing-env
```

### Run inference script

```bash
export HF_TOKEN=hf_...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export SPACE_URL=https://<your-username>-dynamic-pricing-env.hf.space

python inference.py
```

### Deploy to Hugging Face Spaces

```bash
pip install openenv-core
openenv push --repo-id <your-username>/dynamic-pricing-env
```

---

## Project Structure

```
dynamic_pricing_env/
├── models.py                  # Typed Action / Observation / State (dataclasses)
├── market_simulator.py        # Demand model, elasticity, competitor engine, reward fn
├── tasks.py                   # Task configs + deterministic graders (3 tasks)
├── client.py                  # DynamicPricingEnv EnvClient
├── __init__.py                # Package exports
├── inference.py               # Submission inference script (root)
├── openenv.yaml               # OpenEnv spec manifest
├── Dockerfile                 # Container definition
├── pyproject.toml             # pip-installable package
└── server/
    ├── __init__.py
    ├── pricing_environment.py # DynamicPricingEnvironment (Environment subclass)
    ├── app.py                 # FastAPI app via create_app()
    └── requirements.txt       # Server dependencies
```

---

## Baseline Scores

Scores obtained by running `inference.py` with `Qwen/Qwen2.5-72B-Instruct`
via the HuggingFace router (temperature=0.2):

| Task | Difficulty | Baseline Score |
|---|---|---|
| `single_sku_stable` | Easy | ~0.45 |
| `multi_sku_competitors` | Medium | ~0.38 |
| `demand_shocks_perishables` | Hard | ~0.28 |

A random-pricing agent scores approximately 0.15–0.25 across tasks.
A frontier model (GPT-4o, Claude 3.5) is expected to score 0.55–0.70 on Task 1.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | Yes | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | Yes | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | Yes | — | HuggingFace / API key |
| `SPACE_URL` | Yes (inference) | — | Deployed HF Space URL |
| `LOCAL_IMAGE_NAME` | Optional | — | Local Docker image name |
| `PRICING_TASK` | Optional | `single_sku_stable` | Task for the server to run |

---

## License

Apache 2.0
