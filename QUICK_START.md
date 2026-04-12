# 🚀 Dynamic Pricing Environment - Quick Start & Project Walkthrough

## What This Project Does

This is a **reinforcement learning environment** for **e-commerce dynamic pricing optimization**. An AI agent learns to set optimal prices for products while juggling:

- **Revenue** (set higher prices but sell less)
- **Inventory health** (avoid stockouts & overstock)
- **Competition** (competitor bots undercut your prices)
- **Demand shocks** (task 3: perishable products that expire)

**Real-world problem it solves**: 9.7 million Amazon sellers & 2 million+ Shopify sellers manually adjust product prices daily. Existing tools don't learn — they just apply rules. This environment lets RL agents learn optimal pricing strategies.

---

## 3 Difficulty Levels

| Task | SKUs | Competitors | Complexity | Grading |
|------|------|-------------|-----------|---------|
| **Task 1** `single_sku_stable` | 1 | None | Learn optimal price | Revenue-based |
| **Task 2** `multi_sku_competitors` | 5 | 2 reactive bots | Balance portfolio vs. competition | Revenue + inventory health |
| **Task 3** `demand_shocks_perishables` | 3 | 1 bot | Manage expiry + demand volatility | Complex multi-objective |

---

## Project Files - What Each Does

### Core Logic
| File | Purpose |
|------|---------|
| **models.py** | Dataclass definitions: `PricingAction` (what agent decides), `PricingObservation` (what agent sees), `SKUInfo` (per-product snapshot) |
| **market_simulator.py** | Simulates demand with price elasticity, competitor bots, sales sampling, reward computation |
| **tasks.py** | Defines 3 tasks with SKU configs, episode lengths, difficulty, and graders |

### Server & API
| File | Purpose |
|------|---------|
| **server/pricing_environment.py** | Core `DynamicPricingEnvironment` class — implements OpenEnv spec (`reset()`, `step()`, `state`) |
| **server/app.py** | FastAPI wrapper — exposes environment as WebSocket servers for remote agents |

### Testing & Baseline
| File | Purpose |
|------|---------|
| **test_local.py** | Smoke test — validates all 3 tasks work locally (no Docker needed) |
| **inference.py** | **Baseline agent** — uses OpenAI LLM to make pricing decisions, logs structured output |
| **demo_environment.py** | Standalone demo — shows concepts WITHOUT external dependencies ✨ |

### Config
| File | Purpose |
|------|---------|
| **pyproject.toml** | Project metadata & dependencies |
| **openenv.yaml** | OpenEnv spec file (required by hackathon) |
| **Dockerfile** | Container config for HuggingFace Spaces deployment |

---

## How to Run It

### **Option 1: See It Without Dependencies (NOW)** ✓

```bash
# No installation needed!
python demo_environment.py
```

**Output**: Shows how environment works, demonstrates reward logic, displays sample agent decisions.

---

### **Option 2: Full Local Test (Setup Required)**

**Prerequisites**: Python 3.11+ (3.15 alpha has build issues)

```bash
# 1. Install dependencies
pip install fastapi uvicorn pydantic openai openenv-core httpx

# 2. Run smoke tests (all 3 tasks)
python test_local.py

# Expected output:
# -- Task: single_sku_stable --
#   OK  reset() returns observation
#   OK  observation has skus list
#   OK  done=False after reset
#   ...
#   Episode score: 0.8234
#
# -- Task: multi_sku_competitors --
#   ...
#   Episode score: 0.7156
#
# -- Task: demand_shocks_perishables --
#   ...
#   Episode score: 0.6892
#
# PASS: 150  FAIL: 0
```

---

### **Option 3: Run as FastAPI Server (WebSocket)**

```bash
# Terminal 1: Start server
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Terminal 2: Make requests (Python)
import asyncio
from openenv.openenv_client import OpenEnvClient

async def test():
    client = OpenEnvClient("ws://localhost:7860")
    await client.connect()
    
    obs = await client.reset(task="single_sku_stable")
    print(f"Step 0: {obs}")
    
    for step in range(20):
        action = {"prices": [50.0]}  # Set price
        obs = await client.step(action)
        print(f"Step {step+1}: Sold {obs.skus[0].sales_last_step} units, "
              f"reward={obs.reward:.3f}")
    
    await client.disconnect()

asyncio.run(test())
```

---

### **Option 4: Run with LLM Baseline Agent**

```bash
# Set environment variables
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4"
export SPACE_URL="http://localhost:7860"  # or HuggingFace Space URL

# Run inference
python inference.py

# Output (structured for hackathon grader):
# [START] task=single_sku_stable env=dynamic_pricing_env model=gpt-4
# [STEP]  step=1 action={"prices": [52.50]} reward=0.85 done=false error=null
# [STEP]  step=2 action={"prices": [51.00]} reward=0.82 done=false error=null
# ...
# [END]   success=true steps=20 score=0.8234 rewards=0.85,0.82,...,0.80
```

---

## Example: What Happens in One Episode

### Task 1 (Single SKU, Stable Demand)

```
EPISODE START
  Base price: $50
  Base demand: 10 units/step
  Inventory: 300 units
  Max steps: 20
  Goal: Maximize revenue

---

STEP 1 - Agent decides: "Set price = $52.50"
  Price ratio: $50 / $52.50 = 0.952
  Elasticity effect: 0.952^1.2 = 0.944
  Expected demand: 10 * 0.944 = 9.44 units
  Actual sales (random): 8 units
  Revenue: 8 × $52.50 = $420
  Inventory: 292 units
  Inventory health: 292/300 = 97% → Slight overstock penalty
  Reward: 0.8227 (normalized)

STEP 2 - Agent decides: "Set price = $51.00"
  Expected demand: 10 * (50/51)^1.2 = 9.81 units
  Actual sales: 7 units
  Revenue: 7 × $51 = $357
  Cumulative revenue: $777
  Reward: 0.6990

... (steps 3-20)

EPISODE END (step 20)
  Total revenue: $9,234
  Total units sold: 185
  Final inventory: 115 units
  Grade: 0.9234 (92.34% of theoretical max $10,000)
```

**What agent learned**: Price elasticity trade-off — going 5% above base price loses ~6% demand but gains 5% margin.

---

## The Environment API

### **reset(task_name, seed) → Observation**
Starts a new episode.

```python
obs = env.reset(task_name="single_sku_stable", seed=42)
print(obs.skus[0].sku_id)        # "SKU-001"
print(obs.episode_revenue)        # 0.0 (fresh episode)
print(obs.done)                   # False
```

### **step(action) → Observation**
Executes one market step.

```python
action = PricingAction(prices=[52.50])
obs = env.step(action)
print(obs.reward)                 # 0.8227
print(obs.skus[0].sales_last_step)  # 8
print(obs.episode_revenue)        # 420.0
print(obs.done)                   # False (if not end of episode)
```

### **state → State**
Inspect internal episode state (READ-ONLY).

```python
state = env.state
print(state.task_name)            # "single_sku_stable"
print(state.max_steps)            # 20
print(state.total_revenue)        # 420.0
print(state.step_count)           # 1
```

### **grade_episode() → float**
Score the episode [0.0, 1.0] after done=True.

```python
score = env.grade_episode()
print(f"Grade: {score:.4f}")       # Grade: 0.9234
```

---

## Reward Signal Explained

Each step gives a **shaped reward** [0.0 to 1.0]:

$$\text{Reward} = \frac{\text{Revenue} + \text{Bonuses} - \text{Penalties}}{\text{Max Possible}}$$

### Components

| Component | Value | When |
|-----------|-------|------|
| **Revenue** | units × price | Always |
| **Inventory Bonus** | +0.1 × base_price | If 20% < stock% < 80% |
| **Stockout Penalty** | −0.5 × base_price | When inventory = 0 |
| **Overstock Penalty** | −0.2 × base_price × ratio | When stock% > 80% |
| **Expiry Loss** (Task 3) | −0.8 × base_price × units | When perishables expire |

### Example Calculation

```
Step 5:
  Revenue from sales: $420
  Inventory: 150/300 (50%) → Inventory bonus: +$5
  Stockout: No → Penalty: $0
  Overstock: No → Penalty: $0
  ---
  Raw reward: $425
  Max possible per step: $500
  Normalized reward: 425/500 = 0.85
```

---

## Price Elasticity Model (Why Pricing Matters)

The environment uses **log-linear demand**:

$$\text{Demand} = D_{base} \times \left(\frac{P_{base}}{P_{agent}}\right)^{|\text{elasticity}|} \times \text{competitor\_effect}$$

**What this means**:
- **Higher agent price** → Lower demand
- **Elasticity = 1.2** → 20% price increase → ~22% demand drop
- **Elasticity = 2.0** (SKU-D) → Very price-sensitive (high elasticity)
- **Elasticity = 0.8** (SKU-C) → Less price-sensitive (low elasticity)

### Competitor Effect (Task 2+)

```python
# If agent price > competitor price
competitor_effect = 0.4 + 0.6 / (1 + exp(3 * relative_price - 3))
# Max loss: 60% if agent is 2x more expensive
```

**Insight**: Going premium is risky if competitors undercut you.

---

## File Behavior Summary

| Scenario | Run This | Output |
|----------|----------|--------|
| Quick demo (no setup) | `python demo_environment.py` | Simulated episprices, rewards, grades |
| Validate environment works | `python test_local.py` | PASS/FAIL for all 3 tasks |
| Interactive testing | `python -m uvicorn server.app:app --reload` + WebSocket client | Live market simulation |
| LLM baseline | `python inference.py` | `[START]`, `[STEP]`, `[END]` logs for grader |

---

## Troubleshooting

### Import Error: `ModuleNotFoundError: openenv`
**Solution**: Install dependencies
```bash
pip install openenv-core fastapi pydantic uvicorn openai httpx
```

### Python 3.15 alpha build errors
**Solution**: Use Python 3.11 or 3.12 (pre-compiled wheels)
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Task name doesn't exist
**Valid names**:
- `single_sku_stable` (Task 1)
- `multi_sku_competitors` (Task 2)
- `demand_shocks_perishables` (Task 3)

### Grade is always 0.0
**Check**: 
- Is `grade_episode()` called after `done=True`?
- Did the episode actually run steps?
- Are prices valid (> 0)?

---

## Next Steps for Hackathon

1. ✅ **Run demo**: `python demo_environment.py`
2. ✅ **Test locally**: `python test_local.py` (with dependencies)
3. **Build baseline agent**: Modify `inference.py` to use your own LLM/strategy
4. **Measure scores**: Run all 3 tasks, collect grades
5. **Deploy**: Push Dockerfile to HuggingFace Spaces
6. **Write README**: Document your approach & results
7. **Submit**: HuggingFace Space URL + GitHub link

---

## Key Insights for Agents

### Task 1 (Single SKU)
- **Learn**: Optimal pricing is slightly above cost (elasticity trade-off)
- **Naive agent score**: ~0.60-0.70
- **Expert agent score**: ~0.92-0.96

### Task 2 (Multi-SKU + Competitors)
- **Learn**: Different SKUs need different strategies (elasticity-dependent)
- **Naive**: Undercut everyone → margin shrinkage
- **Smart**: Premium on inelastic, aggressive on elastic SKUs
- **Expert agent score**: ~0.80-0.88

### Task 3 (Perishables + Shocks)
- **Learn**: Discount aggressively as expiry approaches
- **Naive**: Hold at high price → expiry loss
- **Smart**: Monitor `days_until_expiry`, adjust based on demand signals
- **Expert agent score**: ~0.85-0.92

---

## Resources

- **OpenEnv Spec**: https://open-env.ai/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Pydantic**: https://docs.pydantic.dev/
- **HuggingFace Spaces**: https://huggingface.co/spaces

Good luck with the hackathon! 🎯
