"""
Task Definitions & Graders
===========================
Three tasks with increasing difficulty.
Each TaskConfig defines the SKU setup, episode length, and a grader.

Grader contract:
  grade(episode_log) -> float in [0.0, 1.0]
  episode_log: dict with keys:
    total_revenue, total_units_sold, stockout_count,
    overstock_penalty_total, steps_taken, max_steps,
    rewards (list of per-step normalized rewards)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from market_simulator import SKUState, CompetitorBot


# ---------------------------------------------------------------------------
# Task configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    name: str
    description: str
    difficulty: str
    max_steps: int
    skus: List[SKUState]
    competitors: List[CompetitorBot]
    shock_prob: float = 0.0       # probability of demand shock per step
    has_perishables: bool = False


# ---------------------------------------------------------------------------
# Grader helpers
# ---------------------------------------------------------------------------

def _revenue_score(total_revenue: float, theoretical_max: float) -> float:
    """Revenue as fraction of theoretical maximum. Clamped to [0,1]."""
    if theoretical_max <= 0:
        return 0.0
    return min(1.0, total_revenue / theoretical_max)


# ---------------------------------------------------------------------------
# Task 1 — Easy: Single SKU, stable demand, no competitors
# ---------------------------------------------------------------------------

TASK1_BASE_PRICE = 50.0
TASK1_BASE_DEMAND = 10.0  # units/step at base price
TASK1_MAX_STEPS = 20
# Theoretical max: sell at base price every step with no stockouts
# base_demand * max_steps * base_price (rough upper bound)
TASK1_THEORETICAL_MAX = TASK1_BASE_DEMAND * TASK1_MAX_STEPS * TASK1_BASE_PRICE  # 10000


def build_task1() -> TaskConfig:
    sku = SKUState(
        sku_id="SKU-001",
        base_price=TASK1_BASE_PRICE,
        current_price=TASK1_BASE_PRICE,
        competitor_price=0.0,       # no competitor
        inventory=300,
        max_inventory=300,
        base_demand=TASK1_BASE_DEMAND,
        elasticity=1.2,             # moderate price sensitivity
        demand_signal=1.0,
    )
    return TaskConfig(
        name="single_sku_stable",
        description=(
            "Single SKU with stable demand. No competitors. "
            "Find the optimal price that maximizes revenue across 20 steps."
        ),
        difficulty="easy",
        max_steps=TASK1_MAX_STEPS,
        skus=[sku],
        competitors=[],
        shock_prob=0.0,
    )


def grade_task1(log: Dict[str, Any]) -> float:
    """
    Score = weighted average of:
      70% revenue score vs theoretical max
      30% penalty for stockouts (lose 0.1 per stockout event)
    """
    rev_score = _revenue_score(log["total_revenue"], TASK1_THEORETICAL_MAX)
    stockout_penalty = min(1.0, log["stockout_count"] * 0.1)
    score = 0.7 * rev_score + 0.3 * (1.0 - stockout_penalty)
    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Task 2 — Medium: 5 SKUs, 2 competitor bots, reactive repricing
# ---------------------------------------------------------------------------

TASK2_MAX_STEPS = 25
TASK2_SKUS = [
    # (sku_id, base_price, base_demand, elasticity, init_inventory)
    ("SKU-A", 80.0,  8.0, 1.0, 200),
    ("SKU-B", 35.0, 15.0, 1.5, 400),
    ("SKU-C", 120.0, 5.0, 0.8, 150),
    ("SKU-D", 25.0, 20.0, 2.0, 500),
    ("SKU-E", 60.0, 10.0, 1.2, 250),
]
# Theoretical max per SKU = base_demand * max_steps * base_price
TASK2_THEORETICAL_MAX = sum(
    d * TASK2_MAX_STEPS * p for _, p, d, _, _ in TASK2_SKUS
)  # ≈ 86250


def build_task2() -> TaskConfig:
    skus = []
    for sku_id, base_price, base_demand, elasticity, inv in TASK2_SKUS:
        skus.append(SKUState(
            sku_id=sku_id,
            base_price=base_price,
            current_price=base_price,
            competitor_price=base_price * 0.95,
            inventory=inv,
            max_inventory=inv,
            base_demand=base_demand,
            elasticity=elasticity,
            demand_signal=1.0,
        ))
    # Two competitors: one aggressive (undercuts fast), one passive
    competitors = [
        CompetitorBot("SKU-A", 80.0,  aggression=0.08),
        CompetitorBot("SKU-C", 120.0, aggression=0.04),
    ]
    return TaskConfig(
        name="multi_sku_competitors",
        description=(
            "5 SKUs with 2 reactive competitor bots. "
            "Balance pricing across the catalogue while staying competitive. "
            "25 steps."
        ),
        difficulty="medium",
        max_steps=TASK2_MAX_STEPS,
        skus=skus,
        competitors=competitors,
        shock_prob=0.0,
    )


def grade_task2(log: Dict[str, Any]) -> float:
    """
    Score = 60% revenue score + 20% inventory health + 20% vs competitor benchmark.
    Competitor benchmark: revenue if agent always priced 5% above competitors.
    """
    rev_score = _revenue_score(log["total_revenue"], TASK2_THEORETICAL_MAX)
    # Inventory health: penalise stockouts and overstock
    inventory_score = max(0.0, 1.0 - log["stockout_count"] * 0.05
                          - log["overstock_penalty_total"] / (TASK2_THEORETICAL_MAX * 0.1))
    # Competitive score: did agent beat a naive +5% above competitor strategy?
    # Naive revenue estimate: ~80% of theoretical (competitor suppresses demand)
    naive_benchmark = TASK2_THEORETICAL_MAX * 0.80
    competitive_score = min(1.0, log["total_revenue"] / naive_benchmark)
    score = 0.6 * rev_score + 0.2 * inventory_score + 0.2 * competitive_score
    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Task 3 — Hard: 10 SKUs, demand shocks, perishable items
# ---------------------------------------------------------------------------

TASK3_MAX_STEPS = 30
TASK3_SKUS = [
    # (sku_id, base_price, base_demand, elasticity, init_inventory, expiry_days, perishable)
    ("PRD-01", 45.0,  12.0, 1.1, 300, None,  False),
    ("PRD-02", 90.0,   6.0, 0.9, 120, None,  False),
    ("PRD-03", 20.0,  25.0, 2.0, 600, None,  False),
    ("PRD-04", 150.0,  3.0, 0.7,  80, None,  False),
    ("PRD-05", 30.0,  18.0, 1.6, 400, None,  False),
    # Perishables — must sell before expiry or lose inventory
    ("FRESH-01", 15.0, 30.0, 2.5, 200, 10, True),
    ("FRESH-02", 25.0, 20.0, 2.0, 150, 8,  True),
    ("FRESH-03", 40.0, 15.0, 1.8, 100, 12, True),
    ("FRESH-04", 10.0, 40.0, 3.0, 250, 7,  True),
    ("FRESH-05", 55.0,  8.0, 1.4,  80, 15, True),
]
TASK3_THEORETICAL_MAX = sum(
    d * TASK3_MAX_STEPS * p for _, p, d, *_ in TASK3_SKUS
)  # ≈ 162000


def build_task3() -> TaskConfig:
    skus = []
    for row in TASK3_SKUS:
        sku_id, base_price, base_demand, elasticity, inv, expiry, perishable = row
        skus.append(SKUState(
            sku_id=sku_id,
            base_price=base_price,
            current_price=base_price,
            competitor_price=base_price * 0.93,
            inventory=inv,
            max_inventory=inv,
            base_demand=base_demand,
            elasticity=elasticity,
            demand_signal=1.0,
            days_until_expiry=expiry,
            is_perishable=perishable,
        ))
    # Four aggressive competitors
    competitors = [
        CompetitorBot("PRD-01", 45.0,  aggression=0.07),
        CompetitorBot("PRD-03", 20.0,  aggression=0.10),
        CompetitorBot("FRESH-01", 15.0, aggression=0.12),
        CompetitorBot("FRESH-04", 10.0, aggression=0.15),
    ]
    return TaskConfig(
        name="demand_shocks_perishables",
        description=(
            "10 SKUs including 5 perishable items with expiry deadlines. "
            "Demand shocks occur randomly. 4 aggressive competitor bots. "
            "30 steps — balance revenue, freshness sell-through, and competitive positioning."
        ),
        difficulty="hard",
        max_steps=TASK3_MAX_STEPS,
        skus=skus,
        competitors=competitors,
        shock_prob=0.15,
        has_perishables=True,
    )


def grade_task3(log: Dict[str, Any]) -> float:
    """
    Composite score:
      40% revenue (vs theoretical max)
      25% perishable sell-through (fraction of fresh inventory sold before expiry)
      20% inventory health (no stockouts, no overstock)
      15% resilience under shocks (steps with shock where revenue ≥ 50% of no-shock avg)
    """
    rev_score = _revenue_score(log["total_revenue"], TASK3_THEORETICAL_MAX)

    # Perishable sell-through
    total_fresh_capacity = sum(
        inv for _, _, _, _, inv, _, perishable in TASK3_SKUS if perishable
    )
    fresh_sold = log.get("fresh_units_sold", 0)
    fresh_score = min(1.0, fresh_sold / total_fresh_capacity) if total_fresh_capacity > 0 else 1.0

    # Inventory health
    inv_health = max(0.0, 1.0
                     - log["stockout_count"] * 0.03
                     - log.get("expiry_loss_units", 0) * 0.01
                     - log["overstock_penalty_total"] / (TASK3_THEORETICAL_MAX * 0.05))

    # Resilience: average reward during shock steps vs non-shock
    shock_rewards = log.get("shock_step_rewards", [])
    normal_rewards = log.get("normal_step_rewards", [])
    if shock_rewards and normal_rewards:
        avg_shock = sum(shock_rewards) / len(shock_rewards)
        avg_normal = sum(normal_rewards) / len(normal_rewards)
        resilience = min(1.0, avg_shock / avg_normal) if avg_normal > 0 else 0.5
    else:
        resilience = 0.5  # neutral if no shocks occurred

    score = (0.40 * rev_score
             + 0.25 * fresh_score
             + 0.20 * inv_health
             + 0.15 * resilience)
    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "single_sku_stable":         (build_task1, grade_task1),
    "multi_sku_competitors":      (build_task2, grade_task2),
    "demand_shocks_perishables":  (build_task3, grade_task3),
}


def list_tasks():
    return list(TASK_REGISTRY.keys())


def build_task(name: str) -> TaskConfig:
    builder, _ = TASK_REGISTRY[name]
    return builder()


def grade_task(name: str, log: Dict[str, Any]) -> float:
    _, grader = TASK_REGISTRY[name]
    return grader(log)
