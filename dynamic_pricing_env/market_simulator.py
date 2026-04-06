"""
Market Simulator
================
Simulates realistic e-commerce demand with:
  - Price elasticity (higher price → fewer sales)
  - Demand signals (seasonal trends, flash spikes)
  - Rule-based competitor bots that react to agent pricing
  - Inventory dynamics (holding cost, stockout, expiry)
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# SKU internal state (server-side, not exposed to agent)
# ---------------------------------------------------------------------------

@dataclass
class SKUState:
    sku_id: str
    base_price: float           # "fair market" anchor
    current_price: float
    competitor_price: float
    inventory: int
    max_inventory: int
    base_demand: float          # expected units/step at base_price
    elasticity: float           # price sensitivity (negative: higher price→less demand)
    demand_signal: float = 1.0  # multiplier set each step
    sales_last_step: int = 0
    days_until_expiry: Optional[int] = None
    is_perishable: bool = False


# ---------------------------------------------------------------------------
# Competitor engine
# ---------------------------------------------------------------------------

class CompetitorBot:
    """
    Simple rule-based repricing competitor.
    Strategy: match the agent if agent goes below competitor; otherwise
    undercut by a small random margin.
    """

    def __init__(self, sku_id: str, base_price: float, aggression: float = 0.05):
        self.sku_id = sku_id
        self.base_price = base_price
        self.aggression = aggression  # fraction they try to undercut
        self.price = base_price

    def reprice(self, agent_price: float) -> float:
        """React to agent's new price."""
        if agent_price < self.price:
            # Agent undercut us — match plus small discount
            self.price = agent_price * (1.0 - random.uniform(0, self.aggression))
        else:
            # Agent raised price — competitor raises slightly too, staying cheaper
            target = agent_price * (1.0 - random.uniform(0.01, self.aggression))
            # Don't go below 60% of base price
            self.price = max(target, self.base_price * 0.6)
        return round(max(self.price, 0.01), 2)


# ---------------------------------------------------------------------------
# Demand model
# ---------------------------------------------------------------------------

def compute_demand(
    sku: SKUState,
    agent_price: float,
    competitor_price: float,
) -> float:
    """
    Log-linear demand model:
      demand = base_demand × (base_price / agent_price)^elasticity
               × demand_signal
               × competitor_effect

    competitor_effect: if agent price is higher than competitor, some customers
    defect (max 60% loss at 2× competitor price).
    """
    if agent_price <= 0:
        return 0.0

    # Own-price elasticity
    price_ratio = sku.base_price / agent_price
    own_effect = price_ratio ** abs(sku.elasticity)

    # Cross-price effect from competitor
    if competitor_price > 0:
        relative = agent_price / competitor_price
        # sigmoid-style: neutral at 1.0, lose up to 60% if 2× more expensive
        competitor_effect = 1.0 / (1.0 + math.exp(3.0 * (relative - 1.0)))
        competitor_effect = 0.4 + 0.6 * competitor_effect  # floor at 0.4
    else:
        competitor_effect = 1.0

    raw = sku.base_demand * own_effect * sku.demand_signal * competitor_effect
    return max(0.0, raw)


def sample_sales(expected_units: float) -> int:
    """Poisson sample of actual units sold."""
    if expected_units <= 0:
        return 0
    return random.randint(
        max(0, int(expected_units * 0.7)),
        int(expected_units * 1.3) + 1,
    )


# ---------------------------------------------------------------------------
# Demand signal generator (seasonality + random shocks)
# ---------------------------------------------------------------------------

def get_demand_signal(step: int, rng: random.Random, shock_prob: float = 0.0) -> float:
    """
    Smooth seasonal trend + optional random shocks.
    Returns a multiplier around 1.0.
    """
    # Slow sine wave (period ~20 steps)
    seasonal = 1.0 + 0.25 * math.sin(2 * math.pi * step / 20)

    shock = 1.0
    if rng.random() < shock_prob:
        # Flash sale (demand spike) or demand slump
        shock = rng.choice([1.8, 1.6, 0.4, 0.3])

    return round(seasonal * shock, 3)


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_step_reward(
    skus: List[SKUState],
    sales: List[int],
    prices: List[float],
    stockouts: List[bool],
    expiry_losses: List[int],
    max_possible_revenue: float,
) -> Tuple[float, float]:
    """
    Returns (raw_reward, normalized_reward_0_1).

    Components:
      + revenue from sales
      + inventory health bonus (stock in 20–80% of max)
      - stockout penalty (lost sales signal)
      - overstock penalty (holding cost proxy)
      - expiry loss penalty
    """
    revenue = sum(s * p for s, p in zip(sales, prices))

    health_bonus = 0.0
    stockout_penalty = 0.0
    overstock_penalty = 0.0
    expiry_penalty = 0.0

    for sku, sold, stockout, exp_loss in zip(skus, stockouts, stockouts, expiry_losses):
        inv_ratio = sku.inventory / sku.max_inventory if sku.max_inventory > 0 else 0
        if 0.2 <= inv_ratio <= 0.8:
            health_bonus += sku.base_price * 0.1
        if stockout:
            stockout_penalty += sku.base_price * 0.5
        if inv_ratio > 0.9:
            overstock_penalty += sku.base_price * 0.2 * (inv_ratio - 0.9) * 10
        if exp_loss > 0:
            expiry_penalty += exp_loss * sku.base_price * 0.8

    raw = revenue + health_bonus - stockout_penalty - overstock_penalty - expiry_penalty
    # Normalize: use max_possible_revenue as ceiling
    normalized = raw / max_possible_revenue if max_possible_revenue > 0 else 0.0
    normalized = max(0.0, min(1.0, normalized))
    return raw, normalized
