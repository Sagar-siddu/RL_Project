"""
Dynamic Pricing Environment — Typed Models
Action, Observation, and State using dataclasses (OpenEnv pattern).
"""

from dataclasses import dataclass, field
from typing import List, Optional
from openenv.core.env_server import Action, Observation
from openenv.core.env_server.types import State


@dataclass
class PricingAction(Action):
    """
    Agent sets a new price for each SKU.
    prices: list of floats, one per SKU, in the same order as observation.skus.
    Each price must be > 0.
    """
    prices: List[float] = field(default_factory=list)


@dataclass
class SKUInfo:
    """Per-SKU snapshot included in each observation."""
    sku_id: str
    current_price: float
    competitor_price: float
    inventory: int
    sales_last_step: int
    demand_signal: float
    days_until_expiry: Optional[int] = None


@dataclass
class PricingObservation(Observation):
    """
    Full market snapshot returned by reset() and step().
    """
    skus: List[SKUInfo] = field(default_factory=list)
    step_number: int = 0
    episode_revenue: float = 0.0
    episode_units_sold: int = 0
    last_action_prices: List[float] = field(default_factory=list)
    last_action_error: Optional[str] = None
    task_name: str = "single_sku_stable"
    done: bool = False
    reward: float = 0.0


@dataclass
class PricingState(State):
    """Internal episode state — returned by state() endpoint."""
    task_name: str = "single_sku_stable"
    max_steps: int = 30
    num_skus: int = 1
    total_revenue: float = 0.0
    total_units_sold: int = 0
    stockout_count: int = 0
    overstock_penalty_total: float = 0.0