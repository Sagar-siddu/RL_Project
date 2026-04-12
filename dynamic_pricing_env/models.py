"""
Dynamic Pricing Environment — Typed Models
Action, Observation, and State using dataclasses (OpenEnv pattern).
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pydantic import Field
from openenv.core.env_server import Action, Observation
from openenv.core.env_server.types import State


class PricingAction(Action):
    """
    Agent sets a new price for each SKU.
    prices: list of floats, one per SKU, in the same order as observation.skus.
    Each price must be > 0.
    """
    prices: List[float] = Field(default_factory=list)


@dataclass
class SKUInfo:
    """Per-SKU snapshot included in each observation."""
    sku_id: str
    current_price: float = field(default=0.0)
    competitor_price: float = field(default=0.0)
    inventory: int = field(default=0)
    sales_last_step: int = field(default=0)
    demand_signal: float = field(default=1.0)
    days_until_expiry: Optional[int] = field(default=None)


class PricingObservation(Observation):
    """
    Full market snapshot returned by reset() and step().
    """
    skus: List[SKUInfo] = Field(default_factory=list)
    step_number: int = 0
    episode_revenue: float = 0.0
    episode_units_sold: int = 0
    last_action_prices: List[float] = Field(default_factory=list)
    last_action_error: Optional[str] = None
    task_name: str = "single_sku_stable"
    done: bool = False
    reward: float = 0.0


@dataclass
class PricingState:
    """Internal episode state — returned by state() endpoint."""
    task_name: str = field(default="single_sku_stable")
    max_steps: int = field(default=30)
    num_skus: int = field(default=1)
    total_revenue: float = field(default=0.0)
    total_units_sold: int = field(default=0)
    stockout_count: int = field(default=0)
    step_count: int = field(default=0)
    overstock_penalty_total: float = field(default=0.0)