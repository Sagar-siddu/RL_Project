"""
Dynamic Pricing Environment — Core Logic
=========================================
Implements the OpenEnv Environment interface:
  reset() -> PricingObservation
  step(action) -> PricingObservation
  state -> PricingState
"""

import random
import sys
import os
import uuid
from typing import List, Optional, Dict, Any

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import (
    PricingAction,
    PricingObservation,
    PricingState,
    SKUInfo,
)
from market_simulator import (
    SKUState,
    CompetitorBot,
    compute_demand,
    sample_sales,
    get_demand_signal,
    compute_step_reward,
)
from tasks import (
    TaskConfig,
    build_task,
    grade_task,
    TASK_REGISTRY,
)


class DynamicPricingEnvironment(Environment):
    """
    Multi-SKU dynamic pricing RL environment.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_name: str = "single_sku_stable"):
        super().__init__()
        if task_name not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task_name}'. Available: {list(TASK_REGISTRY.keys())}"
            )
        self._task_name = task_name
        self._task: Optional[TaskConfig] = None
        self._skus: List[SKUState] = []
        self._competitors: List[CompetitorBot] = []
        self._rng = random.Random()
        self._episode_log: Dict[str, Any] = {}
        # Construct PricingState with no positional surprises
        self._state = PricingState()
        self._state.task_name = task_name
        self._max_possible_revenue = 1.0
        self._shock_step_rewards: List[float] = []
        self._normal_step_rewards: List[float] = []
        self._fresh_units_sold: int = 0
        self._expiry_loss_units: int = 0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, task_name: Optional[str] = None) -> PricingObservation:
        if task_name and task_name in TASK_REGISTRY:
            self._task_name = task_name

        self._rng = random.Random(seed) if seed is not None else random.Random()

        self._task = build_task(self._task_name)
        self._skus = [self._clone_sku(s) for s in self._task.skus]
        self._competitors = self._task.competitors

        self._shock_step_rewards = []
        self._normal_step_rewards = []
        self._fresh_units_sold = 0
        self._expiry_loss_units = 0

        self._max_possible_revenue = max(
            1.0,
            sum(s.base_demand * s.base_price for s in self._skus) * self._task.max_steps
        )

        # Reset state by attribute assignment to avoid constructor arg issues
        self._state = PricingState()
        self._state.task_name = self._task_name
        self._state.max_steps = self._task.max_steps
        self._state.num_skus = len(self._skus)
        self._state.total_revenue = 0.0
        self._state.total_units_sold = 0
        self._state.stockout_count = 0
        self._state.overstock_penalty_total = 0.0
        # step_count is on the base State class — set it directly
        self._state.step_count = 0

        self._episode_log = {
            "total_revenue": 0.0,
            "total_units_sold": 0,
            "stockout_count": 0,
            "overstock_penalty_total": 0.0,
            "steps_taken": 0,
            "max_steps": self._task.max_steps,
            "rewards": [],
        }

        return self._build_observation(reward=0.0, done=False, error=None)

    def step(self, action: PricingAction) -> PricingObservation:
        if self._task is None:
            return self._build_observation(
                reward=0.0, done=True,
                error="Environment not reset. Call reset() first."
            )

        error = self._validate_action(action)
        if error:
            return self._build_observation(reward=0.0, done=False, error=error)

        prices = list(action.prices)

        if len(prices) < len(self._skus):
            prices += [s.current_price for s in self._skus[len(prices):]]
        prices = prices[:len(self._skus)]

        is_shock_step = False
        for sku in self._skus:
            sig = get_demand_signal(
                self._state.step_count,
                self._rng,
                shock_prob=self._task.shock_prob,
            )
            if sig > 1.4 or sig < 0.6:
                is_shock_step = True
            sku.demand_signal = sig

        competitor_map = {c.sku_id: c for c in self._competitors}
        for sku, price in zip(self._skus, prices):
            if sku.sku_id in competitor_map:
                sku.competitor_price = competitor_map[sku.sku_id].reprice(price)

        sales: List[int] = []
        stockouts: List[bool] = []
        expiry_losses: List[int] = []

        for sku, price in zip(self._skus, prices):
            expected = compute_demand(sku, price, sku.competitor_price)
            units = sample_sales(expected)
            units = min(units, sku.inventory)

            sku.inventory -= units
            sku.current_price = round(price, 2)
            sku.sales_last_step = units
            sales.append(units)
            stockouts.append(sku.inventory == 0)

            exp_loss = 0
            if sku.is_perishable and sku.days_until_expiry is not None:
                sku.days_until_expiry -= 1
                if sku.days_until_expiry <= 0:
                    exp_loss = sku.inventory
                    self._expiry_loss_units += exp_loss
                    sku.inventory = 0
                    sku.days_until_expiry = None
            expiry_losses.append(exp_loss)

            if sku.is_perishable:
                self._fresh_units_sold += units

        raw_reward, norm_reward = compute_step_reward(
            self._skus, sales, prices, stockouts, expiry_losses,
            max_possible_revenue=self._max_possible_revenue / self._task.max_steps,
        )

        if is_shock_step:
            self._shock_step_rewards.append(norm_reward)
        else:
            self._normal_step_rewards.append(norm_reward)

        step_revenue = sum(s * p for s, p in zip(sales, prices))
        self._state.total_revenue += step_revenue
        self._state.total_units_sold += sum(sales)
        self._state.stockout_count += sum(stockouts)
        self._state.overstock_penalty_total += sum(
            s.base_price * 0.2 * max(0, s.inventory / s.max_inventory - 0.9) * 10
            for s in self._skus
        )
        self._state.step_count += 1

        self._episode_log.update({
            "total_revenue": self._state.total_revenue,
            "total_units_sold": self._state.total_units_sold,
            "stockout_count": self._state.stockout_count,
            "overstock_penalty_total": self._state.overstock_penalty_total,
            "steps_taken": self._state.step_count,
            "fresh_units_sold": self._fresh_units_sold,
            "expiry_loss_units": self._expiry_loss_units,
            "shock_step_rewards": self._shock_step_rewards,
            "normal_step_rewards": self._normal_step_rewards,
        })
        self._episode_log["rewards"].append(norm_reward)

        done = self._state.step_count >= self._task.max_steps

        return self._build_observation(
            reward=norm_reward, done=done, error=None,
            last_prices=prices,
        )

    @property
    def state(self) -> PricingState:
        return self._state

    # ------------------------------------------------------------------
    # Episode grading
    # ------------------------------------------------------------------

    def grade_episode(self) -> float:
        return grade_task(self._task_name, self._episode_log)

    def get_episode_log(self) -> Dict[str, Any]:
        return dict(self._episode_log)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        reward: float,
        done: bool,
        error: Optional[str],
        last_prices: Optional[List[float]] = None,
    ) -> PricingObservation:
        sku_infos = [
            SKUInfo(
                sku_id=s.sku_id,
                current_price=round(s.current_price, 2),
                competitor_price=round(s.competitor_price, 2),
                inventory=s.inventory,
                sales_last_step=s.sales_last_step,
                demand_signal=s.demand_signal,
                days_until_expiry=s.days_until_expiry,
            )
            for s in self._skus
        ]
        return PricingObservation(
            skus=sku_infos,
            step_number=self._state.step_count,
            episode_revenue=self._state.total_revenue,
            episode_units_sold=self._state.total_units_sold,
            last_action_prices=last_prices or [],
            last_action_error=error,
            task_name=self._task_name,
            done=done,
            reward=reward,
        )

    def _validate_action(self, action: PricingAction) -> Optional[str]:
        if not action.prices:
            return "prices list is empty"
        for i, p in enumerate(action.prices):
            if p is None or p <= 0:
                return f"Price at index {i} must be > 0, got {p}"
            if p > 10000:
                return f"Price at index {i} exceeds maximum (10000)"
        return None

    @staticmethod
    def _clone_sku(s: SKUState) -> SKUState:
        from dataclasses import replace
        return replace(s)