"""
Minimal Demonstration of Dynamic Pricing Environment Logic
============================================================

This standalone script demonstrates the core concepts WITHOUT external dependencies.
It shows:
1. Price elasticity demand model
2. Competitor repricing  
3. Step-by-step simulation
4. Reward computation
5. Episode grading

Run: python demo_environment.py
"""

import math
import random
from dataclasses import dataclass
from typing import List, Optional


# ============ CORE CLASSES ============

@dataclass
class SKU:
    """A product in inventory."""
    sku_id: str
    base_price: float
    current_price: float
    competitor_price: float
    inventory: int
    max_inventory: int
    base_demand: float
    elasticity: float
    demand_signal: float = 1.0
    sales_last_step: int = 0


class Market:
    """Simulates market dynamics."""
    
    def __init__(self, sku: SKU):
        self.sku = sku
        self.rng = random.Random(42)
        self.episode_revenue = 0.0
        self.episode_units_sold = 0
        self.step_number = 0
        self.max_steps = 20
        
    def compute_demand(self) -> float:
        """
        Log-linear demand model:
        demand = base_demand × (base_price / agent_price)^elasticity 
                 × demand_signal × competitor_effect
        """
        if self.sku.current_price <= 0:
            return 0.0
        
        # Own-price elasticity
        price_ratio = self.sku.base_price / self.sku.current_price
        own_effect = price_ratio ** abs(self.sku.elasticity)
        
        # Competitor effect (sigmoid-like)
        if self.sku.competitor_price > 0:
            relative = self.sku.current_price / self.sku.competitor_price
            # If agent price > competitor: lose customers (max 60% loss)
            competitor_effect = 1.0 / (1.0 + math.exp(3.0 * (relative - 1.0)))
            competitor_effect = 0.4 + 0.6 * competitor_effect
        else:
            competitor_effect = 1.0
        
        raw = (self.sku.base_demand * own_effect * 
               self.sku.demand_signal * competitor_effect)
        return max(0.0, raw)
    
    def sample_sales(self, expected_units: float) -> int:
        """Poisson-like sales sampling with variance."""
        if expected_units <= 0:
            return 0
        return self.rng.randint(
            max(0, int(expected_units * 0.7)),
            int(expected_units * 1.3) + 1
        )
    
    def competitor_reprice(self, agent_price: float) -> None:
        """Competitor bot reacts to agent pricing."""
        if agent_price < self.sku.competitor_price:
            # Agent undercut us - match + discount
            self.sku.competitor_price = agent_price * (1.0 - self.rng.uniform(0, 0.05))
        else:
            # Agent raised price - competitor undercuts slightly
            target = agent_price * (1.0 - self.rng.uniform(0.01, 0.05))
            self.sku.competitor_price = max(target, self.sku.base_price * 0.6)
    
    def compute_reward(self, units_sold: int, sales_revenue: float) -> float:
        """Shaped per-step reward [0.0, 1.0]."""
        base_reward = sales_revenue
        
        # Inventory health bonus: reward 20-80% stock level
        stock_ratio = self.sku.inventory / self.sku.max_inventory
        inventory_bonus = 0.0
        if 0.2 <= stock_ratio <= 0.8:
            inventory_bonus = 0.1 * self.sku.base_price
        
        # Penalties
        stockout_penalty = 0.0
        if self.sku.inventory == 0:
            stockout_penalty = 0.5 * self.sku.base_price
        
        overstock_penalty = 0.0
        if stock_ratio > 0.8:
            excess_ratio = (stock_ratio - 0.8) / 0.2
            overstock_penalty = 0.2 * self.sku.base_price * excess_ratio
        
        raw_reward = (base_reward + inventory_bonus - 
                      stockout_penalty - overstock_penalty)
        
        # Normalize by max possible revenue
        max_possible = self.sku.base_demand * self.sku.base_price
        normalized = raw_reward / max_possible if max_possible > 0 else 0.0
        
        return max(0.0, min(1.0, normalized))  # Clamp to [0,1]
    
    def step(self, agent_price: float) -> dict:
        """Execute one market step."""
        self.step_number += 1
        
        # 1. Update agent's price
        self.sku.current_price = agent_price
        
        # 2. Demand signal (seasonal + optional shocks)
        seasonal = 1.0 + 0.25 * math.sin(2 * math.pi * self.step_number / 20)
        self.sku.demand_signal = seasonal
        
        # 3. Compute demand
        expected_demand = self.compute_demand()
        
        # 4. Sample actual sales
        units_sold = self.sample_sales(expected_demand)
        
        # 5. Update inventory
        units_sold = min(units_sold, self.sku.inventory)  # Can't sell more than in stock
        self.sku.inventory -= units_sold
        self.sku.sales_last_step = units_sold
        
        # 6. Competitor reprice
        self.competitor_reprice(agent_price)
        
        # 7. Compute metrics
        sales_revenue = units_sold * agent_price
        self.episode_revenue += sales_revenue
        self.episode_units_sold += units_sold
        
        # 8. Compute reward
        reward = self.compute_reward(units_sold, sales_revenue)
        
        # 9. Check if done
        done = (self.step_number >= self.max_steps)
        
        return {
            "step": self.step_number,
            "agent_price": agent_price,
            "competitor_price": round(self.sku.competitor_price, 2),
            "demand_signal": round(self.sku.demand_signal, 3),
            "expected_demand": round(expected_demand, 2),
            "units_sold": units_sold,
            "sales_revenue": round(sales_revenue, 2),
            "inventory": self.sku.inventory,
            "episode_revenue": round(self.episode_revenue, 2),
            "episode_units_sold": self.episode_units_sold,
            "reward": round(reward, 4),
            "done": done,
        }
    
    def grade_episode(self) -> float:
        """Grade the episode [0.0, 1.0]."""
        # Theoretical max: sell at base price every step with no stockouts
        theoretical_max = self.sku.base_demand * self.max_steps * self.sku.base_price
        
        revenue_score = self.episode_revenue / theoretical_max if theoretical_max > 0 else 0.0
        revenue_score = min(1.0, revenue_score)
        
        # Stockout penalty (any stockout event = 0.1 per step)
        # (simplified for demo)
        grade = min(1.0, revenue_score)
        return round(grade, 4)


# ============ DEMO ============

def demo_task_1():
    """Task 1: Single SKU, stable demand, no competitor."""
    print("\n" + "="*70)
    print("TASK 1: single_sku_stable")
    print("="*70)
    print("Setup: 1 SKU, 20 steps, stable demand, no competitor")
    print()
    
    sku = SKU(
        sku_id="SKU-001",
        base_price=50.0,
        current_price=50.0,
        competitor_price=0.0,
        inventory=300,
        max_inventory=300,
        base_demand=10.0,
        elasticity=1.2,
    )
    
    market = Market(sku)
    
    print(f"Initial state:")
    print(f"  Base price: ${sku.base_price:.2f}")
    print(f"  Base demand: {sku.base_demand:.1f} units/step")
    print(f"  Inventory: {sku.inventory} units")
    print(f"  Elasticity: {sku.elasticity}")
    print(f"  Theoretical max revenue: ${sku.base_demand * 20 * sku.base_price:.2f}")
    print()
    
    # Simulate 5 steps with different pricing strategies
    print("SIMULATION (agent is learning pricing strategy):")
    print()
    
    prices = [52.50, 51.00, 50.00, 51.50, 50.50]  # Example agent pricing
    
    total_reward = 0.0
    for i, price in enumerate(prices):
        result = market.step(price)
        total_reward += result['reward']
        
        print(f"Step {result['step']}:")
        print(f"  Agent price:        ${result['agent_price']:.2f}")
        print(f"  Demand signal:      {result['demand_signal']:.3f}x (seasonal)")
        print(f"  Expected demand:    {result['expected_demand']:.2f} units")
        print(f"  Units sold:         {result['units_sold']} units")
        print(f"  Sales revenue:      ${result['sales_revenue']:.2f}")
        print(f"  Inventory:          {result['inventory']} units remaining")
        print(f"  Cumulative revenue: ${result['episode_revenue']:.2f}")
        print(f"  Step reward:        {result['reward']:.4f}")
        print()
    
    print("... (steps 6-20 continue)")
    print()
    
    # Fast-forward remaining steps with random pricing
    random.seed(42)
    for step in range(6, market.max_steps + 1):
        price = sku.base_price + random.uniform(-5, 5)
        market.step(price)
    
    grade = market.grade_episode()
    
    print(f"EPISODE SUMMARY:")
    print(f"  Steps completed:    {market.step_number}/{market.max_steps}")
    print(f"  Total revenue:      ${market.episode_revenue:.2f}")
    print(f"  Total units sold:   {market.episode_units_sold}")
    print(f"  Final inventory:    {market.sku.inventory}")
    print(f"  Episode grade:      {grade:.4f}")


def demo_task_2():
    """Task 2: Multi-SKU with competitors."""
    print("\n" + "="*70)
    print("TASK 2: multi_sku_competitors (SIMPLIFIED)")
    print("="*70)
    print("Setup: 2 SKUs, 10 steps (for demo), reactive competitors")
    print()
    
    skus = [
        SKU(
            sku_id="SKU-A",
            base_price=80.0,
            current_price=80.0,
            competitor_price=76.0,
            inventory=200,
            max_inventory=200,
            base_demand=8.0,
            elasticity=1.0,
        ),
        SKU(
            sku_id="SKU-B",
            base_price=35.0,
            current_price=35.0,
            competitor_price=33.0,
            inventory=400,
            max_inventory=400,
            base_demand=15.0,
            elasticity=1.5,
        ),
    ]
    
    print("SKU Setup:")
    for sku in skus:
        print(f"  {sku.sku_id}: base_price=${sku.base_price:.2f}, "
              f"elasticity={sku.elasticity}, competitor_price=${sku.competitor_price:.2f}")
    print()
    
    print("SIMULATION (agent balances pricing across portfolio):")
    print()
    
    # Example decisions: SKU-A premium, SKU-B aggressive
    actions = [
        {"skuA": 82.0, "skuB": 34.0},  # Premium A, aggressive B
        {"skuA": 81.0, "skuB": 33.5},  # Slight adjust
        {"skuA": 83.0, "skuB": 33.0},  # Go higher on A, lower on B
    ]
    
    total_revenue = 0.0
    for step_num, action in enumerate(actions, 1):
        print(f"Step {step_num}:")
        print(f"  Agent prices: SKU-A=${action['skuA']:.2f}, "
              f"SKU-B=${action['skuB']:.2f}")
        
        for sku_name, price in [("A", action['skuA']), ("B", action['skuB'])]:
            sku = skus[0] if sku_name == "A" else skus[1]
            sku.current_price = price
            
            expected = sku.base_demand * (sku.base_price / price) ** sku.elasticity
            units = max(0, int(expected + random.uniform(-2, 2)))
            units = min(units, sku.inventory)
            sku.inventory -= units
            
            revenue = units * price
            total_revenue += revenue
            
            print(f"    SKU-{sku_name}: {units} units × ${price:.2f} = ${revenue:.2f}")
        
        print(f"  Cumulative revenue: ${total_revenue:.2f}")
        print()
    
    print("... (steps 4-10 continue)")
    print()
    print(f"Episode would end with:")
    print(f"  Total revenue so far: ${total_revenue:.2f}")
    print(f"  Agent learned: Balance portfolio against competitor moves")


def demo_reward_function():
    """Show how reward is computed."""
    print("\n" + "="*70)
    print("REWARD FUNCTION DETAILS")
    print("="*70)
    print()
    
    print("Per-step reward factors:")
    print()
    print("1. REVENUE = units_sold × price")
    print("   Example: 8 units × $52.50 = $420.00")
    print()
    
    print("2. INVENTORY HEALTH BONUS = +0.1 × base_price (if 20% < stock% < 80%)")
    print("   Example: If inventory is 60 units out of 300 (20%), no bonus")
    print("           If inventory is 150 units out of 300 (50%), +$5.00 bonus")
    print()
    
    print("3. STOCKOUT PENALTY = -0.5 × base_price (when inventory = 0)")
    print("   Example: -$25.00 when stockout occurs")
    print()
    
    print("4. OVERSTOCK PENALTY = -0.2 × base_price × excess_ratio")
    print("   Example: If inventory is 270 out of 300 (90%), excess_ratio = (90-80)/20 = 0.5")
    print("           Penalty = -0.2 × $50 × 0.5 = -$5.00")
    print()
    
    print("FINAL REWARD = (Revenue + Bonus - Penalties) / max_possible_per_step")
    print("              Clamped to [0.0, 1.0]")
    print()
    
    print("Example calculation:")
    print("  Revenue: $420.00")
    print("  Inventory bonus: +$3.00 (healthy level)")
    print("  Stockout penalty: $0 (stock available)")
    print("  Overstock penalty: $0 (good level)")
    print("  Raw reward: $423.00")
    print("  Max possible: $500.00")
    print("  Normalized reward: 423 / 500 = 0.846")
    print()


if __name__ == "__main__":
    demo_task_1()
    demo_task_2()
    demo_reward_function()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print()
    print("Key takeaways:")
    print("  • Price elasticity creates demand trade-off: higher price → lower volume")
    print("  • Competitor bots create strategic tension: undercut vs. premium")
    print("  • Inventory health has value: not just revenue")
    print("  • Per-step rewards guide long-term learning (shaped rewards)")
    print()
    print("To run the full environment with all 3 tasks:")
    print("  1. Install deps: pip install openenv-core fastapi pydantic")
    print("  2. Run tests: python test_local.py")
    print("  3. Start server: python -m uvicorn server.app:app --reload")
    print()
