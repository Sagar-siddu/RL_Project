#!/usr/bin/env python3
"""
Test client for the Dynamic Pricing Environment API.
Runs through a complete episode with intelligent pricing strategy.
"""

import httpx
import json
import sys
from typing import Any

# Configuration
API_BASE_URL = "http://127.0.0.1:7860"
TASK_NAME = "single_sku_stable"

class PricingEnvClient:
    """Client for interacting with the pricing environment API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)
        self.episode_step = 0
        self.total_revenue = 0.0
        self.cumulative_reward = 0.0
    
    def reset(self) -> dict[str, Any]:
        """Reset the environment and start a new episode."""
        print("\n" + "="*70)
        print("RESETTING ENVIRONMENT")
        print("="*70)
        
        response = self.client.post(f"{self.base_url}/reset")
        response.raise_for_status()
        data = response.json()
        
        self.episode_step = 0
        self.total_revenue = 0.0
        self.cumulative_reward = 0.0
        
        obs = data.get("observation", {})
        print(f"\n✓ Episode reset successfully")
        print(f"  Task: {obs.get('task_name', 'N/A')}")
        print(f"  Initial observation received")
        
        return data
    
    def step(self, prices: list[float]) -> dict[str, Any]:
        """Execute one step with the given prices."""
        self.episode_step += 1
        
        # OpenEnv expects just the action value (prices list)
        action_payload = {"value": prices}
        
        response = self.client.post(f"{self.base_url}/step", json=action_payload)
        response.raise_for_status()
        data = response.json()
        
        obs = data.get("observation", {})
        reward = data.get("reward", 0.0)
        done = data.get("done", False)
        
        self.cumulative_reward += reward if reward else 0.0
        
        # Track revenue from observation
        episode_revenue = obs.get("episode_revenue", 0.0)
        self.total_revenue = episode_revenue if episode_revenue else self.total_revenue
        
        # Print step info
        num_skus = len(obs.get("skus", []))
        print(f"\n[Step {self.episode_step}] Prices: {prices} → Reward: {reward:.4f}")
        
        if num_skus > 0:
            sku = obs["skus"][0]  # For single SKU task
            print(f"  └─ SKU {sku.get('sku_id')}: "
                  f"Price=${sku.get('current_price'):.2f}, "
                  f"Competitor=${sku.get('competitor_price'):.2f}, "
                  f"Inventory={sku.get('inventory')} units")
        
        print(f"  Cumulative Revenue: ${self.total_revenue:.2f}")
        print(f"  Episode Done: {done}")
        
        return data, done
    
    def grade(self) -> dict[str, Any]:
        """Grade the completed episode."""
        print("\n" + "="*70)
        print("GRADING EPISODE")
        print("="*70)
        
        response = self.client.post(f"{self.base_url}/grade")
        response.raise_for_status()
        data = response.json()
        
        grade = data.get("grade", 0.0)
        print(f"\n✓ Episode Grade: {grade:.4f}")
        print(f"  Total Steps: {self.episode_step}")
        print(f"  Total Revenue: ${self.total_revenue:.2f}")
        print(f"  Cumulative Reward: {self.cumulative_reward:.4f}")
        
        return data
    
    def run_episode(self, max_steps: int = 20, use_strategy: bool = True) -> float:
        """Run a complete episode.
        
        Args:
            max_steps: Maximum number of steps to run
            use_strategy: If True, use intelligent pricing strategy. Otherwise use random prices.
            
        Returns:
            Final episode grade
        """
        # Reset
        reset_data = self.reset()
        obs = reset_data.get("observation", {})
        
        # Get initial market info
        skus = obs.get("skus", [])
        num_skus = len(skus)
        
        print(f"\nStarting episode with {num_skus} product(s)")
        print(f"Max steps: {max_steps}")
        
        # Run steps
        done = False
        base_prices = [50.0] * num_skus  # Start at $50 for all products
        
        for step in range(max_steps):
            if done:
                print("\n⚠ Episode ended early (done=True)")
                break
            
            # Pricing strategy
            if use_strategy:
                # Simple strategy: adjust based on competitor prices
                prices = []
                for i, sku in enumerate(skus):
                    competitor_price = sku.get("competitor_price", base_prices[i])
                    # Undercut competitor by 5% but maintain minimum margin
                    adjusted_price = max(competitor_price * 0.95, 20.0)
                    # Add some randomness for exploration
                    import random
                    randomness = random.uniform(0.98, 1.02)
                    prices.append(adjusted_price * randomness)
            else:
                # Random prices between $20-$80
                import random
                prices = [random.uniform(20.0, 80.0) for _ in range(num_skus)]
            
            # Execute step
            step_data, done = self.step(prices)
            obs = step_data.get("observation", {})
            skus = obs.get("skus", [])
        
        # Grade episode
        grade_data = self.grade()
        final_grade = grade_data.get("grade", 0.0)
        
        print("\n" + "="*70)
        print("EPISODE COMPLETE")
        print("="*70)
        
        return final_grade


def main():
    """Main entry point."""
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "DYNAMIC PRICING ENVIRONMENT - TEST CLIENT".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        client = PricingEnvClient(API_BASE_URL)
        
        # Run episode with strategy
        print("\n🎯 Running intelligent pricing strategy...")
        final_grade = client.run_episode(max_steps=20, use_strategy=True)
        
        print("\n" + "="*70)
        print(f"✅ FINAL EPISODE GRADE: {final_grade:.4f}")
        print("="*70)
        
        # Summary
        print("\n📊 EPISODE SUMMARY:")
        print(f"  • Steps executed: {client.episode_step}/20")
        print(f"  • Total revenue: ${client.total_revenue:.2f}")
        print(f"  • Cumulative reward: {client.cumulative_reward:.4f}")
        print(f"  • Final grade: {final_grade:.4f}")
        
        if final_grade > 0.8:
            print("\n🌟 EXCELLENT performance! (Grade > 0.8)")
        elif final_grade > 0.6:
            print("\n✓ GOOD performance! (Grade > 0.6)")
        elif final_grade > 0.4:
            print("\n△ ACCEPTABLE performance (Grade > 0.4)")
        else:
            print("\n⚠ NEEDS IMPROVEMENT (Grade ≤ 0.4)")
        
        print("\n" + "="*70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        print("\n✓ Test client finished\n")


if __name__ == "__main__":
    sys.exit(main())
