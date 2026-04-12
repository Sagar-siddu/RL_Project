"""
Interactive Gradio UI for Dynamic Pricing Environment
Connects to FastAPI server for multi-episode pricing optimization
"""

import os
import gradio as gr
import requests
import json
from typing import Optional

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

class PricingDemo:
    def __init__(self):
        self.current_episode_id = None
        self.step_count = 0
        self.total_reward = 0.0
        self.history = []
    
    def create_episode(self, task_name: str):
        """Create a new episode"""
        try:
            response = requests.post(
                f"{API_URL}/episode/create",
                params={"task_name": task_name},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.current_episode_id = data["episode_id"]
                self.step_count = 0
                self.total_reward = 0.0
                self.history = []
                
                obs = data["observation"]
                status = f"✅ Episode created: {self.current_episode_id}\n\n"
                status += f"📊 Task: {data['task_name']}\n"
                status += f"💰 SKUs: {len(obs.get('prices', []))}\n"
                status += f"📈 Initial prices: {obs.get('prices', [])}\n"
                
                return status, json.dumps(obs, indent=2), ""
            else:
                error = response.json().get("error", "Unknown error")
                return f"❌ Error: {error}", "", ""
        except Exception as e:
            return f"❌ Connection error: {str(e)}", "", ""
    
    def execute_step(self, action_str: str):
        """Execute a step with given prices"""
        if not self.current_episode_id:
            return "❌ No active episode. Create episode first.", "", ""
        
        try:
            # Parse prices from input
            prices = [float(x.strip()) for x in action_str.split(",")]
            
            response = requests.post(
                f"{API_URL}/episode/{self.current_episode_id}/step",
                json={"action": {"prices": prices}},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                obs = data["observation"]
                reward = data["reward"]
                done = data["done"]
                
                self.step_count += 1
                self.total_reward += reward
                
                # Record in history
                self.history.append({
                    "step": self.step_count,
                    "prices": prices,
                    "reward": f"{reward:.4f}",
                    "revenue": f"${obs.get('revenue', 0):.2f}",
                    "inventory": obs.get('inventory', [0])[0] if obs.get('inventory') else 0
                })
                
                # Format response
                obs_display = f"Step {self.step_count}\n"
                obs_display += f"✅ Reward: {reward:.4f}\n"
                obs_display += f"💰 Revenue: ${obs.get('revenue', 0):.2f}\n"
                obs_display += f"📦 Inventory: {obs.get('inventory', [])}\n"
                obs_display += f"⏱️ Demand: {obs.get('demand', 0):.1f} units\n"
                
                if done:
                    obs_display += f"\n🏁 Episode complete at step {self.step_count}!"
                
                # Format history table
                history_text = f"\nTotal Reward: {self.total_reward:.4f}\n\n"
                history_text += "Recent History:\n"
                for entry in self.history[-5:]:
                    history_text += f"  Step {entry['step']}: {entry['prices']} → Reward {entry['reward']}\n"
                
                return obs_display, json.dumps(obs, indent=2), history_text
            else:
                error = response.json().get("error", "Unknown error")
                return f"❌ Error: {error}", "", ""
        
        except ValueError:
            return "❌ Invalid prices format. Use comma-separated numbers (e.g., '50.0, 60.5')", "", ""
        except Exception as e:
            return f"❌ Error: {str(e)}", "", ""
    
    def grade_episode(self):
        """Grade the current episode"""
        if not self.current_episode_id:
            return "❌ No active episode to grade."
        
        try:
            response = requests.post(
                f"{API_URL}/episode/{self.current_episode_id}/grade",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                grade = data["grade"]
                return f"📊 Grade: {grade:.4f}\n✅ {data['message']}\n\nSteps taken: {self.step_count}\nTotal reward: {self.total_reward:.4f}"
            else:
                error = response.json().get("error", "Unknown error")
                return f"❌ Error: {error}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

# Initialize demo
demo = PricingDemo()

# Build Gradio interface
with gr.Blocks(title="Dynamic Pricing RL Environment") as interface:
    gr.Markdown("""
    # 🏪 Dynamic Pricing RL Environment
    
    **Learn to price SKUs optimally across a simulated e-commerce market.**
    
    Set prices for SKUs and watch the market respond. Balance revenue, inventory, and competition.
    Competing agents will reprice reactively. Your goal: maximize cumulative reward!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Episode Setup")
            task_dropdown = gr.Dropdown(
                choices=["single_sku_stable", "multi_sku_competitors", "demand_shocks_perishables"],
                value="single_sku_stable",
                label="Task Type"
            )
            create_btn = gr.Button("🚀 Create Episode", variant="primary", scale=1)
        
        with gr.Column(scale=2):
            episode_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=4
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 💰 Price Action")
            prices_input = gr.Textbox(
                label="Prices (comma-separated)",
                placeholder="e.g., 50.0, 60.5, 75.0",
                lines=2
            )
            step_btn = gr.Button("→ Execute Step", variant="primary", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Observation")
            obs_output = gr.Textbox(
                label="Market State",
                interactive=False,
                lines=6
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            json_output = gr.Textbox(
                label="Raw JSON Observation",
                interactive=False,
                lines=8
            )
        
        with gr.Column(scale=1):
            history_output = gr.Textbox(
                label="Episode History",
                interactive=False,
                lines=8
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        grade_btn = gr.Button("📈 Grade Episode", variant="stop", scale=1)
        grade_output = gr.Textbox(
            label="Grade Result",
            interactive=False,
            lines=3
        )
    
    # Event handlers
    create_btn.click(
        fn=demo.create_episode,
        inputs=[task_dropdown],
        outputs=[episode_status, json_output, history_output]
    )
    
    step_btn.click(
        fn=demo.execute_step,
        inputs=[prices_input],
        outputs=[obs_output, json_output, history_output]
    )
    
    grade_btn.click(
        fn=demo.grade_episode,
        outputs=[grade_output]
    )

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
