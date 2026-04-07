"""
FastAPI server for DynamicPricingEnv.
"""

import os
import sys
from typing import Optional, Dict

# Ensure project root is importable regardless of working directory
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openenv.core.env_server import create_app

from server.pricing_environment import DynamicPricingEnvironment
from models import PricingAction, PricingObservation

TASK_NAME = os.getenv("PRICING_TASK", "single_sku_stable")

# Global environment state storage
_env_instances: Dict[str, DynamicPricingEnvironment] = {}
_current_episode_id: str = "default"

app: FastAPI = create_app(
    env=lambda: DynamicPricingEnvironment(task_name=TASK_NAME),
    action_cls=PricingAction,
    observation_cls=PricingObservation,
    max_concurrent_envs=10,
)

# Request/Response models
class PriceAction(BaseModel):
    prices: list[float]

class StepRequest(BaseModel):
    action: PriceAction

# Custom endpoints with proper state management
@app.post("/custom/reset")
async def custom_reset():
    """Reset environment and maintain state for subsequent calls."""
    global _current_episode_id
    _current_episode_id = "default"
    env = DynamicPricingEnvironment(task_name=TASK_NAME)
    _env_instances[_current_episode_id] = env
    
    try:
        obs = env.reset()
        obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else obs.__dict__
        return {
            "episode_id": _current_episode_id,
            "observation": obs_dict,
            "done": False,
            "reward": 0.0
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Reset failed: {str(e)}"}
        )

@app.post("/custom/step")
async def custom_step(request: StepRequest):
    """Execute step with maintained state."""
    global _current_episode_id
    
    # Get the environment instance
    if _current_episode_id not in _env_instances:
        return JSONResponse(
            status_code=400,
            content={"error": "Environment not reset. Call /custom/reset first."}
        )
    
    try:
        env = _env_instances[_current_episode_id]
        
        # Create action
        action = PricingAction(prices=request.action.prices)
        
        # Execute step (returns obs, reward, done, truncated, info)
        obs, reward, done, truncated, info = env.step(action)
        
        obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else obs.__dict__
        
        # Clean up if episode is done
        if done or truncated:
            del _env_instances[_current_episode_id]
        
        return {
            "episode_id": _current_episode_id,
            "observation": obs_dict,
            "reward": float(reward),
            "done": bool(done or truncated)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Step failed: {str(e)}"}
        )

@app.get("/custom/state")
async def custom_state():
    """Get current environment state."""
    if _current_episode_id not in _env_instances:
        return JSONResponse(
            status_code=400,
            content={"error": "Environment not initialized. Call /custom/reset first."}
        )
    
    try:
        env = _env_instances[_current_episode_id]
        state = env.state
        
        state_dict = state.__dict__ if hasattr(state, '__dict__') else state
        
        return {
            "episode_id": _current_episode_id,
            "state": state_dict
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"State retrieval failed: {str(e)}"}
        )

@app.post("/custom/grade")
async def custom_grade():
    """Grade the current episode."""
    if _current_episode_id not in _env_instances:
        return JSONResponse(
            status_code=400,
            content={"error": "No active episode to grade."}
        )
    
    try:
        env = _env_instances[_current_episode_id]
        grade = env.grade_episode()
        
        return {
            "episode_id": _current_episode_id,
            "grade": float(grade),
            "message": "Episode graded successfully"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Grading failed: {str(e)}"}
        )

# Add root endpoint
@app.get("/")
async def root():
    return JSONResponse({
        "message": "Dynamic Pricing Environment API",
        "status": "running",
        "task": TASK_NAME,
        "docs": "Visit http://localhost:7860/docs for interactive API documentation",
        "available_endpoints": {
            "POST /reset": "Reset the environment",
            "POST /step": "Execute one step in the environment",
            "GET /state": "Get current environment state",
            "POST /grade": "Grade the current episode",
            "GET /docs": "Interactive API documentation (Swagger UI)"
        }
    })