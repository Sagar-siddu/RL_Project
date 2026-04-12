"""
FastAPI server for DynamicPricingEnv with multi-episode support.
"""

import os
import sys
import uuid
from typing import Optional, Dict, List

# Ensure project root is importable regardless of working directory
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openenv.core.env_server import create_app

from server.pricing_environment import DynamicPricingEnvironment
from models import PricingAction, PricingObservation

TASK_NAME = os.getenv("PRICING_TASK", "single_sku_stable")

# Global environment state storage - now supports multiple episodes
_env_instances: Dict[str, DynamicPricingEnvironment] = {}
_current_episode_id: str = "default"  # For backward compatibility

app: FastAPI = create_app(
    env=lambda: DynamicPricingEnvironment(task_name=TASK_NAME),
    action_cls=PricingAction,
    observation_cls=PricingObservation,
    max_concurrent_envs=100,  # Increased for load testing
)

# Request/Response models
class PriceAction(BaseModel):
    prices: List[float]

class StepRequest(BaseModel):
    action: PriceAction

class ResetResponse(BaseModel):
    episode_id: str
    observation: dict
    done: bool
    reward: float

# ==================== Multi-Episode Endpoints ====================

@app.post("/episode/create")
async def create_episode(task_name: Optional[str] = Query(None)) -> dict:
    """Create a new episode with unique ID and return the ID."""
    episode_id = str(uuid.uuid4())[:8]
    task = task_name or TASK_NAME
    
    try:
        env = DynamicPricingEnvironment(task_name=task)
        _env_instances[episode_id] = env
        obs = env.reset()
        obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else obs.__dict__
        
        return {
            "episode_id": episode_id,
            "task_name": task,
            "observation": obs_dict,
            "done": False,
            "reward": 0.0
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Episode creation failed: {str(e)}"}
        )

@app.post("/episode/{episode_id}/step")
async def episode_step(episode_id: str, request: StepRequest):
    """Execute step in a specific episode."""
    if episode_id not in _env_instances:
        return JSONResponse(
            status_code=404,
            content={"error": f"Episode '{episode_id}' not found"}
        )
    
    try:
        env = _env_instances[episode_id]
        action = PricingAction(prices=request.action.prices)
        obs = env.step(action)
        
        obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else obs.__dict__
        
        return {
            "episode_id": episode_id,
            "observation": obs_dict,
            "reward": float(obs.reward),
            "done": bool(obs.done)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Step failed: {str(e)}"}
        )

@app.get("/episode/{episode_id}/state")
async def episode_state(episode_id: str):
    """Get state of a specific episode."""
    if episode_id not in _env_instances:
        return JSONResponse(
            status_code=404,
            content={"error": f"Episode '{episode_id}' not found"}
        )
    
    try:
        env = _env_instances[episode_id]
        state = env.state
        state_dict = state.__dict__ if hasattr(state, '__dict__') else state
        
        return {
            "episode_id": episode_id,
            "state": state_dict
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"State retrieval failed: {str(e)}"}
        )

@app.post("/episode/{episode_id}/grade")
async def episode_grade(episode_id: str):
    """Grade a specific episode."""
    if episode_id not in _env_instances:
        return JSONResponse(
            status_code=404,
            content={"error": f"Episode '{episode_id}' not found"}
        )
    
    try:
        env = _env_instances[episode_id]
        grade = env.grade_episode()
        
        return {
            "episode_id": episode_id,
            "grade": float(grade),
            "message": "Episode graded successfully"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Grading failed: {str(e)}"}
        )

@app.delete("/episode/{episode_id}")
async def delete_episode(episode_id: str):
    """Delete a specific episode to free memory."""
    if episode_id not in _env_instances:
        return JSONResponse(
            status_code=404,
            content={"error": f"Episode '{episode_id}' not found"}
        )
    
    try:
        del _env_instances[episode_id]
        return {"message": f"Episode '{episode_id}' deleted successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Deletion failed: {str(e)}"}
        )

@app.get("/episodes/list")
async def list_episodes():
    """List all active episodes."""
    return {
        "total_episodes": len(_env_instances),
        "episode_ids": list(_env_instances.keys()),
        "episodes": {
            eid: {
                "task_name": _env_instances[eid]._task_name,
                "step_count": _env_instances[eid]._state.step_count,
                "total_revenue": _env_instances[eid]._state.total_revenue
            }
            for eid in _env_instances
        }
    }

# ==================== Backward Compatibility Endpoints ====================

@app.post("/custom/reset")
async def custom_reset():
    """Reset environment and maintain state for subsequent calls (backward compat)."""
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
    """Execute step with maintained state (backward compat)."""
    global _current_episode_id
    
    if _current_episode_id not in _env_instances:
        return JSONResponse(
            status_code=400,
            content={"error": "Environment not reset. Call /custom/reset first."}
        )
    
    try:
        env = _env_instances[_current_episode_id]
        action = PricingAction(prices=request.action.prices)
        obs = env.step(action)
        obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else obs.__dict__
        
        return {
            "episode_id": _current_episode_id,
            "observation": obs_dict,
            "reward": float(obs.reward),
            "done": bool(obs.done)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Step failed: {str(e)}"}
        )

@app.get("/custom/state")
async def custom_state():
    """Get current environment state (backward compat)."""
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
    """Grade the current episode (backward compat)."""
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

# ==================== Root & Info ====================

@app.get("/")
async def root():
    return JSONResponse({
        "message": "Dynamic Pricing Environment API",
        "status": "running",
        "default_task": TASK_NAME,
        "active_episodes": len(_env_instances),
        "docs": "Visit http://localhost:7860/docs for interactive API documentation",
        "endpoints": {
            "single_episode_mode": {
                "POST /custom/reset": "Reset default episode",
                "POST /custom/step": "Execute step on default episode",
                "GET /custom/state": "Get state of default episode",
                "POST /custom/grade": "Grade default episode"
            },
            "multi_episode_mode": {
                "POST /episode/create": "Create new episode with unique ID",
                "POST /episode/{id}/step": "Execute step on episode {id}",
                "GET /episode/{id}/state": "Get state of episode {id}",
                "POST /episode/{id}/grade": "Grade episode {id}",
                "DELETE /episode/{id}": "Delete episode {id}",
                "GET /episodes/list": "List all active episodes"
            }
        }
    })