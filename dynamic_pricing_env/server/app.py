"""
FastAPI server for DynamicPricingEnv.
"""

import os
import sys

# Ensure project root is importable regardless of working directory
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openenv.core.env_server import create_app

from server.pricing_environment import DynamicPricingEnvironment
from models import PricingAction, PricingObservation

TASK_NAME = os.getenv("PRICING_TASK", "single_sku_stable")

app: FastAPI = create_app(
    env=lambda: DynamicPricingEnvironment(task_name=TASK_NAME),
    action_cls=PricingAction,
    observation_cls=PricingObservation,
    max_concurrent_envs=10,
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