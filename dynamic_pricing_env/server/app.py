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
from openenv.core.env_server import create_app

from server.pricing_environment import DynamicPricingEnvironment
from models import PricingAction, PricingObservation

TASK_NAME = os.getenv("PRICING_TASK", "single_sku_stable")

app: FastAPI = create_app(
    env_class=DynamicPricingEnvironment,
    action_cls=PricingAction,
    observation_cls=PricingObservation,
    max_concurrent_envs=10,
    env_kwargs={"task_name": TASK_NAME},
)