"""
DynamicPricingEnv client.
Wraps openenv EnvClient with typed Action/Observation.
"""

from openenv.core.env_client import EnvClient
from models import PricingAction, PricingObservation


class DynamicPricingEnv(EnvClient[PricingAction, PricingObservation]):
    action_cls = PricingAction
    observation_cls = PricingObservation
