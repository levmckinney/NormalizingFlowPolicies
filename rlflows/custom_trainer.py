from rlflows.custom_torch_ppo import CustomPPOTorchPolicy
from ray.rllib.agents.ppo.ppo import *



CustomKLUpdatePPOTrainer = build_trainer(
    name="PPO",
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    default_policy=CustomPPOTorchPolicy,
    execution_plan=execution_plan,
)