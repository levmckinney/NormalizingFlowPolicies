import numpy as np
from torch.distributions.normal import Normal
from scripts import train
import ray.rllib.agents.ppo as ppo
import ray
import yaml
import torch

with open('configs/gmm5.yml', 'r') as file:
    config = yaml.safe_load(file)
config.update({'num_workers':0})

ray.init()
agent = ppo.PPOTrainer(config)
agent.restore('ray_results/gaussian_test/PPO_HalfCheetahPyBulletEnv-v0_2ea6c_00000_0_2021-04-12_15-26-53/checkpoint_32/checkpoint-32')
policy = agent.workers.local_worker().get_policy()

volume = np.prod(policy.action_space.high - policy.action_space.low)
print(policy.action_space.high - policy.action_space.low)
N=10000
insides = []
print(policy.action_space)
obs = policy.observation_space.sample()
for i in range(N):
    a = agent.compute_action(obs)
    print(a)
    insides.append(policy.action_space.contains(a))
print(sum(insides)/N)
