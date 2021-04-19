import numpy as np
from torch.distributions.normal import Normal
from scripts import train
import ray.rllib.agents.ppo as ppo
import ray
import yaml
import torch

with open('configs/flow.yml', 'r') as file:
    config = yaml.safe_load(file)
config.update({'num_workers':0})

ray.init()
agent = ppo.PPOTrainer(config)
#agent.restore('flow_tests2/flows_fine_tune_kl_fix/PPO_HalfCheetahPyBulletEnv-v0_202a3_00000_0_2021-04-12_10-40-09/checkpoint_5/checkpoint-5')
policy = agent.workers.local_worker().get_policy()
emb = torch.zeros(1, 6).to(0)
base_dist = Normal(torch.zeros(1, 6, device=0), torch.ones(1, 6, device=0))
lops = []
N = 10000
for i in range(N):
    a = torch.from_numpy(policy.action_space.sample()).unsqueeze(0).to(0)
    z, log_abs_det = policy.model.flow(a, context=emb, inverse_mode=True)
    logp_z = base_dist.log_prob(z).sum(-1)
    lops.append(logp_z + log_abs_det)

volume = np.prod(policy.action_space.high - policy.action_space.low)
print(policy.action_space.high - policy.action_space.low)
print(torch.stack(lops).shape)
print(torch.exp(torch.logsumexp(torch.stack(lops), dim=0))*volume*(1/N))

insides = []
print(policy.action_space)
for i in range(N):
    z = base_dist.sample()
    a, _ = policy.model.flow(z, context=emb)
    insides.append(policy.action_space.contains(a.squeeze().cpu().detach().numpy()))
print(sum(insides)/N)
