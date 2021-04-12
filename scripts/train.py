import ray
import rlflows
import random
import ray.rllib.agents.ppo as ppo
from argparse import ArgumentParser
from ray import tune
from ray.tune.schedulers.pb2 import PopulationBasedTraining
from ray.tune.registry import register_env
import yaml

# Based on https://docs.ray.io/en/master/tune/examples/pb2_ppo_example.html

def env_creator(env_config):
    import gym
    import pybulletgym
    env = gym.make("HalfCheetahPyBulletEnv-v0")
    env.reset()
    return env
register_env("HalfCheetahPyBulletEnv-v0", env_creator)

# Postprocess the perturbed config to ensure it's still valid used if PBT.
def explore(config):
    # Ensure we collect enough timesteps to do sgd.
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # Ensure we run at least one sgd iter.
    if config["lambda"] > 1:
        config["lambda"] = 1
    config["train_batch_size"] = int(config["train_batch_size"])
    return config

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--tune', action='store_true', type=bool)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--t_ready', type=int, default=50000)
    parser.add_argument('--perturb', type=float, default=0.25)
    args = parser.parse_args()


    with open("example.yaml", 'r') as file:
        config = yaml.safe_load(args.config)

    ray.init()

    pbt = PopulationBasedTraining(
        time_attr='timesteps_total',
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=args.t_ready,
        resample_probability=args.perturb,
        quantile_fraction=args.perturb,  # copy bottom % with top %
        # Specifies the search space for these hyperparams
        hyperparam_mutations={
            "clip_param": tune.uniform(0.1, 0.5),
            "lr": tune.loguniform(1e-3, 1e-5),
            "entropy_coeff": tune.loguniform(2e-4, 5e-1),
            "kl_target": tune.loguniform(0.01, 0.1)
        },
        custom_explore_fn=explore)

    tune.run(ppo.PPOTrainer,
        local_dir=args.logdir,
        name="flow_tunev1",
        num_samples=args.samples,
        stop={'episode_reward_mean': 650, "timesteps_total": 3e6}, # This is convergence for this version of half cheta
        config=config, 
        scheduler=pbt)
