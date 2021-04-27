import os
import ray
import rlflows
from argparse import ArgumentParser
from ray import tune
from ray.tune.schedulers.pb2 import PopulationBasedTraining
import yaml


# Based on https://docs.ray.io/en/master/tune/examples/pb2_ppo_example.html

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

def train():
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, help="name of the experiment")
    parser.add_argument('--logdir', type=str, help="directory to place checkpoints and tensorboard logs")
    parser.add_argument('--config', type=str, help="yaml config file see configs folder for examples")
    parser.add_argument('--stop_after_ts', default=1e7, help="maximum number of timesteps to train for")
    parser.add_argument('--keep_checkpoints_num', type=int, default=1, help="see tune docs")
    parser.add_argument('--checkpoint_freq', type=int, default=0, help="see tune docs")
    parser.add_argument('--resume', action='store_true', help="resume if run with the same name exists already in logdir")
    parser.add_argument('--pbt', action='store_true', help="enable population based training")
    parser.add_argument('--samples', type=int, default=10, help="number of runs to preform with this config")
    parser.add_argument('--t_ready', type=int, default=50000, help="how often to mutate in pbt")
    parser.add_argument('--perturb', type=float, default=0.25, help="portion of runs to preturb")
    args, _ = parser.parse_known_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

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
            "lr": tune.loguniform(1e-5, 5e-4),
            "entropy_coeff": tune.loguniform(2e-4, 5e-1),
            "kl_target": tune.loguniform(0.005, 0.05)
        },
        custom_explore_fn=explore)

    resume = args.resume
    if resume:
        experiment_path = os.path.join(args.logdir, args.name)
        if not os.path.exists(experiment_path):
            resume = False # first run don't try and resume

    tune.run(rlflows.CustomKLUpdatePPOTrainer,
        checkpoint_freq=args.checkpoint_freq,
        local_dir=args.logdir,
        name=args.name,
        resume=resume,
        num_samples=args.samples,
        stop={"timesteps_total": args.stop_after_ts},
        config=config,
        keep_checkpoints_num=args.keep_checkpoints_num,
        scheduler=(pbt if args.pbt else None))


if __name__ == '__main__':
    train()