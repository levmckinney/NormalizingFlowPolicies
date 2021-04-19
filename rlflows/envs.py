
from ray.tune.registry import register_env
def env_creator(env_config):
    import gym
    import pybulletgym
    env = gym.make("HalfCheetahPyBulletEnv-v0")
    env.reset()
    return env
register_env("HalfCheetahPyBulletEnv-v0", env_creator)
