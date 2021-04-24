
from ray.tune.registry import register_env
def cheetah_env_creator(env_config):
    import gym
    import pybulletgym
    env = gym.make("HalfCheetahPyBulletEnv-v0")
    env.reset()
    return env
register_env("HalfCheetahPyBulletEnv-v0", cheetah_env_creator)

def ant_env_creator(env_config):
    import gym
    import pybulletgym
    env = gym.make("AntPyBulletEnv-v0")
    env.reset()
    return env
register_env("AntPyBulletEnv-v0", ant_env_creator)

def ant_env_creator(env_config):
    import gym
    import pybulletgym
    env = gym.make("HumanoidPyBulletEnv-v0")
    env.reset()
    return env
register_env("HumanoidPyBulletEnv-v0", ant_env_creator)
