
import gymnasium
from gymnasium.envs.registration import registry, make, spec

def register(id, *args, **kvargs):
    if id in registry:
        return
    else:
        return gymnasium.envs.registration.register(id, *args, **kvargs)

register(id='AntBulletEnv-v0',
         entry_point='env.env.ant_env:AntEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)