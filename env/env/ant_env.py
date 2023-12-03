
from env.robot.ant import Ant
from env.env.walker_base_env import WalkerBaseBulletEnv

class AntEnv(WalkerBaseBulletEnv):
    def __init__(self, env_config={}):
        self.robot = Ant()
        env_config['render_mode'] = 'rgb_array'
        super().__init__(self.robot, env_config=env_config)