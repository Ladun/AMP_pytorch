
from ..robot.ant import Ant
from .walker_base_env import WalkerBaseBulletEnv

class AntEnv(WalkerBaseBulletEnv):
    def __init__(self, env_config={}):
        self.robot = Ant()
        super().__init__(self.robot, env_config=env_config)