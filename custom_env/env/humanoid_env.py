

from ..robot.humanoid import Humanoid
from .walker_base_env import WalkerBaseBulletEnv

class HumanoidEnv(WalkerBaseBulletEnv):
    def __init__(self, env_config={}):
        self.robot = Humanoid()
        super().__init__(self.robot, env_config=env_config)
        self.electricity_cost = 4.25 * self.electricity_cost
        self.stall_torque_cost = 4.25 * self.stall_torque_cost