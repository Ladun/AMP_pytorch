
import pybullet
import numpy as np

from env.env.env_bases import BaseBulletEnv
from env.scene.stadium import StadiumScene

class WalkerBaseBulletEnv(BaseBulletEnv):
    def __init__(self, robot, env_config=None):
        super().__init__(robot, env_config)
        self.stateId = -1

        # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
        self.electricity_cost = -2.0
        # cost for running electric current through a motor even at zero rotational speed, small
        self.stall_torque_cost = -0.1
        # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
        self.foot_collision_cost = -1.0
        # to distinguish ground and other objects
        self.foot_ground_object_names = set(["floor"])
        # discourage stuck joints
        self.joints_at_limit_cost = -0.1

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = StadiumScene(bullet_client=bullet_client, gravity=9.8,
                                          timestep=0.0165/4, frame_skip=4)
        return self.stadium_scene

    def reset(self, seed=None, options=None):

        if self.stateId >= 0:
            self._p.restoreState(self.stateId)

        r = super().reset(seed=seed, options=options)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(self._p,
                                                                                             self.stadium_scene.ground_plane_mjcf)
        self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in
                               self.foot_ground_object_names])
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        if self.stateId < 0:
            self.stateId = self._p.saveState()

        return r

    def step(self, action):
        # if multiplayer, action first applied to all robots,
        # then global step() called, then _step() for all robots with the same actions
        if not self.scene.multiplayer:
            self.robot.apply_action(action)
            self.scene.global_step()

        state = self.robot.calc_state()

        # state[0] is body height above ground, body_rpy[1] is pitch
        alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[1]))
        terminated = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            terminated = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(self.robot.feet):
            contact_ids = set((x[2], x[4]) for x in f.contact_list())

            if self.ground_ids & contact_ids:
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost = self.electricity_cost * float(np.abs(action*self.robot.joint_speeds).mean())
        electricity_cost += self.stall_torque_cost * float(np.square(action).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)

        self.rewards = [
            alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
        ]

        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(terminated), False, {}

    def close(self):
        super().close()
        self.stateId = -1

    def camera_adjust(self):
        x, y, z = self.body_xyz
        self.camera.move_and_look_at(0, 0, 0, x, y, 1.0)
