
import os
import numpy as np

import gymnasium
import pybullet
from pybullet_utils import bullet_client

try:
    if os.environ["PYBULLET_EGL"]:
        import pkgutil
except:
    pass


class Camera():
    def __init__(self, dist=10, yaw=10, pitch=-20):
        self.dist = dist
        self.yaw = yaw
        self.pitch = pitch

    def move_and_look_at(self, i, j, k, x, y, z):
        lookat = [x, y, z]
        self._p.resetDebugVisualizerCamera(self.dist,
                                           self.yaw,
                                           self.pitch,
                                           [x, y, z])


class BaseBulletEnv(gymnasium.Env):

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self, robot, env_config=None):
        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.robot = robot

        # rendering option
        assert env_config is None or env_config['render_mode'] in self.metadata["render_modes"]
        self.render_mode = env_config['render_mode']
        self._cam = Camera()
        self._render_width = 640
        self._render_height = 320

    @property
    def observation_space(self):
        return self.robot.observation_space

    @property
    def action_space(self):
        return self.robot.action_space

    def configure(self, args):
        self.robot.args = args

    def create_single_player_scene(self, bullet_client):
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.np_random = self.np_random

        if self.physicsClientId < 0:
            self.ownsPhysicsClient = True

            if self.render_mode == 'human':
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = bullet_client.BulletClient()
            self._p.setTimeStep(10)
            self._p.resetSimulation()
            self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
            try:
                if os.environ["PYBULLET_EGL"]:
                    con_mode = self._p.getConnectionInfo()['connectionMethod']
                    if con_mode==self._p.DIRECT:
                        egl = pkgutil.get_loader('eglRenderer')
                        if (egl):
                            self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                        else:
                            self._p.loadPlugin("eglRendererPlugin")
            except:
                pass
            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)

        if self.scene is None:
            self.scene = self.create_single_player_scene(self._p)
        if not self.scene.multiplayer and self.ownsPhysicsClient:
            self.scene.episode_restart(self._p)

        self.robot.scene = self.scene

        self.frame = 0
        self.done = 0
        self.reward = 0
        dump = 0
        obs = self.robot.reset(self._p)
        self.potential = self.robot.calc_potential()
        return obs, {}

    def step(self, action):
        return [], 0, True, False, {}

    def render(self):
        if self.render_mode != "rgb_array":
            return np.array([])

        try:
            base_pos = self.robot.body_xyz
        except:
            base_pos = [0, 0, 0]

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam.dist,
            yaw=self._cam.yaw,
            pitch=self._cam.pitch,
            roll=0,
            upAxisIndex=2)

        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width)/self._render_height,
            nearVal=0.1, farVal=100.0)

        (_, _, px, _, _) = self._p.getCameraImage(
            width=self._render_width, height=self._render_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


    def close(self):
        if self.ownsPhysicsClient:
            if self.physicsClientId >= 0:
                self._p.disconnect()
        self.physicsClientId = -1


