import gymnasium
import tqdm
import numpy as np
from numpngw import write_apng
from collections import deque
import glob
import os
import json
import omegaconf 
import logging

import torch
from torch import nn
from torch.utils.data.sampler import WeightedRandomSampler

import gymnasium as gymte
from gymnasium.vector import AsyncVectorEnv

from algo.models.policy import Actor
from algo.models.discriminator import Discriminator
from algo.utils.general import TimerManager, set_seed, get_config, get_device
from algo.data.preprocess_cmu_data import (
    parse_amc, parse_asf, create_joint_mapping, parse_humanoid_xml,
    convert_amc_to_walker_state
)
from algo.data.preprocess_deepmimic_data import get_disc_motion_dim, parse_skeleton_file
from algo.data.motion_dataset import MotionDataset


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler()])


def get_inference_video(frame, total_time, vid_name="render.png"):
    timestep = 1. / frame
    images = []
    timer_manager = TimerManager()
    env =  gymnasium.make("HumanoidBulletEnv-v0", env_config={"render_mode":"rgb_array"})
    env.reset()

    try:
        env.reset()
        with timer_manager.get_timer("Total"):
            for i in tqdm.tqdm(range(frame * total_time)):
                with timer_manager.get_timer("one step"):
                    env.step(np.random.uniform(env.robot.action_space.low,
                                               env.robot.action_space.high,
                                               env.robot.action_space.shape))

                with timer_manager.get_timer("one render"):
                    img = env.render()
                images.append(img)
        with timer_manager.get_timer("create images"):
            write_apng(vid_name, images, delay=timestep * 1000)
    except Exception as e:
        env.close()
        print(e)
        
    for k, v in timer_manager.timers.items():
        print(f"\t\t {k} time: {v.get()} sec")

    return images

def env_test():
    from getkey import getkey, keys
    
    timer_manager = TimerManager()
    env =  gymnasium.make("HumanoidBulletEnv-v0", env_config={"render_mode":"human"})
    env.reset()
    
    print(" Information ==========================")
    print(env.robot.action_space.shape)
    print(env.observation_space)

    print(" =========== ==========================")
    try:
        env.reset()
        while True:
            with timer_manager.get_timer("one step"):
                s, r, term, trun, info = env.step(np.random.uniform(env.robot.action_space.low,
                                            env.robot.action_space.high,
                                            env.robot.action_space.shape))
                
            for name, joint in zip(env.robot.motor_names, env.robot.ordered_joints):
                print(f"{name}: {joint.current_relative_position()}")

            with timer_manager.get_timer("one render"):
                img = env.render()
                
                
            
            key = getkey(blocking=True)
            print(key)
    except Exception as e:
        env.close()
        print(e)
        
    for k, v in timer_manager.timers.items():
        print(f"\t\t {k} time: {v.get()} sec")

def data_preprocessing_test():
    asf_file = 'data/asf/02.asf'
    amc_file = 'data/run/02_03.amc'
    
    skeleton_data = parse_asf(asf_file)
    motion_data = parse_amc(amc_file, skeleton_data)
    humanoid_joints = parse_humanoid_xml("custom_env/data/humanoid_symmetric.xml")

    # 결과 출력 (예시)
    # print("Units:", skeleton_data['units'])
    # print("Root:", skeleton_data['root'])
    # print("Bones:", list(skeleton_data['bone_data'].keys()))
    # print("Hierarchy:", skeleton_data['hierarchy'])
    
    asf_joints = list(skeleton_data['bone_data'].keys())
    joint_mapping = create_joint_mapping(asf_joints, humanoid_joints)
    print("Humanoid_joints: ", len(humanoid_joints), humanoid_joints)
    print("asf_joints: ", len(asf_joints), asf_joints)
    print("joint_mapping: ", len(joint_mapping), joint_mapping)
    
    print(f"{'Mujoco':<20}{'ASF':<15}{'ASF Axis':<20}")
    for key, val in joint_mapping.items():
        print(f"{key:<20}{val:<15}{skeleton_data['bone_data'][val]['axis']:<20}")
    
    walk_target_x, walk_target_y = 1e3, 0  # 예시 목표 위치
    walker_states = convert_amc_to_walker_state(motion_data[0:3], joint_mapping, humanoid_joints, 
                                                walk_target_x, walk_target_y)
    
    with open("test_file.txt", 'w') as f:
        for i in range(walker_states.shape[0]):
            text = " ".join(str(round(v, 4)) for v in walker_states[i])
            f.write(f"{i} {text}\n")
    
    print(walker_states.shape)
    
def env_random_state_test():  
    def make_env(env_name):
        def _init():
            env = gym.make(env_name, env_config={"render_mode":"rgb_array"})
            return env
        return _init

    num_envs = 2
    for i in range(4):
        envs = AsyncVectorEnv([make_env("HumanoidBulletEnv-v0") for _ in range(num_envs)])
        
        envs.reset(seed=0)
        
        total_reward = np.zeros((num_envs,))
        for _ in range(5):
            s, r, term, trun, info = envs.step(np.zeros(envs.action_space.shape))
            total_reward += r
        print(total_reward)
        
def loss_test():
    
    loss_fn = nn.MSELoss(reduce=None, reduction='sum')
    set_seed(0)
    a = torch.randn(3, 5)
    b = torch.randn(3, 5)
    
    loss = loss_fn(a, b)
    
    print(loss.shape, loss, loss / 15)
    print( torch.square(a - b).sum())

def unity_env_test():
    
    import time
    from envs.unity_env import VertorizedUnityEnv
    from algo.replay_buffer import UnityReplayBuffer
    from algo.data.preprocess_deepmimic_data import DOFS
    
    np.set_printoptions(precision=6, suppress=True)
    skeleton_info = parse_skeleton_file("data/characters/humanoid3d.txt")
    
    def convert_data_in_disc_form(data):
        
        # shape of observastion from env
        # [[pos_i, nor_i, tan_i, lin_vel_i, ang_vel_i] for i in range(15)]
        def get_data_by_id(d, id):
            p = 5 * 3 * id
            return d[:, p: p + 5 * 3]
        
        obs = []
        end_effector_id = skeleton_info["end_effector_id"]
        dofs = DOFS["Humanoid"]
        
        # 1. root linear velocity
        obs.append(get_data_by_id(data, 0)[:, 9:12])
        # 2. root angular velocity
        obs.append(get_data_by_id(data, 0)[:, 12:15])
        
        for key, _ in dofs[2:]:
            # 3. normal vector
            obs.append(get_data_by_id(data, key)[:, 3:6])
            # 4. tangent vector
            obs.append(get_data_by_id(data, key)[:, 6:9])
            # 5. linear velocity
            obs.append(get_data_by_id(data, key)[:, 9:12])
            # 6. angular velocity
            obs.append(get_data_by_id(data, key)[:, 12:15]) 
            
        # 7. end effector position
        for key in end_effector_id:
            obs.append(get_data_by_id(data, key)[:, 0:3])   
        
        obs = torch.concat(obs, dim=1)
        return obs
    
    timer_manager = TimerManager()
    
    env = VertorizedUnityEnv(None, time_scale=0.8)    
    memory = UnityReplayBuffer(env.num_agents, 0.99, 0.95, torch.device('cpu'))
    
    state, _ = env.reset()
    
    action_space = env.action_space
    obs_space = env.observation_space
    
    print(f"Action: {action_space}")
    print(f"Obs: {obs_space}")
    print(f"Num of agent: {env.num_agents}")
    t = time.time()
    with timer_manager.get_timer("total collect trajectory"):        
        for i in range(10000):
            
            action = np.stack([env.action_space.sample() for _ in range(len(state))])
            with timer_manager.get_timer("one step"):        
                with timer_manager.get_timer("step"):     
                    next_state, reward, terminated, truncated, info = env.step(action)
      
                    if len(info['terminal_steps']) > 0:
                        agent_id =info['terminal_steps'].agent_id
                        ns, r, te, tr = info['terminal_state']
                        
                        memory.store(agent_id=info['terminal_steps'].agent_id,
                                    state=state[agent_id],
                                    action=action[agent_id],
                                    reward=r,
                                    value=np.ones_like(r),
                                    logprob=np.ones_like(r),
                                    done=te + tr)
                    
                    while len(info['decision_steps']) == 0:                
                        empty_ac = np.empty((0, *env.action_space.shape))   
                        next_state, reward, terminated, truncated, info = env.step(empty_ac)

                        if len(info['terminal_steps']) > 0:
                            agent_id =info['terminal_steps'].agent_id
                            ns, r, te, tr = info['terminal_state']
                            
                            memory.store(agent_id=info['terminal_steps'].agent_id,
                                        state=state[agent_id],
                                        action=action[agent_id],
                                        reward=r,
                                        value=np.ones_like(r),
                                        logprob=np.ones_like(r),
                                        done=te + tr)
                            
                    memory.store(agent_id=info['decision_steps'].agent_id,
                                state=state,
                                action=action,
                                reward=reward,
                                value=np.ones_like(reward),
                                logprob=np.ones_like(reward),
                                done=np.zeros_like(reward))
            
            t = time.time()
            done = terminated + truncated
        
    print(next_state.shape)
    print(reward.shape)
    print(done.shape)
        
    # for k,v in memory.temp_memory[0].items():
    #     shape = v[2].shape if v is not None else None
    #     value = v[2] if v is not None else None
    #     print(f"{k}, {shape}: {value}")
    print(f"Memory size: {len(memory)}")
    traj = memory.compute_gae_and_get(next_state, torch.tensor(np.ones_like(done)), done)
    
    print(traj.keys())
    print(traj['state'].shape)
    
    disc_s = convert_data_in_disc_form(traj['state'])
    print(disc_s.shape)
    
    for k, v in timer_manager.timers.items():
        print(f"\t {k} time: {round(v.clear(), 5)} sec")
        
    env.close()
    
def motion_dataset_test():
    from torch.utils.data import DataLoader
    
    config = get_config("configs/Humanoid_unity.yaml")
        
    motion_dataset = MotionDataset(config.train.gail.dataset_file, config.train.gail.skeleton_file)
    
    weight_sum = sum(motion_dataset.weights)
    
    print(len(motion_dataset.weights), weight_sum)
    print(set(motion_dataset.weights))
        
def actor_test_for_onnx():
    set_seed(0)
    
    device = get_device("cpu")
    config = get_config(os.path.join("checkpoints/Humanoid/exp20241120193131", "config.yaml"))   
    policy = Actor(config, device)    
    
    temp_x = torch.ones(4, config.env.state_dim, requires_grad=True)
    action, log_prob, ent = policy(temp_x)
    
    print(action, log_prob, ent)
    print(action.shape, log_prob.shape, ent.shape)
    

    
# images = get_inference_video(240, 1)
# print(len(images))

# env_test()

# motion_dataset = MotionDataset("data", "data/asf")

# data_preprocessing_test()

# env_random_state_test()

# loss_test()


# unity_env_test()

# motion_dataset_test()

# actor_test_for_onnx()


env_path = "AMP_Env/Builds/ArticulationBodyHeading/ArticulationBodyHeading"

if not (glob.glob(env_path) or glob.glob(env_path + ".*")):
    print("None")
cwd = os.getcwd()
launch_string = None
true_filename = os.path.basename(os.path.normpath(env_path))
candidates = glob.glob(os.path.join(cwd, env_path + ".exe"))
print(candidates)
if len(candidates) == 0:
    candidates = glob.glob(env_path + ".exe")
if len(candidates) == 0:
    # Look for e.g. 3DBall\UnityEnvironment.exe
    crash_handlers = set(
        glob.glob(os.path.join(cwd, env_path, "UnityCrashHandler*.exe"))
    )
    candidates = [
        c
        for c in glob.glob(os.path.join(cwd, env_path, "*.exe"))
        if c not in crash_handlers
    ]
if len(candidates) > 0:
    launch_string = candidates[0]