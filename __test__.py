import gymnasium
import tqdm
import numpy as np
from numpngw import write_apng
from collections import deque
import glob
import os

import torch

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

import custom_env
from algo.utils.general import TimerManager
from algo.data.preprocess_data import (
    parse_amc, parse_asf, create_joint_mapping, parse_humanoid_xml,
    convert_amc_to_walker_state
)
from algo.data.motion_dataset import MotionDataset

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


# images = get_inference_video(240, 1)
# print(len(images))

env_test()

# motion_dataset = MotionDataset("data", "data/asf")

# data_preprocessing_test()