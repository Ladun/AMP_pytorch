import numpy as np
import re
import json

from typing import Any, Dict, List, Optional, Tuple, Union

from ..utils.math import *

DOFS = {
    "Humanoid": [
        [-1,  1], # duration of frame in seconds
        [ 0,  7], # root position(3D) +  root rotation(4D)
        [ 1,  4], # chest rotation
        [ 2,  4], # neck rotation
        [ 3,  4], # right hip rotation
        [ 4,  1], # right knee rotation
        [ 5,  4], # right ankel rotation
        [ 6,  4], # right shoulder rotation
        [ 7,  1], # right elbow rotation
        [ 9,  4], # left hip rotation
        [ 10, 1], # left knee rotation
        [ 11, 4], # left ankel rotation
        [ 12, 4], # left shoulder rotation
        [ 13, 1]  # left elbow rotation
    ]
}
    
def get_disc_motion_dim(skeleton_info):
    
    dofs = DOFS["Humanoid"]
    dim = 0    
    dim += 6
    dim += len(dofs[2:]) * 12
    dim += len( skeleton_info["end_effector_id"]) * 3
    
    return dim
    

def parse_skeleton_file(path) -> dict:
    
    with open(path, "r") as f:
        d = json.load(f)        
        
    joints = {v["ID"]: v for v in d["Skeleton"]["Joints"]}
    body = {v["ID"]: v for v in d["BodyDefs"]}
    shape = {v["ID"]: v for v in d["DrawShapeDefs"]}
    
    end_effector_id = [v["ID"] for v in d["Skeleton"]["Joints"] if v["IsEndEffector"] == 1]
    
    return {
        "joints": joints,
        "body": body,
        "shape": shape,
        "end_effector_id": end_effector_id
    }

def parse_motion_file(path, skeleton_info) -> np.array:
    with open(path, "r") as f:
        d = json.load(f)
        
    
    frames = d['Frames']
    all_frames = []
    
    dofs = DOFS["Humanoid"]
    # Load frame data for each joint
    for data in frames:
        frame = 0
        dofIdx = 0        
        
        frame_joints = {}
        while dofIdx < len(dofs):
            values = np.array(data[frame: frame + dofs[dofIdx][1]])
            
            frame_joints[dofs[dofIdx][0]] = values
            frame += dofs[dofIdx][1]
            dofIdx += 1
            
        all_frames.append(frame_joints)
    
    # Calculate discriminator observation
    observations = []
    
    all_positions = []
    for i in range(0, len(all_frames)):
        obs = []
        cur_joints = all_frames[i]
        positions = {-1: np.array((0, 0, 0))}
        rotations = {-1: np.array((1, 0, 0, 0))}
        
        if i == 0:
            # 1. root linear velocity
            obs.append(np.array((0, 0, 0)))
            # 2. root angular velocity
            obs.append(np.array((0, 0, 0)))
        else:
            prev_joints = all_frames[i - 1]            
            duration = prev_joints[-1]
            
            # 1. root linear velocity
            obs.append((cur_joints[0][:3] - prev_joints[0][:3]) /  duration)
            # 2. root angular velocity
            obs.append(get_angular_vel(cur_joints[0][3:7], prev_joints[0][3:7], duration))
        
        # root position and rotation
        positions[0] = calc_cur_pos(0, positions, rotations, skeleton_info)
        rotations[0] = get_rotation(cur_joints[0])
        for key, _ in dofs[2:]:
            cur_j = cur_joints[key]    
            
            # local rotation (normal, tangent vector)
            # identity quaternion
            rot = np.array((1, 0, 0, 0))
            p = key
            while p != -1:
                rot = quat_quat_multiply(get_rotation(cur_joints[p]), rot)
                p = skeleton_info["joints"][p]["Parent"]
            rotations[key] = rot
            
            # 3. normal vector
            obs.append(quat_vec_multiply(rot, np.array((0, -1, 0))))
            # 4. tangent vector
            obs.append(quat_vec_multiply(rot, np.array((1, 0, 0))))  

            # calc current position
            cur_pos = calc_cur_pos(key, positions, rotations, skeleton_info)
            positions[key] = cur_pos
            # velocity
            if i == 0:
                # 5. linear velocity
                obs.append(np.array((0, 0, 0)))
                # 6. angular velocity
                obs.append(np.array((0, 0, 0)))
            else:
                prev_joints = all_frames[i - 1]
                duration = prev_joints[-1] # prev_joints is not list, it is dictionary
                prev_pos = all_positions[i - 1][key]
                
                # Calculate velocity
                # 5. linear velocity
                obs.append((cur_pos - prev_pos) /  duration)
                # 6. angular velocity
                obs.append(get_angular_vel(get_rotation(cur_j), get_rotation(prev_joints[key]), duration))
        
        # 7. end effector position
        for key in skeleton_info["end_effector_id"]:
            obs.append(calc_cur_pos(key, positions, rotations, skeleton_info))        
        
        all_positions.append(positions)            
        observations.append(np.concatenate(obs))
    
    observations = np.stack(observations).astype(np.float32)
    return observations    
    
    
def calc_cur_pos(key, positions, rotations, skeleton_info):
    parent_pos = positions[skeleton_info["joints"][key]["Parent"]]
    parent_rot = rotations[skeleton_info["joints"][key]["Parent"]]
    cur_pos = parent_pos + quat_vec_multiply(parent_rot, np.array([skeleton_info["joints"][key][f"Attach{v}"] for v in ["X", "Y", "Z"]]))
    return cur_pos
            
    
    
def get_rotation(values: np.array) -> np.array:
    
    if len(values) == 4:
        return values
    if len(values) == 1:
        return np.array((0, 0, np.sin(values[0] / 2), np.cos(values[0] / 2)))
    
    return values[3:]
    
if __name__ == "__main__":
    
    skeleton_info = parse_skeleton_file("data/deepmimic/humanoid3d.txt")
    parsed_data = parse_motion_file("data/deepmimic/humanoid3d_run.txt", skeleton_info)
    
    print(parsed_data[1][:30])
    print(parsed_data.shape)
    
    #print(quaternion_multiply(np.array([-1, 1, 2, 3]), np.array([1, 2, 3, 4])))
    