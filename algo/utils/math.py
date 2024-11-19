import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

    
    
# queternion => (w, x, y, z)

def quat_inverse(rot: np.array) -> np.array:
    
    assert len(rot) == 4, f"Wrong data recieved {rot}"
    return np.array((rot[0], -rot[1], -rot[2], -rot[3]))


def normalize_quaternion(quat: np.array) -> np.array:
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("Quaternion has zero norm and cannot be normalized.")
    return quat / norm   

def get_angular_vel(cur_rot : np.array, prev_rot: np.array, dur :float) -> np.array:    
    inv_prev_rot = prev_rot
    inv_prev_rot[1:4] *= -1
    deltaRot = quat_quat_multiply(cur_rot, inv_prev_rot)
    
    rad, axis = quat_to_angle_axis(deltaRot)
    
    angular_vel = axis * rad  * (1 / dur)
    return angular_vel

def quat_to_angle_axis(quat: np.array) -> Tuple[float, np.array]:
    
    theta = 2 * np.arccos(quat[0])
    if quat[0] > 1:
        print(quat[0], theta)
    
    if np.sin(theta / 2) != 0:
        axis = quat[1:4] / np.sin(theta / 2)
    else:
        axis = np.array((0, 0, 0))
    
    return theta, axis

    
def quat_quat_multiply(quat0: np.array, quat1: np.array) -> np.array:
    
    quat0 = normalize_quaternion(quat0)
    quat1 = normalize_quaternion(quat1)
    
    w0, v0 = quat0[0], quat0[1:] 
    w1, v1 = quat1[0], quat1[1:] 
    
    w = w0 * w1 - np.dot(v0, v1) 
    v = w0 * v1 + w1 * v0 + np.cross(v0, v1) 

    return normalize_quaternion(np.array([w, *v]))

def quat_vec_multiply(quat: np.array, vec: np.array) ->np.array:
    quat = normalize_quaternion(quat)
    num = quat[1] * 2
    num2 = quat[2] * 2
    num3 = quat[3] * 2
    num4 = quat[1] * num
    num5 = quat[2] * num2
    num6 = quat[3] * num3
    num7 = quat[1] * num2
    num8 = quat[1] * num3
    num9 = quat[2] * num3
    num10 = quat[0] * num
    num11 = quat[0] * num2
    num12 = quat[0] * num3
    
    return np.array(
        ((1 - (num5 + num6)) * vec[0] + (num7 - num12) * vec[1] + (num8 + num11) * vec[2],
         (num7 + num12) * vec[0] + (1 - (num4 + num6)) * vec[1] + (num9 - num10) * vec[2],
         (num8 - num11) * vec[0] + (num9 + num10) * vec[1] + (1 - (num4 + num5)) * vec[2])
    )