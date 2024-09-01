from dataclasses import dataclass
import tyro
from getkey import getkey
import time

from custom_env.env.humanoid_env import HumanoidEnv
from algo.data.preprocess_data import *

    
@dataclass
class Args:
    asf_file: str 
    amc_file: str
    
    

if __name__ == "__main__":
    args = tyro.cli(Args)

    skeleton_data = parse_asf(args.asf_file)
    motion_data = parse_amc(args.amc_file, skeleton_data)
    humanoid_joints = parse_humanoid_xml("custom_env/data/humanoid_symmetric.xml") 
       
    asf_joints = list(skeleton_data['bone_data'].keys())
    joint_mapping = create_joint_mapping(asf_joints, humanoid_joints)
        
    walker_states = convert_amc_to_walker_setting(motion_data, 
                                                  joint_mapping, 
                                                  humanoid_joints,
                                                  use_joint=[
                                                    'abdomen_z', 'abdomen_y', 'abdomen_x',
                                                    'right_hip_x', 'right_hip_z', 'right_hip_y', 'right_knee',
                                                    'left_hip_x', 'left_hip_z', 'left_hip_y', 'left_knee',
                                                    'right_shoulder1', 'right_shoulder2', 
                                                    'right_elbow',
                                                    'left_shoulder1', 'left_shoulder2', 
                                                    'left_elbow'
                                                 ])
    
    print(f" Total frame: {walker_states.shape[0]}")
    print(walker_states[0: 2])
    
    env = HumanoidEnv(env_config={"render_mode": "human"})
    env.reset()
    idx = 0
    
    try:
        env.reset()
        while True:
            #print(f"{idx} / {walker_states.shape[0]}")
            frame = walker_states[idx]
            state = env.set_robot_state(frame)
            
            img = env.render()
            
            key = getkey(blocking=False)            
            if key == 'a':
                break
            idx = (idx + 1) % walker_states.shape[0]
            
            time.sleep(0.02)
    except Exception as e:
        env.close()
        print(e)