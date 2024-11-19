import os
import numpy as np
from torch.utils.data import Dataset
import glob
import tqdm
import logging

from .preprocess_cmu_data import *
from .preprocess_deepmimic_data import *


logger = logging.getLogger(__name__)

class MotionDataset(Dataset):
    def __init__(self, motion_files, skeleton_file):
        self.motion_files = motion_files
        self.data = []
        self.labels = []

        skeleton_info = parse_skeleton_file(skeleton_file)
        for motion_path in motion_files:
            motion = parse_motion_file(motion_path, skeleton_info)
            
            self.data.append(motion)
            
    def __len__(self):
        l = 0
        for d in self.data:
            
            l += len(d) - 1
        return l

    def __getitem__(self, idx):
        cur = 0
        while idx >= len(self.data[cur]) - 1 and cur < len(self.data):
            cur += 1
            idx -= len(self.data[cur]) - 1
        
        state = self.data[cur][idx]
        next_state = self.data[cur][idx + 1]
        
        return state, next_state
    

# This code no longer used
class CMUMotionDataset(Dataset):
    def __init__(self, data_dirs, asf_dir):
        self.data_dirs = data_dirs
        self.data = []
        self.labels = []

        # Process and save data if no files are saved      
       
        skeleton_data = {}
        humanoid_joints = parse_humanoid_xml("custom_env/data/humanoid_symmetric.xml")                        
        
        # Load asf files
        asf_files = glob.glob(os.path.join(asf_dir, '*.asf'))
        for asf_file in asf_files:
            asf_data = parse_asf(asf_file)
            asf_file_name = os.path.split(asf_file)[1]
            asf_file_name = os.path.splitext(asf_file_name)[0]
            
            asf_joints = list(asf_data['bone_data'].keys())
            joint_mapping = create_joint_mapping(asf_joints, humanoid_joints)
            
            skeleton_data[asf_file_name] = {"asf_data": asf_data, "joint_mapping": joint_mapping}

        # Load amc files
        for data_dir in data_dirs:
            amc_files = glob.glob(os.path.join(data_dir, '**', '*.amc'), recursive=True)
            for amc_file in tqdm.tqdm(amc_files, desc=f"Load amc file from {data_dir}"):
                asf_file_name = os.path.split(amc_file)[1]
                asf_file_name = os.path.splitext(asf_file_name)[0]
                asf_file_name = asf_file_name.split("_")[0]
                
                if asf_file_name in skeleton_data:                
                    motion_data = parse_amc(amc_file, skeleton_data[asf_file_name]['asf_data'])
                    walker_states = convert_amc_to_walker_state(motion_data, skeleton_data[asf_file_name]['joint_mapping'], 
                                                                humanoid_joints, walk_target_x=1e3, walk_target_y=0)
                    self.data.append(walker_states)
                else:
                    logger.info(f"ASF file '{asf_file_name}.asf' doesn't exist")

        self.data = np.concatenate(self.data, axis=0, dtype=np.float32)
            
    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        state = self.data[idx]
        next_state = self.data[idx + 1]
        
        return state, next_state