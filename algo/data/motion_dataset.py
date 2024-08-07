import os
import numpy as np
from torch.utils.data import Dataset
import glob
import tqdm
import logging

from .preprocess_data import *


logger = logging.getLogger(__name__)

class MotionDataset(Dataset):
    def __init__(self, data_dir, asf_dir):
        self.data_dir = data_dir
        self.data = []
        self.labels = []

        # Check if there is a saved numpy file
        saved_file = os.path.join(data_dir, 'processed_motion_data.npz')
        if os.path.exists(saved_file):
            # If there is a saved file, load it
            logger.info(f"Load preprocessed file from '{saved_file}'")
            loaded_data = np.load(saved_file)
            self.data = loaded_data['data']
        else:
            # Process and save data if no files are saved      
            logger.info(f"Since there is no preprocessed file, preprocess the file and save it in '{saved_file}'.")
            
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
            amc_files = glob.glob(os.path.join(data_dir, '**/*.amc'))
            for amc_file in tqdm.tqdm(amc_files, desc="Load amc file"):
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

            np.savez(saved_file, data=self.data)
            
    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        state = self.data[idx]
        next_state = self.data[idx + 1]
        
        return state, next_state