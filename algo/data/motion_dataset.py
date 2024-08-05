import os
import numpy as np
from torch.utils.data import Dataset
import glob

from .preprocess_data import *

class MotionDataset(Dataset):
    def __init__(self, data_dir, asf_file, xml_file):
        self.data_dir = data_dir
        self.data = []
        self.labels = []

        # 저장된 numpy 파일이 있는지 확인
        saved_file = os.path.join(data_dir, 'processed_motion_data.npz')
        if os.path.exists(saved_file):
            # 저장된 파일이 있으면 로드
            loaded_data = np.load(saved_file)
            self.data = loaded_data['data']
            self.labels = loaded_data['labels']
        else:
            # 저장된 파일이 없으면 데이터 처리 및 저장
            skeleton_data = parse_asf(asf_file)
            humanoid_joints, _ = parse_humanoid_xml(xml_file)
            asf_joints = list(skeleton_data['bone_data'].keys())
            joint_mapping = create_joint_mapping(asf_joints, humanoid_joints)

            amc_files = glob.glob(os.path.join(data_dir, '*.amc'))
            for amc_file in amc_files:
                motion_data = parse_amc(amc_file, skeleton_data)
                walker_states = convert_amc_to_walker_state(motion_data, skeleton_data, joint_mapping, 
                                                            humanoid_joints, walk_target_x=1e3, walk_target_y=0)
                self.data.append(walker_states)

            self.data = np.concatenate(self.data, axis=0)

            # 처리된 데이터를 파일로 저장
            np.savez(saved_file, data=self.data)
            
    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        state = self.data[idx]
        next_state = self.data[idx + 1]
        
        return state, next_state