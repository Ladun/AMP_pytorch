import numpy as np
import re
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R


# This code no longer used

def parse_humanoid_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    joints = {}

    for joint in root.findall(".//joint"):
        name = joint.get('name')
        axis = joint.get('axis')
        
        if not name:
            continue
        if axis:
            axis = [float(x) for x in axis.split()]
        
        joint_limit = None
        if 'range' in joint.attrib:
            range_str = joint.get('range')
            min_angle, max_angle = map(np.radians, map(float, range_str.split()))
            joint_limit = (min_angle, max_angle)
            
        joints[name] = {'axis': axis, 'joint_limit': joint_limit}

    return joints

def parse_asf(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # 섹션별로 내용 분리
    sections = re.split(r':(\w+)', content)[1:]
    sections = {sections[i]: sections[i + 1].strip() for i in range(0, len(sections), 2)}

    # root 섹션 파싱
    root_data = {}
    for line in sections['root'].split('\n'):
        if line.strip():
            key, value = line.strip().split(None, 1)
            root_data[key] = value

    # 뼈 데이터 파싱
    bone_data = {'root': root_data}  # root 데이터 추가
    for bone in re.findall(r'begin(.*?)end', sections['bonedata'], re.DOTALL):
        bone_info = {}
        lines = bone.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            parts = line.split(None, 1)
            key = parts[0]
            if key == 'name':
                name = parts[1]
            elif key == 'direction':
                bone_info[key] = np.array([float(x) for x in parts[1].split()])
            elif key == 'length':
                bone_info[key] = float(parts[1])
            elif key == 'axis':
                bone_info[key] = parts[1] 
            elif key == 'dof':
                bone_info[key] = parts[1].split()
            elif key == 'limits':
                bone_info['limits'] = []
                while i < len(lines) and '(' in lines[i]:
                    limits_line = lines[i].strip()
                    if "limits" in limits_line:
                        limits_line = limits_line[6:].strip()
                    bone_info['limits'].append([float(x) for x in limits_line.strip('()').split()])
                    i += 1
                i -= 1 
            else:
                bone_info[key] = parts[1] if len(parts) > 1 else None
            i += 1
        bone_data[name] = bone_info

    # 계층 구조 파싱
    hierarchy = {}
    hierarchy_section = sections['hierarchy'].split('\n')[1:-1]  # 첫 줄과 마지막 줄 제외
    for line in hierarchy_section:
        parent, *children = line.strip().split()
        hierarchy[parent] = children

    return {
        'units': sections['units'],
        'root': root_data,
        'bone_data': bone_data,
        'hierarchy': hierarchy
    }

def parse_amc(file_path, skeleton_data):
    with open(file_path, 'r') as f:
        content = f.readlines()
    
    frames = []
    current_frame = None
    bone_dofs = {name: bone_info.get('dof', []) for name, bone_info in skeleton_data['bone_data'].items()}
    
    start_line = 0
    
    for i, line in enumerate(content):
        line = line.strip()
            
        if line == ":DEGREES":
            start_line = i + 1
            break
    
    for line in content[start_line:]:
        line = line.strip()
        
        if not line:
            continue
        
        if line.isdigit():
            if current_frame:
                frames.append(current_frame)
            current_frame = {}
        else:
            parts = line.split()
            bone = parts[0]
            values = [float(x) for x in parts[1:]]
            
            if bone == 'root':
                # root 데이터 처리
                root_order = skeleton_data['root']['order'].split()
                current_frame['root'] = {k: v for k, v in zip(root_order, values)}
            elif bone in bone_dofs:
                # 다른 뼈들의 데이터 처리
                bone_motion = {}
                for dof, value in zip(bone_dofs[bone], values):
                    bone_motion[dof] = value
                current_frame[bone] = bone_motion
    
    if current_frame:
        frames.append(current_frame)
    
    return frames

def get_humanoid_joint_order():
    """
    humanoid_symmetric.xml 파일에 기반한 관절 순서를 반환합니다.
    """
    return [
        'abdomen_z', 'abdomen_y', 'abdomen_x',
        'right_hip_x', 'right_hip_z', 'right_hip_y', 'right_knee',
        'left_hip_x', 'left_hip_z', 'left_hip_y', 'left_knee',
        'right_shoulder1', 'right_shoulder2', 'right_elbow',
        'left_shoulder1', 'left_shoulder2', 'left_elbow'
    ]

def create_joint_mapping(asf_joints, humanoid_joints):
    mapping = {}
    for h_joint, _  in humanoid_joints.items():
        if h_joint in asf_joints:
            mapping[h_joint] = h_joint
        else:
            if 'abdomen' in h_joint:
                similar_joint = next((j for j in asf_joints if 'lowerback' in j.lower()), None)
            elif 'hip' in h_joint:
                side = 'l' if 'left' in h_joint else 'r'
                similar_joint = next((j for j in asf_joints if f'{side}femur' in j.lower()), None)
            elif 'knee' in h_joint:
                side = 'l' if 'left' in h_joint else 'r'
                similar_joint = next((j for j in asf_joints if f'{side}tibia' in j.lower()), None)
            elif 'shoulder' in h_joint:
                side = 'l' if 'left' in h_joint else 'r'
                similar_joint = next((j for j in asf_joints if f'{side}humerus' in j.lower()), None)
            elif 'elbow' in h_joint:
                side = 'l' if 'left' in h_joint else 'r'
                similar_joint = next((j for j in asf_joints if f'{side}radius' in j.lower()), None)
            else:
                similar_joint = None

            if similar_joint:
                mapping[h_joint] = similar_joint
            else:
                print(f"Warning: No matching joint found for {h_joint}")
    return mapping

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])

def estimate_foot_positions(root_pos, root_rot, joint_angles):
    def compute_foot_pos(hip_angles, knee_angle, side):
        hip_angles = np.radians(hip_angles)
        knee_angle = np.radians(knee_angle)
        
        hip_rotation = R.from_euler('xyz', hip_angles)
        thigh_vec = hip_rotation.apply([0, 0, -0.34])
        shin_vec = R.from_euler('y', knee_angle).apply([0, 0, -0.3])
        foot_vec = [0, 0, -0.15]
        
        foot_pos = thigh_vec + shin_vec + foot_vec
        foot_pos[1] *= side
        
        return foot_pos

    left_hip_angles = [joint_angles.get(f'left_hip_{axis}', 0) for axis in 'xyz']
    right_hip_angles = [joint_angles.get(f'right_hip_{axis}', 0) for axis in 'xyz']
    left_knee_angle = joint_angles.get('left_knee', 0)
    right_knee_angle = joint_angles.get('right_knee', 0)

    left_foot_rel = compute_foot_pos(left_hip_angles, left_knee_angle, 1)
    right_foot_rel = compute_foot_pos(right_hip_angles, right_knee_angle, -1)

    hip_offset = root_rot.apply([0, 0.1, -0.04])
    left_foot_pos = root_pos + root_rot.apply(left_foot_rel + [0, 0.1, -0.04])
    right_foot_pos = root_pos + root_rot.apply(right_foot_rel + [0, -0.1, -0.04])

    return left_foot_pos, right_foot_pos


def convert_amc_to_walker_state(motion_data, joint_mapping, humanoid_joints, walk_target_x, walk_target_y, initial_z=None):
    converted_frames = []
    humanoid_joint_order = get_humanoid_joint_order()
    
    for i, frame in enumerate(motion_data):
        if 'root' in frame:
            root_pos = np.array([frame['root'].get(axis, 0) for axis in ['TZ', 'TX', 'TY']])
            # scaling position
            root_pos = root_pos / 16 
            
            root_rot = R.from_euler('xyz', [frame['root'].get(axis, 0) for axis in ['RZ', 'RX', 'RY']], degrees=True)
        else:
            root_pos = np.zeros(3)
            root_rot = R.identity()
        
        z = root_pos[2]
        if initial_z is None:
            initial_z = z
        
        walk_target_theta = np.arctan2(walk_target_y - root_pos[1], walk_target_x - root_pos[0])
        angle_to_target = walk_target_theta - root_rot.as_euler('xyz')[2]        
                
        if i > 0:
            prev_pos = np.array([motion_data[i-1]['root'].get(axis, 0) for axis in ['TZ', 'TX', 'TY']])
            velocity = (root_pos - prev_pos)
        else:
            velocity = np.zeros(3)
        
        vx, vy, vz = velocity
        
        more = np.array([
            z - initial_z,
            np.sin(angle_to_target),
            np.cos(angle_to_target),
            vx,
            vy,
            vz,
            root_rot.as_euler('xyz')[0], 
            root_rot.as_euler('xyz')[1]
        ], dtype=np.float32)
        
        j = []
        joint_angles = {}
        for h_joint in humanoid_joint_order:
            asf_joint = joint_mapping.get(h_joint)    
            
            if asf_joint and asf_joint in frame:
                data = frame[asf_joint]
                h_axis = humanoid_joints[h_joint]['axis']
                
                if "elbow" in h_joint:
                    h_axis = [0, 1, 0]
                    
                if h_axis:
                    rotation = R.from_euler('xyz', [data.get('rz', 0), data.get('rx', 0), data.get('ry', 0)], degrees=True)
                    rot_vector = rotation.as_rotvec()
                    rad = np.dot(rot_vector, h_axis)                    
                    rad = np.clip(rad, humanoid_joints[h_joint]['joint_limit'][0],  humanoid_joints[h_joint]['joint_limit'][1])
                    
                    
                    # 각속도 계산
                    if i > 0:
                        prev_data = motion_data[i-1][asf_joint]
                        prev_rotation = R.from_euler('xyz', [prev_data.get('rz', 0), prev_data.get('rx', 0), prev_data.get('ry', 0)], degrees=True)
                        angular_speed = (rotation * prev_rotation.inv()).as_rotvec()
                        angular_speed_proj = np.dot(angular_speed, h_axis)
                    else:
                        angular_speed_proj = 0
                    
                    if "shoulder" in h_joint:
                        rad = -rad   
                    
                    j.extend([rad, angular_speed_proj])
                else:
                    j.extend([0, 0])
            else:
                j.extend([0, 0])

        left_foot_pos, right_foot_pos = estimate_foot_positions(root_pos, root_rot, joint_angles)
        feet_contact = [
            1.0 if left_foot_pos[2] < 0.02 else 0.0,
            1.0 if right_foot_pos[2] < 0.02 else 0.0
        ]
        
        state = np.concatenate([more, j, feet_contact])  
        state = np.clip(state, -5, 5)         
        converted_frames.append(state)
        
    return np.array(converted_frames)


def convert_amc_to_walker_debug(motion_data, joint_mapping, humanoid_joints, use_joint=None):
    converted_frames = []
    humanoid_joint_order = get_humanoid_joint_order()
    
    for i, frame in enumerate(motion_data):
        if 'root' in frame:
            root_pos = np.array([frame['root'].get(axis, 0) for axis in ['TZ', 'TX', 'TY']])
            root_rot = R.from_euler('xyz', [frame['root'].get(axis, 0) for axis in ['RZ', 'RX', 'RY']], degrees=True)
        else:
            root_pos = np.zeros(3)
            root_rot = R.identity()
                
        if i > 0:
            prev_pos = np.array([motion_data[i-1]['root'].get(axis, 0) for axis in ['TZ', 'TX', 'TY']])
            velocity = (root_pos - prev_pos)
        else:
            velocity = np.zeros(3)
        
        vx, vy, vz = velocity
        
        more = np.array([
            *(root_pos / 16),
            vx,
            vy,
            vz,
            *root_rot.as_euler('xyz')
        ], dtype=np.float32)
        
        j = []
        joint_angles = {}
        for h_joint in humanoid_joint_order:
            asf_joint = joint_mapping.get(h_joint)
            
            if use_joint:
                if h_joint not in use_joint:
                    j.extend([0, 0])
                    continue
            
            if asf_joint and asf_joint in frame:
                data = frame[asf_joint]
                h_axis = humanoid_joints[h_joint]['axis']
                
                if "elbow" in h_joint:
                    h_axis = [0, 1, 0]
                    
                if h_axis:
                    rotation = R.from_euler('xyz', [data.get('rz', 0), data.get('rx', 0), data.get('ry', 0)], degrees=True)
                    rot_vector = rotation.as_rotvec()
                    rad = np.dot(rot_vector, h_axis)                    
                    rad = np.clip(rad, humanoid_joints[h_joint]['joint_limit'][0],  humanoid_joints[h_joint]['joint_limit'][1])
                    
                    # 각속도 계산
                    if i > 0:
                        prev_data = motion_data[i-1][asf_joint]
                        prev_rotation = R.from_euler('xyz', [prev_data.get('rx', 0), prev_data.get('ry', 0), prev_data.get('rz', 0)], degrees=True)
                        angular_speed = (rotation * prev_rotation.inv()).as_rotvec()
                        angular_speed_proj = np.dot(angular_speed, h_axis)
                    else:
                        angular_speed_proj = 0
                    
                    if "shoulder" in h_joint:
                        rad = -rad   
                    
                    j.extend([rad, 0]) 
                else:
                    j.extend([0, 0])
            else:
                j.extend([0, 0])

        left_foot_pos, right_foot_pos = estimate_foot_positions(root_pos, root_rot, joint_angles)
        feet_contact = [
            1.0 if left_foot_pos[2] < 0.02 else 0.0,
            1.0 if right_foot_pos[2] < 0.02 else 0.0
        ]
        
        state = np.concatenate([more, j, feet_contact])  
        # state = np.clip(state, -5, 5)         
        converted_frames.append(state)
        
    return np.array(converted_frames)


if __name__ == "__main__":
                
    np.set_printoptions(suppress=True)
    # 사용 예시
    asf_file = 'data/asf/02.asf'
    amc_file = 'data/run/02_03.amc'
    test_frame = 1

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
    walker_states = convert_amc_to_walker_debug(motion_data[0:test_frame], joint_mapping, humanoid_joints)
    walker_states_2 = convert_amc_to_walker_state(motion_data[0:test_frame], joint_mapping, humanoid_joints, 0, 0, 0)
    
    print(f"Total frames: {len(motion_data)}")
    for i, frame in enumerate(motion_data[:test_frame]):
        print(f"\nFrame {i+1}: {walker_states.shape}")
        print(walker_states[i])
        humanoid_joint_order = get_humanoid_joint_order()
        
        print(f"root: {frame['root']} ")
        for j, h_joint in enumerate(humanoid_joint_order):
            asf_joint = joint_mapping.get(h_joint)
            if asf_joint and asf_joint in frame:
                rad = walker_states[i, 9 + j * 2]
                ang = np.degrees(rad)
                print(f"{h_joint}[{humanoid_joints[h_joint]['axis']}] -- {asf_joint}: {frame[asf_joint]}")
                print(f"\tdebug |{rad} {ang}")
                print(f"\tnormal|{rad} {ang}")
            else:
                print(asf_joint, frame.keys())