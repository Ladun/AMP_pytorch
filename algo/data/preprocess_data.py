import numpy as np
import re
import xml.etree.ElementTree as ET

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
        joints[name] = {'axis': axis}

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
        if line.strip() == ":DEGREES":
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

def convert_amc_to_walker_state(motion_data, joint_mapping, humanoid_joints, walk_target_x, walk_target_y, initial_z=None):
    converted_frames = []
    humanoid_joint_order = get_humanoid_joint_order()
    
    for i, frame in enumerate(motion_data):
        if 'root' in frame:
            root_pos = np.array([frame['root'].get(axis, 0) for axis in ['tx', 'ty', 'tz']])
            root_rot = np.array([frame['root'].get(axis, 0) for axis in ['rx', 'ry', 'rz']])
        else:
            # root 데이터가 없는 경우 기본값 사용
            root_pos = np.zeros(3)
            root_rot = np.zeros(3)
        
        z = root_pos[2]
        if initial_z is None:
            initial_z = z
        r, p, yaw = root_rot
        
        walk_target_theta = np.arctan2(walk_target_y - root_pos[1], walk_target_x - root_pos[0])
        angle_to_target = walk_target_theta - yaw
        
        if i > 0:
            prev_pos = np.array([motion_data[i-1]['root'].get(axis, 0) for axis in ['tx', 'ty', 'tz']])
            velocity = (root_pos - prev_pos) / 0.0333  # Assuming 30 FPS
        else:
            velocity = np.zeros(3)
        
        vx, vy, vz = velocity
        
        more = np.array([
            z - initial_z,
            np.sin(angle_to_target),
            np.cos(angle_to_target),
            0.3 * vx,
            0.3 * vy,
            0.3 * vz,
            r,
            p
        ], dtype=np.float32)
        
        j = []
        for h_joint in humanoid_joint_order:
            asf_joint = joint_mapping.get(h_joint)
            if asf_joint and asf_joint in frame:
                data = frame[asf_joint]
                h_axis = humanoid_joints[h_joint]['axis']
                if h_axis:
                    # Project the ASF joint rotation onto the humanoid joint axis
                    rotation = np.array([data.get('rx', 0), data.get('ry', 0), data.get('rz', 0)])
                    projected_rotation = np.dot(rotation, h_axis)
                    j.append(projected_rotation)
                    
                    # Calculate angular speed
                    if i > 0:
                        prev_data = motion_data[i-1][asf_joint]
                        prev_rotation = np.array([prev_data.get('rx', 0), prev_data.get('ry', 0), prev_data.get('rz', 0)])
                        prev_projected_rotation = np.dot(prev_rotation, h_axis)
                        angular_speed = (projected_rotation - prev_projected_rotation) / 0.0333
                        j.append(angular_speed)
                    else:
                        j.append(0)
                else:
                    j.extend([0, 0])  # Position and velocity
            else:
                j.extend([0, 0])  # Position and velocity

        feet_contact = [0] * 2  # Assuming 2 feet (left and right)
        
        state = np.clip(np.concatenate([more, j, feet_contact]), -5, 5)
        
        converted_frames.append(state)
    
    return np.array(converted_frames)
    


if __name__ == "__main__":
    # 사용 예시
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
    walker_states = convert_amc_to_walker_state(motion_data[0:1], skeleton_data, joint_mapping, 
                                                walk_target_x, walk_target_y)
    
    print(walker_states.shape)