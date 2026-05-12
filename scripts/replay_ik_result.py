import mujoco
import mujoco
import mujoco.viewer
import numpy as np
import os
import math
from tqdm import tqdm
import json
from loop_rate_limiters import RateLimiter

def rotate_body_quat(body_quat: np.array, axis: str, angle: float):

    if axis not in ['x', 'y', 'z']:
        raise ValueError("error: axis must be 'x', 'y', or 'z'")
    
    qx = body_quat[0]
    qy = body_quat[1]
    qz = body_quat[2]
    qw = body_quat[3]
    q_initial = (qw, qx, qy, qz)
    
    half_angle = angle / 2.0
    sin_ha = math.sin(half_angle)
    cos_ha = math.cos(half_angle)
    
    if axis == 'x':
        q_rot = (cos_ha, sin_ha, 0.0, 0.0)
    elif axis == 'y':
        q_rot = (cos_ha, 0.0, sin_ha, 0.0)
    else:
        q_rot = (cos_ha, 0.0, 0.0, sin_ha)
    
    w1, x1, y1, z1 = q_rot
    w2, x2, y2, z2 = q_initial
    
    new_w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    new_x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    new_y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    new_z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    norm = math.sqrt(new_w**2 + new_x**2 + new_y**2 + new_z**2)
    if norm < 1e-12:
        raise ValueError("error: resulting quaternion has near-zero length")
    new_w /= norm
    new_x /= norm
    new_y /= norm
    new_z /= norm

    return np.array([new_x, new_y, new_z, new_w])

model_path = r"/home/xujinhao/cloned/myosuite/myosuite/envs/myo/assets/leg_soccer/myolegs_soccer.xml"
model_path = os.path.abspath(model_path)
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
# viewer = mujoco.viewer.launch_passive(model, data)
# Load motion data from npz file
motion_dir = r"/home/xujinhao/cloned/MyoChallenge2025-LNSGroup/challenge_wrapper/motion_data"

full_map_array = np.array([6, 7, 8, 12, 15, 16, 17, 21, 22, 23, 27, 30, 31, 32])
full_torso_index = np.array([36, 37, 38])     # FE, LB, AR
myo_map_array = np.array([32, 33, 34, 37, 40, 41, 42, 46, 47, 48, 51, 54, 55, 56])
myo_torso_index = np.array([14, 15, 16])      # FE, LB, AR

eq_obj1id = model.eq_obj1id
eq_obj2id = model.eq_obj2id
eq_data = model.eq_data

def query(joint2_value, polycoef):
    joint1_value = polycoef[0] + polycoef[1] * joint2_value + polycoef[2] * joint2_value ** 2 + polycoef[3] * joint2_value ** 3 + polycoef[4] * joint2_value ** 4
    return joint1_value

motion_file_list = [f for f in os.listdir(motion_dir) if f.endswith('.npz') and os.path.isfile(os.path.join(motion_dir, f))]
motion_file_list.sort()
print('All motion file list:', motion_file_list)
# load metadata.json
metadata_path = os.path.join(motion_dir, 'metadata.json')
with open(metadata_path, 'r') as f: 
    metadata = json.load(f)

for motion_file_base in motion_file_list:
    motion_file_path = os.path.join(motion_dir, motion_file_base)
    file_name = os.path.splitext(os.path.basename(motion_file_path))[0]
    traj_name = file_name[:-2]
    traj_side = file_name[-2:]
    # if traj_name != '74_05':
    #     continue
    
    # Find kick_point and kick_frame from metadata
    current_kick_point = None
    current_kick_frame = None
    for entry in metadata:
        if entry['file_name'] == traj_name:
            current_kick_point = entry['kick_point']
            current_kick_frame = entry['kick_frame']
            break
    
    if current_kick_point is None or current_kick_frame is None:
        raise ValueError(f"Metadata (kick_point or kick_frame) not found for file: {file_name}")

    # Reset the model to its initial keyframe state
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    # viewer.sync()
    
    # Get the initial position of the soccer ball in the simulation
    soccer_xpos_init = data.body('soccer_ball').xpos.copy()
    
    # Load the reference motion data
    motion_data = np.load(motion_file_path)
    qpos_traj_ref = motion_data["qpos_traj"]
    xpos_traj_ref = motion_data["xpos_traj"]
    xquat_traj_ref = motion_data["xquat_traj"]
    frame_rate = motion_data['framerate']
    print(f'Processing {motion_file_base}, framerate: {frame_rate}')
    rate = RateLimiter(frequency=frame_rate, warn=False)

    # Adjust kick_point for right foot if necessary (assuming metadata is for left foot or generic)
    if traj_side == '_r':
        current_kick_point[1] = -1 * current_kick_point[1]
        qpos_traj_ref[:, full_torso_index[1]] = -1 * qpos_traj_ref[:, full_torso_index[1]]
        qpos_traj_ref[:, full_torso_index[2]] = -1 * qpos_traj_ref[:, full_torso_index[2]]

    root_trans = np.zeros(3)
    root_trans[:2] = soccer_xpos_init[:2] - np.array(current_kick_point)[:2] - xpos_traj_ref[0][1][:2]

    start_frame = 0
    # Define the end frame for replay and recording: kick frame plus 0.5 seconds of motion
    end_frame = current_kick_frame + int(0.5 * frame_rate)
    
    # Initialize lists to store the full qpos and xpos of the simulated model
    # for each frame during the replay.
    recorded_qpos_traj = []
    recorded_xpos_traj = []

    # Replay the motion and record simulation data
    for i in tqdm(range(start_frame, end_frame), desc=f'Replaying and recording {motion_file_base}'):
        data.qpos[7:10] = xpos_traj_ref[i][1] + root_trans
        data.qpos[10:14] = xquat_traj_ref[i][1]
        data.qpos[myo_map_array] = qpos_traj_ref[i][full_map_array]
        data.qpos[myo_torso_index] = qpos_traj_ref[i][full_torso_index]
        for index in range(14, model.nq):
            jnt_index = index - 12
            if jnt_index in eq_obj1id:
                # the joint index in the eq_obj1id
                jnt1_eq_index = np.where(eq_obj1id==jnt_index)[0][0]
                jnt2_index = eq_obj2id[jnt1_eq_index]
                if jnt2_index > 0:
                    polycoef = eq_data[jnt1_eq_index]
                    data.qpos[index] = query(data.qpos[jnt2_index + 12], polycoef)
                else:
                    data.qpos[index] = 0

        mujoco.mj_forward(model, data)

        recorded_qpos_traj.append(data.qpos.copy())
        recorded_xpos_traj.append(data.xpos.copy())
        
        # viewer.sync() # Update the MuJoCo viewer
        rate.sleep() # Control the replay speed

    # Convert the lists of recorded data into numpy arrays
    recorded_qpos_traj = np.array(recorded_qpos_traj)
    recorded_xpos_traj = np.array(recorded_xpos_traj)

    # Define the output filename for the processed trajectory
    output_filename = f"{traj_name}{traj_side}_kick.npz"
    output_filepath = os.path.join(motion_dir, 'kick', output_filename)

    # Save the recorded simulation data to a new NPZ file
    np.savez(output_filepath, 
            qpos_traj=recorded_qpos_traj,
            xpos_traj=recorded_xpos_traj,
            framerate=frame_rate,
            kick_point=current_kick_point,
            kick_frame=current_kick_frame)
    print(f"Saved processed trajectory to {output_filepath}")

# viewer.close()

# save
# qpos_traj_cut = qpos_traj[start_frame:end_frame, :]
# xpos_traj_cut = xpos_traj[start_frame:end_frame, :]

# array_descriptions = {
#         "qpos_traj": "Trajectory of generalized positions (joint angles). Shape: (frames, nq)",
#         "xpos_traj": "Trajectory of body Cartesian positions. Shape: (frames, nbody, 3)",
#         "framerate": "Framerate of the motion in Hz.",
#     }

# output_path = os.path.join(os.path.dirname(__file__), "12_04_poses_cut_ik.npz")

# np.savez(output_path, 
#         qpos_traj=qpos_traj_cut,
#         xpos_traj=xpos_traj_cut,
#         framerate=frame_rate,
#         array_descriptions=array_descriptions)