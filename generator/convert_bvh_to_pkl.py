import bvhio
import numpy as np
import pickle
from pathlib import Path


def normalize_angle(x):
    return np.atan2(np.sin(x), np.cos(x))

def quat_to_axis_angle(q):
    min_theta = 1e-5
    qw, qx, qy, qz = (0, 1, 2, 3)   #index of (w, x, y, z) component in q

    sin_theta = np.sqrt(1 - q[qw] * q[qw])
    angle = 2 * np.acos(q[qw])
    angle = normalize_angle(angle)

    if np.abs(sin_theta) < min_theta:
        angle = 0
        axis = np.array([0, 0, 1])
    else:
        axis = q[[qx, qy, qz]] / sin_theta

    return axis, angle

def quat_to_exp_map(q):
    axis, angle = quat_to_axis_angle(q)
    return angle * axis

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    return np.clip(q, 0, 1)


if __name__ == "__main__":
    bvh_path = r"data/motions/bvh/climb1.bvh"
    save_path = r"data/motions/climb/climb1.pkl"

    bvh = bvhio.readAsBvh(bvh_path)
    bvh_joints = [x for x, _, _ in bvh.Root.layout()]
    bvh_joint_names = [x.Name for x in bvh_joints]

    mappings = {}
    def create_map(body_name, dof, scale=1):
        mappings[body_name] = {"mode": dof, "scale": scale}

    create_map("pelvis", 0)
    create_map("torso", 3)
    create_map("head", 3)
    create_map("right_upper_arm", 3)
    create_map("right_lower_arm", 1)
    create_map("right_hand", 0)
    create_map("left_upper_arm", 3)
    create_map("left_lower_arm", 1, -1)     # this dof needs to be flipped, idk why
    create_map("left_hand", 0)
    create_map("right_thigh", 3)
    create_map("right_shin", 1)
    create_map("right_foot", 3)
    create_map("left_thigh", 3)
    create_map("left_shin", 1)
    create_map("left_foot", 3)

    # get body and dof names
    body_names = []
    for jnt in bvh_joints:
        jnt_name = jnt.Name
        if jnt_name in mappings:
            body_names.append(jnt_name)


    # extract local rotaions and dof values
    body_rotations = []
    dof_positions = []

    num_frames = bvh.FrameCount
    fps = 1 / bvh.FrameTime
    for idx in range(num_frames):

        dof_positions.append([])
        body_rotations.append([])

        for j, joint in enumerate(bvh_joints):

            if joint.Name in mappings:
                frame_data = joint.Keyframes[idx]
                q = np.array(frame_data.Rotation.to_tuple())
                mode = mappings[joint.Name]["mode"]
                s = mappings[joint.Name]["scale"]

                if mode == 3:
                    x, y, z = quat_to_exp_map(q)
                    dof_positions[idx].append(x * s)
                    dof_positions[idx].append(y * s)
                    dof_positions[idx].append(z * s)
                elif mode == 1:
                    axis, y = quat_to_axis_angle(q)
                    dof_positions[idx].append(y * s)

                body_rotations[-1].append(q)

    dof_positions = np.array(dof_positions, dtype=np.float64)
    body_rotations = np.array(body_rotations, dtype=np.float64)

    # extract world positions
    root = bvhio.convertBvhToHierarchy(bvh.Root).loadRestPose(recursive=True)
    body_positions = []
    for idx in range(num_frames):
        body_positions.append([])
        for j, (joint, _, _) in enumerate(root.loadPose(idx).layout()):
            if joint.Name in mappings:
                frame_data = joint.getKeyframe(idx)
                p = np.array(joint.PositionWorld.to_tuple())
                body_positions[-1].append(p)
    body_positions = np.array(body_positions, dtype=np.float64)

    root_pos = body_positions[:, 0]
    root_rot = np.array([quat_to_exp_map(x[0]) for x in body_rotations])

    frames = np.concatenate([root_pos, root_rot, dof_positions], axis=-1)

    #extract motion direction
    first_position = root.loadPose(0).layout()[0][0].Position
    last_position = root.loadPose(num_frames-1).layout()[0][0].Position
    motion_dir = np.array((first_position - last_position)/np.linalg.norm(first_position - last_position))

    #get boolean grip states
    grips = np.ones((num_frames, 4))
    #manually extracting grips for now
    #0: right_hand, 1: left_hand, 2: right_foot, 3: left_foot
    grips[77:123, 2] = 0
    grip_idx = body_names.index('right_foot')

    #extract hold positions
    hold_pos = np.zeros((5, 3))
    frame_idx = 0
    body_idx = body_names.index('right_hand')
    right_hand_0 = body_positions[frame_idx, body_idx]
    hold_pos[0] = right_hand_0
    body_idx = body_names.index('left_hand')
    left_hand_0 = body_positions[frame_idx, body_idx]
    hold_pos[1] = left_hand_0
    body_idx = body_names.index('right_foot')
    right_foot_0 = body_positions[frame_idx, body_idx]
    hold_pos[2] = right_foot_0
    body_idx = body_names.index('left_foot')
    left_foot_0 = body_positions[frame_idx, body_idx]
    hold_pos[3] = left_foot_0
    #target hold
    frame_idx = num_frames-1
    grip_n = body_positions[frame_idx, grip_idx]
    hold_pos[4] = grip_n

    # save the arrays
    pkl_dict = {
        'fps': fps,
        'loop_mode': 'CLAMP',
        'frames': frames,
        'motion_dir': motion_dir,
        'grips': grips,
        'hold_pos': hold_pos,
    }
    Path(save_path).write_bytes(pickle.dumps(pkl_dict))
