#edited char_model_visual.py for metric evaluation purposes

import time
import torch

from anim.kin_char_model import KinCharModel
from anim.motion_lib import MotionLib
from util.mujoco_visualizer import MujocoSimulator
import util.torch_util as torch_util

import numpy as np


def main():
    localize_motion = True
    char_model = KinCharModel("cpu")
    char_model.load_char_file(r"data/assets/humanoid.xml")

    # the motions.yaml contains a list of .pkl files
    # each .pkl file contains data of a motion clip
    # the .pkl file should contain dict_keys(['fps', 'loop_mode', 'frames', 'contacts'])
    # 'fps' is the motion fps
    # 'loop_mode' should just be the string 'CLAMP'
    # 'frames' is an array of shape (num_frames, 34); 34 = (root_pos) 3 + (root_rot) 3 + (dof_pos) 28
    # 'grips' is an array of shape (num_frames, 4); whether the limb is gripping (1) or not (0)
    #motion_lib = MotionLib(r"data/motions/climb/motions_test.yaml", char_model, char_model._device)
    motion_lib = MotionLib(r"data/motions/climb/motions_gen.yaml", char_model, char_model._device)
    #motion_lib = MotionLib(r"data/motions/climb/motions.yaml", char_model, char_model._device)

    motion_id = motion_lib.sample_motions(1)
    motion_max_time = motion_lib.get_motion_length(motion_id)
    motion_init_time = torch.zeros_like(motion_lib.sample_time(motion_id))
    motion_hold_pos = motion_lib.get_motion_hold_pos(motion_id)
    print(motion_hold_pos[0,4])

    start_root_pos = motion_lib.get_motion_start_root_pos(motion_id)
    start_root_rot = motion_lib.get_motion_start_root_rot(motion_id)
    start_heading_quat_inv = torch_util.calc_heading_quat_inv(start_root_rot)

    # motion_dt = 1 / motion_lib._motion_fps[motion_id]
    motion_dt = 1 / 120
    curr_time = motion_init_time

    count = 0
    dists = -1*np.ones((500,5))
    moved = False
    movedIdx = -1

    with MujocoSimulator(char_model, render_fps=60) as sim:

        if localize_motion:
            motion_hold_pos = torch_util.quat_rotate(start_heading_quat_inv.unsqueeze(1), motion_hold_pos - start_root_pos.unsqueeze(1))

        sim.set_holds_position(motion_hold_pos)

        while True:
            if not sim.viewer.is_running():
                break

            if not sim.paused:
                root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, grips =\
                    motion_lib.calc_motion_frame(motion_id, curr_time)

                if localize_motion:
                    root_pos = torch_util.quat_rotate(start_heading_quat_inv, root_pos - start_root_pos)
                    root_rot = torch_util.quat_multiply(start_heading_quat_inv, root_rot)

                sim.forward_kinematics(root_pos, root_rot, joint_rot)
                sim.update_grip_states(grips)
                sim.render()

                #get limb that's moving
                idx = torch.argwhere(grips == 0)
                if(idx.shape == (1,2)):
                    movedIdx = idx[0,1]
                    if(movedIdx == 0):
                        p = 6
                    elif(movedIdx == 1):
                        p = 9
                    elif(movedIdx == 2):
                        p = 12
                    elif(movedIdx == 3):
                        p = 15
                else:
                    if(movedIdx != -1):
                        moved = True

                #track limb distance to nearest grip
                if(movedIdx != 0 and moved == False):
                    dists[count, 0] = np.linalg.norm(motion_hold_pos[0,0].numpy()-sim._d.xpos[6])
                if(movedIdx != 1 and moved == False):
                    dists[count, 1] = np.linalg.norm(motion_hold_pos[0,1].numpy()-sim._d.xpos[9])
                if(movedIdx != 2 and moved == False):
                    dists[count, 2] = np.linalg.norm(motion_hold_pos[0,2].numpy()-sim._d.xpos[12])
                if(movedIdx != 3 and moved == False):
                    dists[count, 3] = np.linalg.norm(motion_hold_pos[0,3].numpy()-sim._d.xpos[15])
                if(moved == True):
                    dists[count, 4] = np.linalg.norm(motion_hold_pos[0,4].numpy()-sim._d.xpos[p])
                count += 1

                #pause animation after it reaches hold
                #threshold can be updated according to the curr_time when 
                #the hold is reached
                curr_time = torch.remainder(curr_time + motion_dt, motion_max_time)
                if(curr_time > 2.58):
                    print(motion_hold_pos[0,4])
                    print(sim._d.xpos[p])
                    print("Final dist ", np.linalg.norm(motion_hold_pos[0,4].numpy()-sim._d.xpos[p]))
                    print("Max dist ", (np.max(dists)))
                    acc = 0
                    for i in range(0, count):
                        for j in dists[0]:
                            if(j > -1):
                                acc += j
                    print("Mean dist ", acc/5)
                    print(count)
                    sim.paused = not sim.paused

                    #print(char_model.get_joint(5))
                    #print(char_model.get_joint(8))
                    #print(char_model.get_joint(11))
                    #print(char_model.get_joint(14))
                    #break


if __name__ == "__main__":
    main()
