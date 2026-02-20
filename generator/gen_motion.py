import numpy as np
import torch
import pickle
import yaml
import re
from pathlib import Path

from diffusion.mdm import MDM
from anim.kin_char_model import KinCharModel
from anim.motion_lib import MotionLib
from diffusion.mdm_climb_motion_sampler import MDMClimbMotionSampler
import diffusion.mdm as mdm
import diffusion.gen_util as gen_util
from util.motion_util import MotionFrames


def load_mdm(mdm_path: Path) -> mdm.MDM:
    if not mdm_path.is_file():
        assert mdm_path.is_dir()

        # get checkpoint with biggest number
        number_file_pairs = []

        for file in mdm_path.iterdir():
            if file.is_file() and file.suffix == ".pkl":
                match = re.search(r'\d+', file.name)
                if match:
                    number = int(match.group())
                    number_file_pairs.append((number, file))

        if not number_file_pairs:
            return None

        # Return the file with the largest number (latest checkpoint)
        mdm_path = max(number_file_pairs, key=lambda x: x[0])[1]

    print("loading path:", mdm_path)
    ret_mdm: mdm.MDM = pickle.load(mdm_path.open("rb"))
    if ret_mdm.use_ema:
        print('Using EMA model...')
        ret_mdm._denoise_model = ret_mdm._ema_denoise_model
    ret_mdm.update_old_mdm()
    return ret_mdm


def generate_motion(
        mdm_model: MDM,
        mdm_settings: gen_util.MDMGenSettings,
        char_model: KinCharModel,
        hold_pos: torch.Tensor,
        start_root_pos: torch.Tensor,
        start_root_rot: torch.Tensor,
        init_frames: MotionFrames,
        dt: float,
        duration: float
    ):

    device = char_model._device
    total_frames = init_frames.get_copy(device)
    curr_time = dt

    while curr_time < duration:
        prev_frames = MotionFrames(
            root_pos=total_frames.root_pos[:, -2:],
            root_rot=total_frames.root_rot[:, -2:],
            joint_rot=total_frames.joint_rot[:, -2:],
            grips=total_frames.grips[:, -2:],
        )

        ret_frames = gen_util.gen_mdm_motion(
            hold_pos=hold_pos,
            start_root_pos=start_root_pos,
            start_root_rot=start_root_rot,
            prev_frames=prev_frames,
            mdm_model=mdm_model,
            char_model=char_model,
            mdm_settings=mdm_settings,
        )

        total_frames = MotionFrames(
            root_pos=torch.cat([total_frames.root_pos, ret_frames.root_pos[:, 2:]], dim=1),
            root_rot=torch.cat([total_frames.root_rot, ret_frames.root_rot[:, 2:]], dim=1),
            joint_rot=torch.cat([total_frames.joint_rot, ret_frames.joint_rot[:, 2:]], dim=1),
            grips=torch.cat([total_frames.grips, ret_frames.grips[:, 2:]], dim=1),
        )

        curr_time += (mdm_model.sequence_length() - mdm_model._num_prev_states) * dt

    return total_frames

def main():
    # mdm_model = load_mdm(Path(r'output/climb/checkpoints/model_1500.pkl'))
    # mdm_model = load_mdm(Path(r'output/climb/model_2500_nohold.pkl'))
    mdm_model = load_mdm(Path(r'output/climb/model_2500_linhold.pkl'))
    mdm_model.update_old_mdm()
    mdm_model._use_wandb = False

    char_model = mdm_model._kin_char_model.get_copy(mdm_model._device)

    mdm_settings = gen_util.MDMGenSettings()

    dt = 1 / mdm_model._sequence_fps

    motion_lib = MotionLib(r"data/motions/climb/motions_gen.yaml", char_model, char_model._device)
    motion_id = motion_lib.sample_motions(1)
    time0 = torch.zeros_like(motion_lib.sample_time(motion_id))
    time1 = time0 + dt

    hold_pos = motion_lib.get_motion_hold_pos(motion_id)

    # # random offset target holds
    # rand_offset = torch.rand((1, 3), device=mdm_model._device, dtype=torch.float32) * 0.3 - 0.15
    # hold_pos[:, -1] = hold_pos[:, -1] + rand_offset

    start_root_pos = motion_lib.get_motion_start_root_pos(motion_id).unsqueeze(1)
    start_root_rot = motion_lib.get_motion_start_root_rot(motion_id).unsqueeze(1)

    # get first two frames from motion_lib
    root_pos0, root_rot0, _, _, joint_rot0, _, grips0 = motion_lib.calc_motion_frame(motion_id, time0)
    root_pos1, root_rot1, _, _, joint_rot1, _, grips1 = motion_lib.calc_motion_frame(motion_id, time1)

    init_frames = MotionFrames(
        root_pos=torch.stack([root_pos0, root_pos1], dim=1),
        root_rot=torch.stack([root_rot0, root_rot1], dim=1),
        joint_rot=torch.stack([joint_rot0, joint_rot1], dim=1),
        grips=torch.stack([grips0, grips1], dim=1),
    )

    total_frames = generate_motion(
        mdm_model=mdm_model,
        mdm_settings=mdm_settings,
        char_model=char_model,
        hold_pos=hold_pos,
        start_root_pos=start_root_pos,
        start_root_rot=start_root_rot,
        init_frames=init_frames,
        dt=dt,
        duration=2.5
    )

    frames, grips = total_frames.get_mlib_format(char_model)

    # save the motion frames
    pkl_dict = {
        'fps': mdm_model._sequence_fps,
        'loop_mode': 'CLAMP',
        'frames': frames[0].cpu().numpy(),
        'motion_dir': np.zeros((3,)),
        'grips': grips[0].cpu().numpy(),
        'hold_pos': hold_pos[0].cpu().numpy(),
    }
    Path(r"output/climb/output_motion.pkl").write_bytes(pickle.dumps(pkl_dict))
    return


if __name__ == "__main__":
    main()
