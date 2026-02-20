import enum
import torch

import util.torch_util as torch_util
import util.geom_util as geom_util

from diffusion.motion_sampler import MotionSampler
from diffusion.diffusion_util import MDMFrameType, RelativeZStyle, TargetInfo
"""
MDMHeightfieldContactMotionSampler Overview:
This class samples motions from files that come attached with a special terrain class.
The local heightfield of a sampled motion field is extracted from the clip's global heightfield,
and augmented using motion-aware randomization.
"""

class HFAugmentationMode(enum.Enum):
    NOISE = 0
    MAXPOOL_AND_BOXES = 1
    NONE = 2

class MDMClimbMotionSampler(MotionSampler):
    def __init__(self, cfg):
        super().__init__(cfg)

        self._canonicalize_samples = cfg['features']['canonicalize_samples']
        frame_components = cfg['features']['frame_components']
        self._frame_components = []
        for comp in frame_components:
            comp = MDMFrameType[comp]
            self._frame_components.append(comp)

        self._num_rb = len(self._kin_char_model._body_names)

        if self._canonicalize_samples:
            print("samples will be canonicalized")

        self._fps = cfg['sequence_fps']
        self._timestep = 1.0 / self._fps

        # If we wish to train an autoregressive model, then we need to sample
        # from different starting frames
        self._random_start_times = cfg['autoregressive']
        if self._random_start_times:
            print("sampled start times will be random")

        self._sample_seq_time = cfg['sequence_duration']
        self._motion_times = torch.arange(start=0.0, end=self._sample_seq_time, step=self._timestep,
                                          dtype=torch.float32, device=self._device)
        self._seq_len = self._motion_times.shape[0]
        print("sample sequence length:", self._seq_len)

        self._num_prev_states = cfg['num_prev_states']
        self._canon_idx = cfg['num_prev_states'] - 1 # this being disconnected from the MDM trainer might lead to bugs in the future


        self._future_pos_noise_scale = cfg["future_pos_noise_scale"]

        # use this to generate heightmap observations from the box obs
        self._use_saved_heightmaps = cfg["use_saved_heightmaps"]

        self._relative_z_style = RelativeZStyle[cfg["relative_z_style"]]

        self._hf_augmentation_mode = HFAugmentationMode[cfg["hf_augmentation_mode"]]
        hmap_cfg = cfg["heightmap"]
        dx = hmap_cfg["horizontal_scale"]
        dy = dx
        self._dx = dx
        num_x_neg = hmap_cfg["local_grid"]["num_x_neg"]
        num_x_pos = hmap_cfg["local_grid"]["num_x_pos"]
        num_y_neg = hmap_cfg["local_grid"]["num_y_neg"]
        num_y_pos = hmap_cfg["local_grid"]["num_y_pos"]
        self._num_x_neg = num_x_neg
        self._num_x_pos = num_x_pos
        self._num_y_neg = num_y_neg
        self._num_y_pos = num_y_pos
        self._grid_min_point = torch.tensor([-num_x_neg, -num_y_neg], dtype=torch.float32, device=self._device) * dx
        grid_dim_x = num_x_neg + 1 + num_x_pos
        grid_dim_y = num_y_neg + 1 + num_y_pos
        self._grid_dim_x = grid_dim_x
        self._grid_dim_y = grid_dim_y
        self._grid_dims = torch.tensor([grid_dim_x, grid_dim_y], dtype=torch.int64, device=self._device)
        self._num_hf_points = grid_dim_x * grid_dim_y
        zero = torch.zeros(size=(2,), dtype=torch.float32, device=self._device)
        self._generic_heightmap = geom_util.get_xy_grid_points(zero, dx, dy, num_x_neg, num_x_pos, num_y_neg, num_y_pos)


        self._max_h = cfg["heightmap"]["max_h"]
        self._min_h = -self._max_h

        self._use_hf_augmentation = cfg["use_hf_augmentation"]
        if self._use_hf_augmentation:
            self._max_num_boxes = cfg["max_num_boxes"]
            self._box_min_len = cfg["box_min_len"]
            self._box_max_len = cfg["box_max_len"]
            self._hf_maxpool_chance = cfg["hf_maxpool_chance"]
            self._hf_max_maxpool_size = cfg["hf_max_maxpool_size"]
            self._hf_change_height_chance = cfg["hf_change_height_chance"]

        num_prev_states = cfg['num_prev_states']
        assert num_prev_states >= 1
        self._ref_frame_idx = num_prev_states - 1 # the state that the motions are observations are canonicalized wrt

        # for randomly perturbing motions
        self._angle_noise_scale = cfg["angle_noise_scale"]
        self._pos_noise_scale = cfg["pos_noise_scale"]

        # future window times
        self._future_window_min = cfg["future_window_min"]
        self._future_window_max = cfg["future_window_max"]

        self.check_init()
        return

    def update_old_sampler(self):
        self._relative_z_style = RelativeZStyle.RELATIVE_TO_ROOT
        return

    def get_motion_sequences_for_id(self, motion_id: int):
        motion_full_length = self._mlib.get_motion_length(motion_id)
        dt = 1.0 / self._fps
        seq_duration = self._seq_len * dt
        motion_start_times = torch.arange(start=0, end=motion_full_length - seq_duration, step=dt, dtype=torch.float32, device=self._device)
        motion_ids = torch.full_like(motion_start_times, motion_id, dtype=torch.int64)
        motion_data = self.sample_motion_data(motion_ids, motion_start_times, ret_hold_pos=False, ret_target_dir=False)
        return motion_data

    def _sample_motion_future_times(self, motion_ids, start_times,
                                    future_window_min, future_window_max):
        # Sample a random future time after the given start time.
        # Future window timeframe: [start_time + future_window_min, start_time + future_window_max]

        motion_full_lengths = self._mlib.get_motion_length(motion_ids)

        remaining_time = motion_full_lengths - start_times
        max_window_dur = future_window_max - future_window_min
        remaining_time = torch.clamp_max(remaining_time, max_window_dur)

        future_times = torch.rand_like(start_times) * remaining_time + start_times + future_window_min
        return future_times

    def _sample_target_info(self, motion_ids, motion_start_times, canon_pos, canon_heading_quat_inv):
        motion_future_times = self._sample_motion_future_times(motion_ids,
                                                               motion_start_times,
                                                               self._future_window_min,
                                                               self._future_window_max)

        future_pos, future_rot = self._extract_root_pos_and_rot(motion_ids, motion_future_times)

        future_pos += self._future_pos_noise_scale * torch.randn_like(future_pos)

        future_pos = future_pos - canon_pos
        future_pos = torch_util.quat_rotate(canon_heading_quat_inv, future_pos)
        future_rot = torch_util.quat_multiply(canon_heading_quat_inv, future_rot)
        target_info = TargetInfo(future_pos, future_rot)
        return target_info

    def _extract_motion_data(self, motion_ids, motion_start_times, motion_times, seq_len):
        # This function returns all motion data in global coordinates.

        num_samples = motion_ids.shape[0]

        # 2) Put all frames we want to sample into flat tensors of size (num_samples x num_frames)
        # (this is a motion_lib requirement)
        motion_times = motion_start_times.unsqueeze(-1) + motion_times
        motion_times = motion_times.flatten()
        repeated_motion_ids = motion_ids.unsqueeze(-1).expand(-1, seq_len).flatten()

        # 3) Get motion data from motion_lib
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, grips = self._mlib.calc_motion_frame(repeated_motion_ids, motion_times=motion_times)

        # 4) Reshape motion data
        # motion data is currently in the form (num_samples x num_frames) x (num_dof)
        # We reshape it into (num_samples) x (num_frames) x (num_dof)

        root_pos = torch.reshape(root_pos, shape=[num_samples, seq_len, 3])
        root_rot = torch.reshape(root_rot, shape=[num_samples, seq_len, 4])
        root_vel = torch.reshape(root_vel, shape=[num_samples, seq_len, 3])
        root_ang_vel = torch.reshape(root_ang_vel, shape=[num_samples, seq_len, 3])
        joint_rot = torch.reshape(joint_rot, shape=[num_samples, seq_len, -1, 4])
        dof_vel = torch.reshape(dof_vel, shape=[num_samples, seq_len, -1])
        grips = torch.reshape(grips, shape=[num_samples, seq_len, 4])

        return root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, grips

    def _extract_root_pos_and_rot(self, motion_ids, times):
        # TODO: make more efficient
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = self._mlib.calc_motion_frame(motion_ids, motion_times=times)
        return root_pos, root_rot

    def sample_motion_data(self,
                           motion_ids=None,
                           num_samples = None,
                           motion_start_times=None,
                           ret_hold_pos=True,
                           ret_target_dir=True):
        if motion_ids == None:
            assert(num_samples != None)
            motion_ids = self._mlib.sample_motions(num_samples)
        else:
            num_samples = motion_ids.shape[0]

        if motion_start_times is None:
            motion_start_times = self._sample_motion_start_times(motion_ids, self._sample_seq_time)

        ##### GET THE MOTION FRAME DATA FROM MOTION LIB ######
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, grips =\
            self._extract_motion_data(motion_ids, motion_start_times, self._motion_times, self._seq_len)

        # Put motions relative to first frame of motion clip
        start_root_pos = self._mlib.get_motion_start_root_pos(motion_ids).unsqueeze(1)
        start_root_rot = self._mlib.get_motion_start_root_rot(motion_ids).unsqueeze(1)
        start_heading_quat_inv = torch_util.calc_heading_quat_inv(start_root_rot)

        root_pos = torch_util.quat_rotate(start_heading_quat_inv, root_pos - start_root_pos)
        root_rot = torch_util.quat_multiply(start_heading_quat_inv, root_rot)

        body_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
        body_pos = body_pos[..., 1:, :]     # ignore the root body pos, since that is the root position

        # get hold_pos
        if ret_hold_pos:
            hold_pos = self._mlib.get_motion_hold_pos(motion_ids)
            hold_pos = torch_util.quat_rotate(start_heading_quat_inv, hold_pos - start_root_pos)

        # canon_heading_quat_inv = torch_util.calc_heading_quat_inv(root_rot[:, 0, :])
        # target_info = self._sample_target_info(motion_ids, motion_start_times, start_root_pos.squeeze(1), canon_heading_quat_inv)

        motion_ret = dict()

        for frame_type in self._frame_components:
            if frame_type == MDMFrameType.ROOT_POS:
                motion_ret[MDMFrameType.ROOT_POS] = root_pos
            if frame_type == MDMFrameType.ROOT_ROT:
                motion_ret[MDMFrameType.ROOT_ROT] = root_rot
            if frame_type == MDMFrameType.JOINT_POS:
                motion_ret[MDMFrameType.JOINT_POS] = body_pos
            if frame_type == MDMFrameType.JOINT_ROT:
                motion_ret[MDMFrameType.JOINT_ROT] = joint_rot
            if frame_type == MDMFrameType.GRIPS:
                motion_ret[MDMFrameType.GRIPS] = grips

        if ret_hold_pos or ret_target_dir:
            ret = [motion_ret]
            if ret_hold_pos:
                ret.append(hold_pos)
            if ret_target_dir:
                target_dir = torch.zeros((motion_ids.shape[0], 3), device=self._device, dtype=torch.float32)
                ret.append(target_dir)
            return tuple(ret)

        else:
            return motion_ret
