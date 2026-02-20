import time
import torch
import numpy as np
import mujoco
import mujoco.viewer

from anim.kin_char_model import KinCharModel


class MujocoSimulator():

    def __init__(self, char_model: KinCharModel, render_fps=None):
        self._m = mujoco.MjModel.from_xml_path(r'data/assets/blank.xml')
        self._d = mujoco.MjData(self._m)
        self._render_fps = render_fps
        self._visualize_grips = True
        self._visualize_links = True
        self._visualize_holds = True
        self._visualize_climber = True
        self.paused = False

        self._char_model = char_model
        self._body_names = self._char_model._body_names
        self._num_bodies = len(self._body_names)
        self._body_indices_map = {body: idx for idx, body in enumerate(self._body_names)}
        self._parent_indices = self._char_model._parent_indices

    def __enter__(self):
        self.viewer = mujoco.viewer.launch_passive(self._m, self._d, key_callback=self._key_callback, show_left_ui=False, show_right_ui=False)
        self.user_scn = self.viewer.user_scn
        self._create_visualizations()
        self._init_camera()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.viewer is not None:
            self.viewer.close()

    def _init_camera(self):
        self.viewer.cam.azimuth = 0
        self.viewer.cam.elevation = 0
        self.viewer.cam.distance = 2.5
        self.viewer.cam.lookat[:] = 0

    def _key_callback(self, keycode):
        if chr(keycode) == ' ':
            self.paused = not self.paused
        elif chr(keycode) == 'G' or chr(keycode) == 'g':
            self._visualize_grips = not self._visualize_grips
        elif chr(keycode) == 'L' or chr(keycode) == 'l':
            self._visualize_links = not self._visualize_links
            self._toggle_link_visualizations()
        elif chr(keycode) == 'H' or chr(keycode) == 'h':
            self._visualize_holds = not self._visualize_holds
            self._toggle_holds_visualizations()
        elif chr(keycode) == 'C' or chr(keycode) == 'c':
            self._visualize_climber = not self._visualize_climber

    def _create_visualizations(self):
        # create link points
        self._link_pos_ptr = self.user_scn.ngeom
        for idx in range(self._num_bodies):
            mujoco.mjv_initGeom(
                self.user_scn.geoms[self.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=np.array([0, 0, 0]), # Initial position
                size=[0.03, 0, 0],
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 0.0, 1.0, 1.0])
            )
            self.user_scn.ngeom += 1

        # create link lines
        self._link_line_ptr = self.user_scn.ngeom
        for idx in range(self._num_bodies):
            if self._parent_indices[idx] >= 0:
                mujoco.mjv_initGeom(
                    self.user_scn.geoms[self.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    pos=np.array([0, 0, 0]), # Initial position
                    size=[0.0, 0, 0],
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0.0, 0.0, 1.0, 1.0])
                )
                self.user_scn.ngeom += 1

        # create holds
        self._hold_pos_ptr = self.user_scn.ngeom
        for idx in range(5):
            mujoco.mjv_initGeom(
                self.user_scn.geoms[self.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=np.array([0, 0, 0]), # Initial position
                size=[0.05, 0, 0],
                mat=np.eye(3).flatten(),
                rgba=np.array([0.0, 1.0, 0.0, 0.75])
            )
            self.user_scn.ngeom += 1

    def _toggle_link_visualizations(self):
        a = 1.0 if self._visualize_links else 0.0
        for i in range(self._num_bodies):
            geom_idx = self._link_pos_ptr + i
            self.user_scn.geoms[geom_idx].rgba[:] = np.array([1.0, 0.0, 1.0, a])

        for i in range(self._num_bodies - 1):
            geom_idx = self._link_line_ptr + i
            self.user_scn.geoms[geom_idx].rgba[:] = np.array([0.0, 0.0, 1.0, a])

    def _update_link_visualizations(self, link_pos: torch.Tensor):
        link_pos = link_pos.squeeze(0).cpu().numpy()
        for i in range(self._num_bodies):
            geom_idx = self._link_pos_ptr + i
            self.user_scn.geoms[geom_idx].pos[:] = link_pos[i]

        for i in range(self._num_bodies - 1):
            geom_idx = self._link_line_ptr + i
            mujoco.mjv_connector(
                self.user_scn.geoms[geom_idx],
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                width=0.02,
                from_=link_pos[i+1],
                to=link_pos[self._parent_indices[i+1]]
            )

    def _toggle_holds_visualizations(self):
        a = 0.75 if self._visualize_holds else 0.0
        for i in range(5):
            geom_idx = self._hold_pos_ptr + i
            self.user_scn.geoms[geom_idx].rgba[:] = np.array([0.0, 1.0, 0.0, a])

    def set_holds_position(self, hold_pos: torch.Tensor):
        hold_pos = hold_pos.squeeze(0).cpu().numpy()
        for i in range(5):
            geom_idx = self._hold_pos_ptr + i
            self.user_scn.geoms[geom_idx].pos[:] = hold_pos[i]

    def forward_kinematics(self, root_pos: torch.Tensor, root_rot: torch.Tensor, joint_rot: torch.Tensor):
        if self._visualize_links:
            body_pos, _ = self._char_model.forward_kinematics(root_pos, root_rot, joint_rot)
            self._update_link_visualizations(body_pos)

        root_pos = root_pos.squeeze(0).cpu().numpy()
        root_rot = root_rot.squeeze(0).roll(1).cpu().numpy()
        dof_pos = self._char_model.rot_to_mjc_dof(joint_rot)
        dof_pos = dof_pos.squeeze(0).cpu().numpy()

        if not self._visualize_climber:
            root_pos += np.array([0, 0, 10000.0])

        self._d.qpos[0:3] = root_pos
        self._d.qpos[3:7] = root_rot
        self._d.qpos[7:] = dof_pos

    def update_grip_states(self, grips: torch.Tensor):
        # grips: right_hand, left_hand, right_foot, left_foot
        grips = grips.squeeze(0).cpu().numpy()
        a = 1.0 if self._visualize_grips else 0.0
        self._m.site("right_hand").rgba = [1, 0, 0, a * grips[0]]
        self._m.site("left_hand").rgba = [1, 0, 0, a * grips[1]]
        self._m.site("right_foot").rgba = [1, 0, 0, a * grips[2]]
        self._m.site("left_foot").rgba = [1, 0, 0, a * grips[3]]

    def render(self):
        mujoco.mj_forward(self._m, self._d)
        self.viewer.sync()
        if self._render_fps:
            time.sleep(1 / self._render_fps)
