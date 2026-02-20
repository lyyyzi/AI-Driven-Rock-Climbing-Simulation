import asyncio
import os
import numpy as np

from isaacsim.core.api.world import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.util.debug_draw import _debug_draw
from pxr import UsdPhysics, Gf
import omni.physics.tensors.impl.api as physx


class MotionViewer():
    def __init__(self) -> None:
        return

    async def init(self):
        if World.instance():
            World.instance().clear_instance()

        world = World()
        await world.initialize_simulation_context_async()

        # Setting up Physics Scene
        stage = world.stage
        gravity = 0.0
        scene = UsdPhysics.Scene.Define(stage, "/World/physics")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(gravity)

        assets_root_path = get_assets_root_path()
        usd_path = assets_root_path + "/Isaac/IsaacLab/Robots/Classic/Humanoid28/humanoid_28.usd"
        prim_path = "/World/Humanoid"
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        humanoid_handle = Articulation(prim_paths_expr="/World/Humanoid", enable_residual_reports=True)
        world.scene.add(humanoid_handle)
        await world.reset_async()

        data = np.load('C:\Matthew\SFU\CMPT 742\cmpt742-project\motion_viewer\data\motion\humanoid_walk.npz')
        data_dof_names = data["dof_names"]
        data_body_names = data["body_names"]

        root_body_name = humanoid_handle.body_names[0]
        root_body_idx = np.where(data["body_names"] == root_body_name)[0]

        humanoid_body_names = np.array(humanoid_handle.body_names)
        remapped_body_indexes = np.where(data_body_names == humanoid_body_names[:, None])[1]

        humanoid_dof_names = np.array(humanoid_handle.dof_names)
        remapped_dof_indexes = np.where(data_dof_names == humanoid_dof_names[:, None])[1]

        self.root_positions = data["body_positions"][:, root_body_idx, :]
        self.root_rotations = data["body_rotations"][:, root_body_idx, :]
        self.body_positions = data["body_positions"][:, remapped_body_indexes, :]
        self.body_rotations = data["body_rotations"][:, remapped_body_indexes, :]
        self.dof_positions = data["dof_positions"][:, remapped_dof_indexes]

        self.motion_data = data
        self.num_frames = self.dof_positions.shape[0]
        self.data_dt = 1 / data["fps"]
        self.curr_time = 0
        self.frame_idx = 0

        self.humanoid_handle = humanoid_handle
        self.physics_view: physx.ArticulationView = humanoid_handle._physics_view

        self.draw = _debug_draw.acquire_debug_draw_interface()

        world.add_physics_callback("update_kinematic_state", callback_fn=self.update_kinematic_state)

    def update_kinematic_state(self, step_size):
        # self.curr_time += step_size
        # if self.curr_time > self.data_dt:
        self.frame_idx = (self.frame_idx + 1) % self.num_frames
        joint_positions = self.dof_positions[self.frame_idx]
        root_position = self.root_positions[self.frame_idx]
        root_rotation = self.root_rotations[self.frame_idx]
        self.set_joint_positions(joint_positions)
        self.set_root_state(root_position, root_rotation)

        if self.humanoid_handle.is_physics_handle_valid():
            self.draw_clear()
            self.draw_body_positions()

    def set_root_state(self, root_position, root_rotation):
        root_position = root_position.reshape(1, -1)
        root_rotation = root_rotation.reshape(1, -1)
        self.humanoid_handle.set_world_poses(root_position, root_rotation)

    def set_joint_positions(self, joint_positions):
        joint_positions = joint_positions.reshape(1, -1)
        self.humanoid_handle.set_joint_positions(joint_positions)

    def get_joint_positions(self):
        joint_positions = self.humanoid_handle.get_joint_positions()
        return joint_positions.squeeze(axis=0)

    def get_body_link_pose(self):
        # self.physics_view.update_articulations_kinematic()
        poses = self.physics_view.get_link_transforms().copy().squeeze(axis=0)
        poses[..., 3:7] = self._convert_quat(poses[..., 3:7], to="wxyz")
        return poses

    def draw_clear(self):
        self.draw.clear_lines()
        self.draw.clear_points()

    def draw_body_positions(self):
        poses = self.get_body_link_pose()
        body_positions = poses[:, :3]
        body_positions[:, 1] += 1.0
        self._draw_points(body_positions, color=[1, 0, 0, 1], size=5)

    def _draw_points(self, points, color=[1, 0, 0, 1], size=1):
        color, size = np.array(color), np.array([size])
        num_pts = points.shape[0]
        colors = np.repeat(color[np.newaxis, :], num_pts, axis=0)
        sizes = np.repeat(size, num_pts, axis=0)
        self.draw.draw_points(points, colors, sizes)

    def _convert_quat(self, quat: np.ndarray, to = "xyzw") -> np.ndarray:
        if to == "xyzw":
            # wxyz -> xyzw
            return np.roll(quat, -1, axis=-1)
        else:
            # xyzw -> wxyz
            return np.roll(quat, 1, axis=-1)


mv = MotionViewer()
asyncio.ensure_future(mv.init())

# mv.draw_clear()

# body_idx = 5
# print(mv.get_body_link_pose()[body_idx])
# print(np.concatenate([mv.body_positions[mv.frame_idx, body_idx], mv.body_rotations[mv.frame_idx, body_idx]], axis=0))