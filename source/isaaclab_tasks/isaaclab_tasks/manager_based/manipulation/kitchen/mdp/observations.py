# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# def object_position_in_robot_root_frame(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """The position of the object in the robot's root frame."""
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     object_pos_w = object.data.root_pos_w[:, :3]
#     object_pos_b, _ = subtract_frame_transforms(
#         robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
#     )
#     return object_pos_b


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    key: str | None = None,
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    if key is None:
        robot_root_pos_w = robot.data.root_state_w[:, :3]
        robot_root_rot_w = robot.data.root_state_w[:, 3:7]
    else: # floating base robot with base frame not aligned with world frame
        key_index = robot.data.body_names.index(key)
        robot_root_pos_w = robot.data.body_com_pos_w[:, key_index, :]
        robot_root_rot_w = robot.data.body_com_quat_w[:, key_index, :]

    object_pos_b, _ = subtract_frame_transforms(
        robot_root_pos_w, robot_root_rot_w, object_pos_w
    )
    return object_pos_b

def target_object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    command_name: str,
    key: str | None = None,
) -> torch.Tensor:
    """
    The position of the target object in the robot's root frame.
    key: the key of the floating base robot
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    if key is None:
        return command
    else: # floating base robot with base frame not aligned with world frame
        key_index = robot.data.body_names.index(key)
        robot_root_pos_w = robot.data.body_com_pos_w[:, key_index, :]
        robot_root_rot_w = robot.data.body_com_quat_w[:, key_index, :]
        fixed_root_pos_w, fixed_root_rot_w = robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7]

        cmd_2_fixedbase_pos, cmd_2_fixedbase_rot = command[:, :3], command[:, 3:7]

        floatbase_to_fixedbase_pos, floatbase_to_fixedbase_rot = subtract_frame_transforms(
            robot_root_pos_w, robot_root_rot_w, fixed_root_pos_w, fixed_root_rot_w
        )

        cmd_2_floatbase_pos, cmd_2_floatbase_rot = combine_frame_transforms(
            cmd_2_fixedbase_pos, cmd_2_fixedbase_rot, floatbase_to_fixedbase_pos, floatbase_to_fixedbase_rot
        )
        cmd_updated = torch.cat((cmd_2_floatbase_pos, cmd_2_floatbase_rot), dim=-1)

        return cmd_updated
        # convert the command to the robot's root frame

    