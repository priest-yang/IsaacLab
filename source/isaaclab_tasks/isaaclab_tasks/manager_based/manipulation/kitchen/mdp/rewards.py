# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_is_lifted_and_in_gripper(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object"), ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # get the object position
    object_pos = object.data.root_pos_w[:, :3]
    # get the ee position
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    # compute the distance
    distance = torch.norm(object_pos - ee_pos, dim=1)
    # check if the object is lifted above the minimal height
    is_lifted = object.data.root_pos_w[:, 2] > minimal_height
    # check if the object is in the gripper
    is_in_gripper = distance < 0.1
    # return the reward
    return is_lifted * is_in_gripper


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))



def floating_base_close_to_object(
    env: ManagerBasedRLEnv,
    ref_distance: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    key: str | None = None,
) -> torch.Tensor:
    """
    Reward the agent for the floating base being close to the object.
    Only check x,y distance.
    if dis < ref_distance, reward = 0
    # smooth penalty
    else penalty = log(dis+1) / log(ref_distance+1)
    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    # get the floating base position
    if key is None:
        floating_base_pos = robot.data.root_pos_w[:, :2]
    else:
        key_index = robot.data.body_names.index(key)
        floating_base_pos = robot.data.body_com_pos_w[:, key_index, :2]
    # get the object position
    object_pos = object.data.root_pos_w[:, :2]
    # compute the distance
    distance = torch.norm(floating_base_pos - object_pos, dim=1)
    # compute the reward
    if isinstance(ref_distance, torch.Tensor):
        ref_distance = ref_distance.to(distance.device)
    else:
        ref_distance = torch.tensor(ref_distance, device=distance.device)

    reward = torch.where(distance < ref_distance, 0.0, -torch.log(distance + 1.0) / torch.log(ref_distance + 1.0))
    return reward
