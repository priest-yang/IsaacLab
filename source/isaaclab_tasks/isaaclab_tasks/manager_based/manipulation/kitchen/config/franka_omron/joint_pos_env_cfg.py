# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.mytask import mdp
from isaaclab_tasks.manager_based.manipulation.mytask.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka_omron import FRANKA_OMRON_CFG  # isort: skip


@configclass
class FrankaOmronCubeLiftEnvCfg(LiftEnvCfg):

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_OMRON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        print(f"check")

        # Set actions for the specific robot type (franka)

        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint[1-7]"], scale=0.5, use_zero_offset=False
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        self.actions.base_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["mobilebase_.*"],
            scale=0.01, #01,
            use_zero_offset=True, # use default offset is not working for base action
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # # Set Cube as object
        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[1, 0, 0.855], rot=[1, 0, 0, 0]),
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        #         scale=(0.8, 0.8, 0.8),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )
        # bowl
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.75, 0, 0.9], rot=[-0.70710678, 0.70710678, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=r'C:\Academic\Umich\ROB599DLRP\Project\final\ws\IsaacLab\source\isaaclab_assets\data\bowl.usd',
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/omron_v2/world",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/omron_v2/Franka/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


        self.floating_base_key = "mobilebase0_wheeled_base"

        from isaaclab.managers import SceneEntityCfg
        
        # Obs
        
        # # update termination config use "mobilebase0_wheeled_base" pos as robot pos
        # from isaaclab.managers import TerminationTermCfg as DoneTerm
        # # (change to "mobilebase0_wheeled_base" for floating based termination)
        # self.terminations.far_from_object = DoneTerm(
        # func=mdp.root_far_from_object,
        # params={"distance": 2, 
        #         "asset_cfg": SceneEntityCfg("robot"), 
        #         "object_cfg": SceneEntityCfg("object"), 
        #         "key": self.floating_base_key
        #         },
        # )

        # reward: floating base close to object
        from isaaclab.managers import RewardTermCfg as RewTerm
        self.rewards.floating_base_close_to_object = RewTerm(
            func=mdp.floating_base_close_to_object,
            params={"ref_distance": 1.0, 
                    "asset_cfg": SceneEntityCfg("robot"), 
                    "object_cfg": SceneEntityCfg("object"),
                    "key": self.floating_base_key
                    },
            weight=1e-1,
        )







@configclass
class FrankaOmronCubeLiftEnvCfg_PLAY(FrankaOmronCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
