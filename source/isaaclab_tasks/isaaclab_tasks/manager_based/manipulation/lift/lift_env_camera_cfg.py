# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# add camera
from isaaclab.sensors import TiledCameraCfg, CameraCfg
from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[1, 0, 0.8], rot=[0.707, 0, 0, 0.707]),
        # init_state=AssetBaseCfg.InitialStateCfg(pos=[1, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]), #-1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


    # camera

    # add camera
    # <camera name="robot0_agentview_center" pos="-0.6 0 1.15" quat="0.636946 0.332519 -0.319924 -0.61756"/>
    # <camera name="robot0_agentview_left" pos="-0.5 0.35 1.05" quat="0.556238 0.299353 -0.376787 -0.677509" fovy="60"/>
    # <camera name="robot0_agentview_right" pos="-0.5 -0.35 1.05" quat="0.677509 0.376787 -0.299353 -0.556239" fovy="60"/>
    # <camera name="robot0_eye_in_hand" pos="0.05 0 0" quat="0 0.707107 0.707107 0" fovy="75"/>


    agentview_left_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/omron_v2/mobilebase0_support/agentview_left",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.5, 0.35, 1.05), 
                                        rot=(0.556238, 0.299353, -0.376787, -0.677509), 
                                        convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=27.7,  # Adjusted for 60° FOV
            clipping_range=(0.1, 1.0e5), 
            lock_camera=True
        ),
        width=224,
        height=224,
        update_period=0.05,
    )

    agentview_right_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/omron_v2/mobilebase0_support/agentview_right",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.5, -0.35, 1.05), rot=(0.677509, 0.376787, -0.299353, -0.556239), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=27.7,  # Adjusted for 60° FOV
            clipping_range=(0.1, 1.0e5), 
            lock_camera=True
        ),
        width=224,
        height=224,
        update_period=0.05,
    )

    eye_in_hand_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/omron_v2/Franka/panda_hand/eye_in_hand",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.05, 0, 0), rot=(0, 0.707107, 0.707107, 0), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),  # Closer clipping for hand camera
            lock_camera=True
        ),
        width=224,
        height=224,
        update_period=0.05,
    )

    # # camera viz
    # agentview_left_camera_viz: FrameTransformerCfg = MISSING
    # agentview_right_camera_viz: FrameTransformerCfg = MISSING
    # eye_in_hand_camera_viz: FrameTransformerCfg = MISSING




##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        # debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(1-0.8, 1-0.6), pos_y=(-0.25, 0.25), pos_z=(0.65+0.3, 0.85+0.3), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    base_action: mdp.RelativeJointPositionActionCfg | mdp.JointPositionActionCfg= MISSING
    arm_action: mdp.RelativeJointPositionActionCfg | mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.RelativeJointPositionActionCfg | mdp.JointPositionActionCfg= MISSING #mdp.BinaryJointPositionActionCfg = MISSING

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame) 
         
        # modified to floating basis
        object_position_in_robot_frame = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={
                    "key": "mobilebase0_wheeled_base"
                    },
        )
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        
        # modifyd to floating basis
        target_object_position_in_robot_frame = ObsTerm(
            func=mdp.target_object_position_in_robot_root_frame, params={
                "robot_cfg": SceneEntityCfg("robot"),
                "command_name": "object_pose",
                "key": "mobilebase0_wheeled_base"   
            }
        )
        actions = ObsTerm(func=mdp.last_action)


        # add camera observations
        agentview_left_rgb = ObsTerm(func=mdp.raw_image, params={"sensor_cfg": SceneEntityCfg("agentview_left_camera"), "data_type": "rgb"})
        agentview_right_rgb = ObsTerm(func=mdp.raw_image, params={"sensor_cfg": SceneEntityCfg("agentview_right_camera"), "data_type": "rgb"})
        eye_in_hand_rgb = ObsTerm(func=mdp.raw_image, params={"sensor_cfg": SceneEntityCfg("eye_in_hand_camera"), "data_type": "rgb"})


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False # return dict of observations

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.8, -0.5), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=2.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04 + 0.8210}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04 + 0.8210, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04 + 0.8210, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    # floating_base_close_to_object = RewTerm(
    #     func=mdp.floating_base_close_to_object,
    #     params={"ref_distance": 1.0, "asset_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object")},
    #     weight=1e-1,
    # )




@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.8210-0.1, "asset_cfg": SceneEntityCfg("object")}
    )

    # far_from_object = DoneTerm(
    #     func=mdp.root_far_from_object,
    #     params={"distance": 5, "asset_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object")},
    # )






@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class LiftEnvCameraCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=3)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10 # control at 10Hz
        self.episode_length_s = 35.0
        # simulation settings
        self.sim.dt = 0.01 # 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
