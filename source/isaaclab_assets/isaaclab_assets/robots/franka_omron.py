# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_OMRON_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_OMRON_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

FRANKA_OMRON_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path= '/home/johndoe/Documents/IsaacLab/source/isaaclab_assets/data/omron_franka_final.usd',
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos = (0, 0, 0),
        rot = (1, 0, 0, 0),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,

            # base
            "mobilebase_forward": 0.0,
            "mobilebase_side": 0.0,
            "mobilebase_yaw": 0.0,
            "mobilebase_torso_height": 0.0,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        "mobile_base_movement": ImplicitActuatorCfg(
            joint_names_expr=["mobilebase_forward", "mobilebase_side", ],
            effort_limit=100000,
            velocity_limit=0.1,
            stiffness=1e6,
            damping=1e1,
        ),
        "mobile_base_rotate": ImplicitActuatorCfg(
            joint_names_expr=["mobilebase_yaw"],
            effort_limit=100000,
            velocity_limit=0.5,
            stiffness=1e6,
            damping=1e1,
        ),
        "mobile_base_torso": ImplicitActuatorCfg(
            joint_names_expr=["mobilebase_torso_height"],
            effort_limit=100000000,
            velocity_limit=0.1,
            stiffness=1e6,
            damping=1e1,
        ),

    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


FRANKA_OMRON_HIGH_PD_CFG = FRANKA_OMRON_CFG.copy()
FRANKA_OMRON_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_OMRON_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_OMRON_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_OMRON_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_OMRON_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0

FRANKA_OMRON_HIGH_PD_CFG.actuators["mobile_base_movement"].stiffness = 1e6
FRANKA_OMRON_HIGH_PD_CFG.actuators["mobile_base_movement"].damping = 1e4
FRANKA_OMRON_HIGH_PD_CFG.actuators["mobile_base_torso"].stiffness = 1e6
FRANKA_OMRON_HIGH_PD_CFG.actuators["mobile_base_torso"].damping = 1e4
FRANKA_OMRON_HIGH_PD_CFG.actuators["mobile_base_rotate"].stiffness = 1e6
FRANKA_OMRON_HIGH_PD_CFG.actuators["mobile_base_rotate"].damping = 1e4

"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
