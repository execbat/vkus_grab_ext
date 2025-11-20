# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control
* :obj:`FRANKA_ROBOTIQ_GRIPPER_CFG`: Franka robot with Robotiq_2f_85 gripper

Reference: https://github.com/frankaemika/franka_ros
"""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

##
# Configuration
##
import importlib.util
from pathlib import Path
import os

PKG = "vkus_grab_ext"
spec = importlib.util.find_spec(PKG)
pkg_dir = os.path.dirname(spec.origin)
##
# Configuration - Actuators.
##

usd_path_empty = os.path.join(
    pkg_dir,
    "robot",
    "jaka_usd",
    "jaka_zu12.usd",
)

JAKA_ZU12_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path_empty,
        activate_contact_sensors=True,         # activated - that was False
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        
    ),
    init_state=ArticulationCfg.InitialStateCfg( 
        joint_pos={ # RADIANS
            "joint_1": 0.0,
            "joint_2": 1.5708,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "joint_5": -3.14159,
            "joint_6": 0.0,
        },
    ),
    actuators={
        "j1": ImplicitActuatorCfg(
            joint_names_expr=["joint_1"],
            effort_limit_sim=2000.0,            # Force/Torque limit of the joints in the group. 	
            velocity_limit=119.74817,           # Velocity limit of the joints in the group.
            #effort_limit_sim=                  # Velocity limit of the joints in the group.
            #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
            stiffness=21.83886,                 # Stiffness gains (also known as p-gain) of the joints in the group.
            damping=0.00874,                    # Damping gains (also known as d-gain) of the joints in the group.
            armature=0.0,                       # Armature of the joints in the group.
            friction=0.0,                       # The static friction coefficient of the joints in the group.
            dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
            viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
        ),
        "j2": ImplicitActuatorCfg(
            joint_names_expr=["joint_2"],
            effort_limit_sim=8000.0,            # Force/Torque limit of the joints in the group. 	
            velocity_limit=119.74817,           # Velocity limit of the joints in the group.
            #effort_limit_sim=                  # Velocity limit of the joints in the group.
            #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
            stiffness=62.01165,                 # Stiffness gains (also known as p-gain) of the joints in the group.
            damping=0.0248,                     # Damping gains (also known as d-gain) of the joints in the group.
            armature=0.0,                       # Armature of the joints in the group.
            friction=0.0,                       # The static friction coefficient of the joints in the group.
            dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
            viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
        ),        
        "j3": ImplicitActuatorCfg(
            joint_names_expr=["joint_3"],
            effort_limit_sim=8000.0,            # Force/Torque limit of the joints in the group. 	
            velocity_limit=119.74817,           # Velocity limit of the joints in the group.
            #effort_limit_sim=                  # Velocity limit of the joints in the group.
            #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
            stiffness=2534.897715,              # Stiffness gains (also known as p-gain) of the joints in the group.
            damping=1.01396,                    # Damping gains (also known as d-gain) of the joints in the group.
            armature=0.0,                       # Armature of the joints in the group.
            friction=0.0,                       # The static friction coefficient of the joints in the group.
            dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
            viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
        ),          
        "j4": ImplicitActuatorCfg(
            joint_names_expr=["joint_4"],
            effort_limit_sim=2000.0,            # Force/Torque limit of the joints in the group. 	
            velocity_limit=179.90874,           # Velocity limit of the joints in the group.
            #effort_limit_sim=                  # Velocity limit of the joints in the group.
            #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
            stiffness=629.89563,                # Stiffness gains (also known as p-gain) of the joints in the group.
            damping=0.25196,                    # Damping gains (also known as d-gain) of the joints in the group.
            armature=0.0,                       # Armature of the joints in the group.
            friction=0.0,                       # The static friction coefficient of the joints in the group.
            dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
            viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
        ), 
        "j5": ImplicitActuatorCfg(
            joint_names_expr=["joint_5"],
            effort_limit_sim=3000.0,            # Force/Torque limit of the joints in the group. 	
            velocity_limit=179.90874,           # Velocity limit of the joints in the group.
            #effort_limit_sim=                  # Velocity limit of the joints in the group.
            #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
            stiffness=22.04939,                 # Stiffness gains (also known as p-gain) of the joints in the group.
            damping=0.00882,                    # Damping gains (also known as d-gain) of the joints in the group.
            armature=0.0,                       # Armature of the joints in the group.
            friction=0.0,                       # The static friction coefficient of the joints in the group.
            dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
            viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
        ), 
        "j6": ImplicitActuatorCfg(
            joint_names_expr=["joint_6"],
            effort_limit_sim=1000.0,            # Force/Torque limit of the joints in the group. 	
            velocity_limit=179.90874,           # Velocity limit of the joints in the group.
            #effort_limit_sim=                  # Velocity limit of the joints in the group.
            #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
            stiffness=2.25852,                  # Stiffness gains (also known as p-gain) of the joints in the group.
            damping=0.0009,                     # Damping gains (also known as d-gain) of the joints in the group.
            armature=0.0,                       # Armature of the joints in the group.
            friction=0.0,                       # The static friction coefficient of the joints in the group.
            dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
            viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
        ), 
                
    },
    soft_joint_pos_limit_factor=1.0,
)
JAKA_ZU12_CFG.init_state.pos = (0, 0, 1.5)



usd_path_simple_gripper = os.path.join(
    pkg_dir,
    "robot",
    "jaka_usd",
    "jaka_zu12_with_gripper_coll_disabled.usd",
)

JAKA_ZU12_SIMPLE_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path_simple_gripper,
        activate_contact_sensors=True,         # activated - that was False
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        
    ),
)

#JAKA_ZU12_SIMPLE_GRIPPER_CFG.spawn.variants = {"Gripper": "Robotiq_2F_85"}
#JAKA_ZU12_SIMPLE_GRIPPER_CFG.spawn.rigid_props.disable_gravity = True
JAKA_ZU12_SIMPLE_GRIPPER_CFG.spawn.collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
JAKA_ZU12_SIMPLE_GRIPPER_CFG.init_state.joint_pos = {
    "joint_1": 0.0,
    "joint_2": 1.5708,
    "joint_3": 0.0,
    "joint_4": 0.0,
    "joint_5": -3.14159,
    "joint_6": 0.0,
    "finger_joint": 0.0,
    ".*_inner_finger_joint": 0.0,
    ".*_inner_finger_knuckle_joint": 0.0,
    ".*_outer_.*_joint": 0.0,
}
JAKA_ZU12_SIMPLE_GRIPPER_CFG.init_state.pos = (0, 0, 1.5)
JAKA_ZU12_SIMPLE_GRIPPER_CFG.actuators = {
    "j1": ImplicitActuatorCfg(
        joint_names_expr=["joint_1"],
        effort_limit_sim=2000.0,            # Force/Torque limit of the joints in the group. 	
        velocity_limit=119.74817,           # Velocity limit of the joints in the group.
        #effort_limit_sim=                  # Velocity limit of the joints in the group.
        #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
        stiffness=21.83886,                 # Stiffness gains (also known as p-gain) of the joints in the group.
        damping=0.00874,                    # Damping gains (also known as d-gain) of the joints in the group.
        armature=0.0,                       # Armature of the joints in the group.
        friction=0.0,                       # The static friction coefficient of the joints in the group.
        dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
        viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
    ),
    "j2": ImplicitActuatorCfg(
        joint_names_expr=["joint_2"],
        effort_limit_sim=8000.0,            # Force/Torque limit of the joints in the group. 	
        velocity_limit=119.74817,           # Velocity limit of the joints in the group.
        #effort_limit_sim=                  # Velocity limit of the joints in the group.
        #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
        stiffness=62.01165,                 # Stiffness gains (also known as p-gain) of the joints in the group.
        damping=0.0248,                     # Damping gains (also known as d-gain) of the joints in the group.
        armature=0.0,                       # Armature of the joints in the group.
        friction=0.0,                       # The static friction coefficient of the joints in the group.
        dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
        viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
    ),        
    "j3": ImplicitActuatorCfg(
        joint_names_expr=["joint_3"],
        effort_limit_sim=8000.0,            # Force/Torque limit of the joints in the group. 	
        velocity_limit=119.74817,           # Velocity limit of the joints in the group.
        #effort_limit_sim=                  # Velocity limit of the joints in the group.
        #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
        stiffness=2534.897715,              # Stiffness gains (also known as p-gain) of the joints in the group.
        damping=1.01396,                    # Damping gains (also known as d-gain) of the joints in the group.
        armature=0.0,                       # Armature of the joints in the group.
        friction=0.0,                       # The static friction coefficient of the joints in the group.
        dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
        viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
    ),          
    "j4": ImplicitActuatorCfg(
        joint_names_expr=["joint_4"],
        effort_limit_sim=2000.0,            # Force/Torque limit of the joints in the group. 	
        velocity_limit=179.90874,           # Velocity limit of the joints in the group.
        #effort_limit_sim=                  # Velocity limit of the joints in the group.
        #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
        stiffness=629.89563,                # Stiffness gains (also known as p-gain) of the joints in the group.
        damping=0.25196,                    # Damping gains (also known as d-gain) of the joints in the group.
        armature=0.0,                       # Armature of the joints in the group.
        friction=0.0,                       # The static friction coefficient of the joints in the group.
        dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
        viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
    ), 
    "j5": ImplicitActuatorCfg(
        joint_names_expr=["joint_5"],
        effort_limit_sim=3000.0,            # Force/Torque limit of the joints in the group. 	
        velocity_limit=179.90874,           # Velocity limit of the joints in the group.
        #effort_limit_sim=                  # Velocity limit of the joints in the group.
        #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
        stiffness=22.04939,                 # Stiffness gains (also known as p-gain) of the joints in the group.
        damping=0.00882,                    # Damping gains (also known as d-gain) of the joints in the group.
        armature=0.0,                       # Armature of the joints in the group.
        friction=0.0,                       # The static friction coefficient of the joints in the group.
        dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
        viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
    ), 
    "j6": ImplicitActuatorCfg(
        joint_names_expr=["joint_6"],
        effort_limit_sim=1000.0,            # Force/Torque limit of the joints in the group. 	
        velocity_limit=179.90874,           # Velocity limit of the joints in the group.
        #effort_limit_sim=                  # Velocity limit of the joints in the group.
        #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
        stiffness=2.25852,                  # Stiffness gains (also known as p-gain) of the joints in the group.
        damping=0.0009,                     # Damping gains (also known as d-gain) of the joints in the group.
        armature=0.0,                       # Armature of the joints in the group.
        friction=0.0,                       # The static friction coefficient of the joints in the group.
        dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
        viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
    ),
        
    "gripper_drive": ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],  # "right_outer_knuckle_joint" is its mimic joint
        effort_limit_sim=1650,
        velocity_limit_sim=10.0,
        stiffness=17,
        damping=0.02,
    ),
    
    # enable the gripper to grasp in a parallel manner
    "gripper_finger": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_joint"],
        effort_limit_sim=50,
        velocity_limit_sim=10.0,
        stiffness=0.2,
        damping=0.001,
    ),
    # set PD to zero for passive joints in close-loop gripper
    "gripper_passive": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
        effort_limit_sim=1.0,
        velocity_limit_sim=10.0,
        stiffness=0.0,
        damping=0.0,
    ),
    
}




usd_path_gripper_ag_95_160 = os.path.join(
    pkg_dir,
    "robot",
    "jaka_usd",
    "jaka_zu12_experiment.usd",
)

JAKA_ZU12_GRIPPER_AG_95_160_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path_gripper_ag_95_160,
        activate_contact_sensors=True,         # activated - that was False
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=1
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        
    ),
)

#JAKA_ZU12_SIMPLE_GRIPPER_CFG.spawn.variants = {"Gripper": "Robotiq_2F_85"}
#JAKA_ZU12_SIMPLE_GRIPPER_CFG.spawn.rigid_props.disable_gravity = True
JAKA_ZU12_GRIPPER_AG_95_160_CFG.spawn.collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
JAKA_ZU12_GRIPPER_AG_95_160_CFG.init_state.joint_pos = {
    "joint_1": 0.0,
    "joint_2": 1.5708,
    "joint_3": 0.0,
    "joint_4": 0.0,
    "joint_5": -3.14159,
    "joint_6": 0.0,
#    ".*_finger_joint": 0.0,
    "left_outer_knuckle_joint": 0.0,
#    ".*_outer_knuckle_joint": 0.0,
}
JAKA_ZU12_GRIPPER_AG_95_160_CFG.init_state.pos = (0, 0, 1.5)
JAKA_ZU12_GRIPPER_AG_95_160_CFG.actuators = {
    "j1": ImplicitActuatorCfg(
        joint_names_expr=["joint_1"],
        effort_limit_sim=2000.0,            # Force/Torque limit of the joints in the group. 	
        velocity_limit=119.74817,           # Velocity limit of the joints in the group.
        #effort_limit_sim=                  # Velocity limit of the joints in the group.
        #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
        stiffness=21.83886,                 # Stiffness gains (also known as p-gain) of the joints in the group.
        damping=0.00874,                    # Damping gains (also known as d-gain) of the joints in the group.
        armature=0.0,                       # Armature of the joints in the group.
        friction=0.0,                       # The static friction coefficient of the joints in the group.
        dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
        viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
    ),
    "j2": ImplicitActuatorCfg(
        joint_names_expr=["joint_2"],
        effort_limit_sim=8000.0,            # Force/Torque limit of the joints in the group. 	
        velocity_limit=119.74817,           # Velocity limit of the joints in the group.
        #effort_limit_sim=                  # Velocity limit of the joints in the group.
        #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
        stiffness=62.01165,                 # Stiffness gains (also known as p-gain) of the joints in the group.
        damping=0.0248,                     # Damping gains (also known as d-gain) of the joints in the group.
        armature=0.0,                       # Armature of the joints in the group.
        friction=0.0,                       # The static friction coefficient of the joints in the group.
        dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
        viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
    ),        
    "j3": ImplicitActuatorCfg(
        joint_names_expr=["joint_3"],
        effort_limit_sim=8000.0,            # Force/Torque limit of the joints in the group. 	
        velocity_limit=119.74817,           # Velocity limit of the joints in the group.
        #effort_limit_sim=                  # Velocity limit of the joints in the group.
        #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
        stiffness=2534.897715,              # Stiffness gains (also known as p-gain) of the joints in the group.
        damping=1.01396,                    # Damping gains (also known as d-gain) of the joints in the group.
        armature=0.0,                       # Armature of the joints in the group.
        friction=0.0,                       # The static friction coefficient of the joints in the group.
        dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
        viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
    ),          
    "j4": ImplicitActuatorCfg(
        joint_names_expr=["joint_4"],
        effort_limit_sim=2000.0,            # Force/Torque limit of the joints in the group. 	
        velocity_limit=179.90874,           # Velocity limit of the joints in the group.
        #effort_limit_sim=                  # Velocity limit of the joints in the group.
        #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
        stiffness=629.89563,                # Stiffness gains (also known as p-gain) of the joints in the group.
        damping=0.25196,                    # Damping gains (also known as d-gain) of the joints in the group.
        armature=0.0,                       # Armature of the joints in the group.
        friction=0.0,                       # The static friction coefficient of the joints in the group.
        dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
        viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
    ), 
    "j5": ImplicitActuatorCfg(
        joint_names_expr=["joint_5"],
        effort_limit_sim=3000.0,            # Force/Torque limit of the joints in the group. 	
        velocity_limit=179.90874,           # Velocity limit of the joints in the group.
        #effort_limit_sim=                  # Velocity limit of the joints in the group.
        #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
        stiffness=22.04939,                 # Stiffness gains (also known as p-gain) of the joints in the group.
        damping=0.00882,                    # Damping gains (also known as d-gain) of the joints in the group.
        armature=0.0,                       # Armature of the joints in the group.
        friction=0.0,                       # The static friction coefficient of the joints in the group.
        dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
        viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
    ), 
    "j6": ImplicitActuatorCfg(
        joint_names_expr=["joint_6"],
        effort_limit_sim=1000.0,            # Force/Torque limit of the joints in the group. 	
        velocity_limit=179.90874,           # Velocity limit of the joints in the group.
        #effort_limit_sim=                  # Velocity limit of the joints in the group.
        #velocity_limit_sim=                # Velocity limit of the joints in the group applied to the simulation physics solver.
        stiffness=2.25852,                  # Stiffness gains (also known as p-gain) of the joints in the group.
        damping=0.0009,                     # Damping gains (also known as d-gain) of the joints in the group.
        armature=0.0,                       # Armature of the joints in the group.
        friction=0.0,                       # The static friction coefficient of the joints in the group.
        dynamic_friction=0.0,               # The dynamic friction coefficient of the joints in the group.
        viscous_friction=0.0,               # The viscous friction coefficient of the joints in the group.
    ),
        
    "gripper_drive": ImplicitActuatorCfg(
        joint_names_expr=["left_outer_knuckle_joint"],  # "right_outer_knuckle_joint" is its mimic joint
        effort_limit_sim=1650,
        velocity_limit_sim=10.0,
        stiffness=17,
        damping=0.02,
    ),
    
    # enable the gripper to grasp in a parallel manner
#    "gripper_finger": ImplicitActuatorCfg(
#        joint_names_expr=[".*_finger_joint"],
#        effort_limit_sim=50,
#        velocity_limit_sim=10.0,
#        stiffness=0.2,
#        damping=0.001,
#    ),
#    # set PD to zero for passive joints in close-loop gripper
#    "gripper_passive": ImplicitActuatorCfg(
#        joint_names_expr=["left_inner_knuckle_joint", "right_inner_knuckle_joint", "right_outer_knuckle_joint"],
#        effort_limit_sim=1.0,
#        velocity_limit_sim=10.0,
#        stiffness=0.0,
#        damping=0.0,
#    ),
    
}

JAKA_ZU12_GRIPPER_AG_95_160_CFG.spawn.activate_contact_sensors=True

"""Configuration of Franka Emika Panda robot with Robotiq_2f_85 gripper."""
