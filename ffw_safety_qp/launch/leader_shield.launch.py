# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    qp_share = FindPackageShare('ffw_safety_qp')
    cyclo_models = FindPackageShare('cyclo_motion_controller_models')

    args = [
        DeclareLaunchArgument(
            'robot_model',
            default_value='sg2',
            description='FFW robot variant: sg2 | bg2.',
        ),
        DeclareLaunchArgument('control_dt', default_value='0.02'),
        DeclareLaunchArgument('alpha_obs', default_value='5.0'),
        DeclareLaunchArgument('alpha_jl', default_value='10.0'),
        DeclareLaunchArgument('damping', default_value='0.01'),
        DeclareLaunchArgument('slack_penalty', default_value='1.0e4'),
        DeclareLaunchArgument('buffer', default_value='0.10'),
        DeclareLaunchArgument('safe_distance', default_value='0.02'),
    ]

    urdf_path = PathJoinSubstitution([
        cyclo_models, 'models', 'ai_worker',
        PythonExpression(['"ffw_" + "',
                          LaunchConfiguration('robot_model'),
                          '" + "_follower.urdf"']),
    ])
    config_path = PathJoinSubstitution([
        qp_share, 'config',
        PythonExpression(['"shield_" + "',
                          LaunchConfiguration('robot_model'),
                          '" + ".yaml"']),
    ])

    shield_node = Node(
        package='ffw_safety_qp',
        executable='leader_shield_node',
        name='leader_shield',
        output='screen',
        parameters=[
            config_path,
            {
                'urdf_path': urdf_path,
                'control_dt': LaunchConfiguration('control_dt'),
                'alpha_obs': LaunchConfiguration('alpha_obs'),
                'alpha_jl': LaunchConfiguration('alpha_jl'),
                'damping': LaunchConfiguration('damping'),
                'slack_penalty': LaunchConfiguration('slack_penalty'),
                'buffer': LaunchConfiguration('buffer'),
                'safe_distance': LaunchConfiguration('safe_distance'),
            },
        ],
    )

    return LaunchDescription(args + [shield_node])
