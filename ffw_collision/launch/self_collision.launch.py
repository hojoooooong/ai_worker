#!/usr/bin/env python3
#
# Copyright 2024 ROBOTIS CO., LTD.
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
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Sungho Woo

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    urdf_path = PathJoinSubstitution([
        FindPackageShare('ffw_description'),
        'urdf',
        'ffw_sg2_rev1_follower',
        'ffw_sg2_follower.urdf.xacro'
    ])

    # Declare launch argument for log level
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level: debug, info, warn, error'
    )

    return LaunchDescription([
        log_level_arg,
        Node(
            package='ffw_collision',
            executable='self_collision_node',
            name='self_collision_node',
            parameters=[{
                'robot_description': Command(['xacro ', urdf_path]),
                'base_link': 'arm_base_link',
                'enable_marker': True,
                'joint_states_topic': '/joint_states'
            }],
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
            output='screen'
        )
    ])

