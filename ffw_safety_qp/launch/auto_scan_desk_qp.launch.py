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

# auto_scan_desk_qp:
#   1. ffw_safety/scan_accumulate.launch.py  (cloud_accumulator)
#   2. ffw_safety/head_scan_sweeper           (sweeps + finishes accumulation)
#   3. ffw_safety_qp/leader_shield.launch.py  (QP shield instead of FK gate)

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ffw_safety_share = FindPackageShare('ffw_safety')
    qp_share = FindPackageShare('ffw_safety_qp')

    args = [
        DeclareLaunchArgument(
            'robot_model',
            default_value='sg2',
            description='FFW robot variant: sg2 | bg2.',
        ),
        DeclareLaunchArgument('head_pitch', default_value='0.71'),
        DeclareLaunchArgument('yaw_amplitude', default_value='0.8'),
        DeclareLaunchArgument('leg_duration', default_value='2.0'),
        DeclareLaunchArgument('num_sweeps', default_value='1'),
        DeclareLaunchArgument('pre_wait', default_value='3.0'),
        DeclareLaunchArgument('post_wait', default_value='1.0'),
        DeclareLaunchArgument('min_observations', default_value='2'),
        DeclareLaunchArgument('control_dt', default_value='0.02'),
        DeclareLaunchArgument('alpha_obs', default_value='5.0'),
        DeclareLaunchArgument('buffer', default_value='0.10'),
        DeclareLaunchArgument('safe_distance', default_value='0.02'),
    ]

    accumulator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                ffw_safety_share, 'launch', 'scan_accumulate.launch.py',
            ])
        ),
        launch_arguments={
            'min_observations': LaunchConfiguration('min_observations'),
        }.items(),
    )

    sweeper_node = Node(
        package='ffw_safety',
        executable='head_scan_sweeper',
        name='head_scan_sweeper',
        output='screen',
        parameters=[{
            'head_pitch': LaunchConfiguration('head_pitch'),
            'yaw_amplitude': LaunchConfiguration('yaw_amplitude'),
            'leg_duration': LaunchConfiguration('leg_duration'),
            'num_sweeps': LaunchConfiguration('num_sweeps'),
            'pre_wait': LaunchConfiguration('pre_wait'),
            'post_wait': LaunchConfiguration('post_wait'),
        }],
    )
    sweeper_delayed = TimerAction(period=1.0, actions=[sweeper_node])

    shield_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                qp_share, 'launch', 'leader_shield.launch.py',
            ])
        ),
        launch_arguments={
            'robot_model': LaunchConfiguration('robot_model'),
            'control_dt': LaunchConfiguration('control_dt'),
            'alpha_obs': LaunchConfiguration('alpha_obs'),
            'buffer': LaunchConfiguration('buffer'),
            'safe_distance': LaunchConfiguration('safe_distance'),
        }.items(),
    )

    return LaunchDescription(
        args + [accumulator_launch, sweeper_delayed, shield_launch])
