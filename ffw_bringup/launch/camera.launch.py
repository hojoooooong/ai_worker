#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
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
# Authors: Sungho Woo, Woojin Wie, Wonho Yun

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument('launch_wrist_cameras', default_value='true',
                              description='Whether to launch wrist (RealSense) cameras.'),
    ]

    launch_wrist_cameras = LaunchConfiguration('launch_wrist_cameras')

    bringup_launch_dir = os.path.join(get_package_share_directory('ffw_bringup'), 'launch')

    # ZED camera launch
    camera_zed = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_launch_dir, 'camera_zed.launch.py')),
        launch_arguments={'camera_model': 'zedm'}.items()
    )

    # ZED topic relay nodes
    relay_left_head = Node(
        package='topic_tools',
        executable='relay',
        name='relay_cam_left_head',
        arguments=[
            '/zed/zed_node/left/image_rect_color/compressed',
            '/robot/camera/cam_left_head/image_raw/compressed'
        ],
        output='screen'
    )

    relay_right_head = Node(
        package='topic_tools',
        executable='relay',
        name='relay_cam_right_head',
        arguments=[
            '/zed/zed_node/right/image_rect_color/compressed',
            '/robot/camera/cam_right_head/image_raw/compressed'
        ],
        output='screen'
    )

    relay_left_head_info = Node(
        package='topic_tools',
        executable='relay',
        name='relay_cam_left_head_info',
        arguments=[
            '/zed/zed_node/left/camera_info',
            '/robot/camera/cam_left_head/image_raw/compressed/camera_info'
        ],
        output='screen'
    )

    relay_right_head_info = Node(
        package='topic_tools',
        executable='relay',
        name='relay_cam_right_head_info',
        arguments=[
            '/zed/zed_node/right/camera_info',
            '/robot/camera/cam_right_head/image_raw/compressed/camera_info'
        ],
        output='screen'
    )

    # RealSense cameras launch
    camera_realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_launch_dir, 'camera_realsense.launch.py')),
        condition=IfCondition(launch_wrist_cameras),
    )

    # RealSense topic relay nodes
    # Relay from /camera_left/camera_left/color/image_rect_raw/compressed
    # to /robot/camera/cam_left_wrist/image_raw/compressed
    relay_left_wrist = Node(
        package='topic_tools',
        executable='relay',
        name='relay_cam_left_wrist',
        arguments=[
            '/camera_left/camera_left/color/image_rect_raw/compressed',
            '/robot/camera/cam_left_wrist/image_raw/compressed'
        ],
        output='screen'
    )

    relay_right_wrist = Node(
        package='topic_tools',
        executable='relay',
        name='relay_cam_right_wrist',
        arguments=[
            '/camera_right/camera_right/color/image_rect_raw/compressed',
            '/robot/camera/cam_right_wrist/image_raw/compressed'
        ],
        output='screen'
    )

    relay_left_wrist_info = Node(
        package='topic_tools',
        executable='relay',
        name='relay_cam_left_wrist_info',
        arguments=[
            '/camera_left/camera_left/color/camera_info',
            '/robot/camera/cam_left_wrist/image_raw/compressed/camera_info'
        ],
        output='screen'
    )

    relay_right_wrist_info = Node(
        package='topic_tools',
        executable='relay',
        name='relay_cam_right_wrist_info',
        arguments=[
            '/camera_right/camera_right/color/camera_info',
            '/robot/camera/cam_right_wrist/image_raw/compressed/camera_info'
        ],
        output='screen'
    )

    # Delay ZED relay nodes to start after ZED camera is ready
    zed_relay_nodes = TimerAction(
        period=5.0,
        actions=[relay_left_head, relay_right_head,
                 relay_left_head_info, relay_right_head_info]
    )

    # Delay RealSense relay nodes to start after RealSense cameras are ready
    realsense_relay_nodes = TimerAction(
        period=15.0,
        actions=[relay_left_wrist, relay_right_wrist,
                 relay_left_wrist_info, relay_right_wrist_info],
        condition=IfCondition(launch_wrist_cameras),
    )

    return LaunchDescription(
        declared_arguments + [
            camera_zed,
            zed_relay_nodes,
            TimerAction(period=10.0, actions=[camera_realsense],
                        condition=IfCondition(launch_wrist_cameras)),
            realsense_relay_nodes,
        ]
    )
