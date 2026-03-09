#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ffw_teleop',
            executable='vr_raw_sh5_consumer',
            name='vr_raw_sh5_consumer',
            output='screen',
            parameters=[{
                'enable_lift_publishing': True,
                'enable_head_publishing': False,
                'enable_base_publishing': False,
                'hand_pose_is_head_relative': True,
            }],
        ),
    ])
