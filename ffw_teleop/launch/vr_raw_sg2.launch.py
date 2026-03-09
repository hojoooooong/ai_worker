#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ffw_teleop',
            executable='vr_raw_sg2_consumer',
            name='vr_raw_sg2_consumer',
            output='screen',
        ),
    ])
