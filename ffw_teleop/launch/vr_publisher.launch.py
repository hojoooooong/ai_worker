#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition, UnlessCondition

from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments
    declared_arguments = []

    # VR Publisher type selection
    declared_arguments.append(
        DeclareLaunchArgument(
            'type',
            default_value='bh5',
            description='VR publisher type: bh5 or sg2 (default: bh5)',
            choices=['bh5', 'sg2']
        )
    )

    # Common parameters
    declared_arguments.append(
        DeclareLaunchArgument(
            'vr_publishing_enabled',
            default_value='false',
            description='Enable VR publishing by default (true/false)'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'fps',
            # default_value='100',
            default_value='30',
            description='FPS for VR hand tracking'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'low_pass_filter_alpha',
            default_value='0.3',
            description='Low-pass filter alpha value for joint smoothing'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'scaling_vr',
            default_value='1.1',
            description='VR data scaling factor'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'pitch_offset',
            default_value='-0.5',
            description='Head pitch offset in radians'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'timer_period',
            # default_value='0.005',
            default_value='0.01',
            description='Timer period for hand trajectory publishing (seconds)'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'sg2_goal_pose_position_scale',
            default_value='1.0',
            description='SG2 goal pose position scale (head-relative arm reach gain)'
        )
    )

    # Get launch configuration
    type = LaunchConfiguration('type')
    vr_publishing_enabled = LaunchConfiguration('vr_publishing_enabled')
    fps = LaunchConfiguration('fps')
    low_pass_filter_alpha = LaunchConfiguration('low_pass_filter_alpha')
    scaling_vr = LaunchConfiguration('scaling_vr')
    pitch_offset = LaunchConfiguration('pitch_offset')
    timer_period = LaunchConfiguration('timer_period')
    sg2_goal_pose_position_scale = LaunchConfiguration('sg2_goal_pose_position_scale')

    # Common parameters for both publishers
    common_parameters = {
        'vr_publishing_enabled': vr_publishing_enabled,
        'fps': fps,
        'low_pass_filter_alpha': low_pass_filter_alpha,
        'scaling_vr': scaling_vr,
        'pitch_offset': pitch_offset,
        'timer_period': timer_period,
    }

    # BH5 VR Trajectory Publisher Node
    vr_trajectory_publisher_bh5_node = Node(
        package='ffw_teleop',
        executable='vr_publisher_bh5',
        name='vr_trajectory_publisher_bh5',
        output='screen',
        parameters=[common_parameters],
        condition=IfCondition(PythonExpression([
            "'", type, "' == 'bh5'"
        ])),
        remappings=[
            # Add any topic remappings if needed
        ],
    )

    # SG2 VR Trajectory Publisher Node (bg2 type kept for compatibility)
    vr_trajectory_publisher_sg2_node = Node(
        package='ffw_teleop',
        executable='vr_publisher_sg2',
        name='vr_trajectory_publisher_sg2',
        output='screen',
        parameters=[{
            'goal_pose_position_scale': sg2_goal_pose_position_scale,
        }],
        condition=IfCondition(PythonExpression([
            "'", type, "' == 'sg2'"
        ])),
        remappings=[
            # Add any topic remappings if needed
        ],
    )

    nodes = [
        vr_trajectory_publisher_bh5_node,
        vr_trajectory_publisher_sg2_node,
    ]

    return LaunchDescription(declared_arguments + nodes)
