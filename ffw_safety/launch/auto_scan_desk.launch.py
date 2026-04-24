#!/usr/bin/env python3

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
    pkg_share = FindPackageShare('ffw_safety')

    declared_args = [
        DeclareLaunchArgument('head_pitch', default_value='0.71'),
        DeclareLaunchArgument('yaw_amplitude', default_value='0.8'),
        DeclareLaunchArgument('leg_duration', default_value='2.0'),
        DeclareLaunchArgument('num_sweeps', default_value='1'),
        DeclareLaunchArgument('pre_wait', default_value='3.0'),
        DeclareLaunchArgument('post_wait', default_value='1.0'),
        DeclareLaunchArgument('min_observations', default_value='2'),
        DeclareLaunchArgument('safety_margin', default_value='0.015'),
    ]

    accumulator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [pkg_share, 'launch', 'scan_accumulate.launch.py'])
        ),
        launch_arguments={
            'min_observations': LaunchConfiguration('min_observations'),
        }.items(),
    )

    sweeper = Node(
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

    sweeper_delayed = TimerAction(period=1.0, actions=[sweeper])

    safety_filter_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [pkg_share, 'launch', 'leader_safety_filter.launch.py'])
        ),
        launch_arguments={
            'safety_margin': LaunchConfiguration('safety_margin'),
        }.items(),
    )

    return LaunchDescription(
        declared_args + [
            accumulator_launch, sweeper_delayed, safety_filter_launch])
