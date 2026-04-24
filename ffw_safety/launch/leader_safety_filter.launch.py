#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('ffw_safety')
    default_capsule_config = PathJoinSubstitution(
        [pkg_share, 'config', 'capsules.yaml'])

    declared_args = [
        DeclareLaunchArgument(
            'safety_margin', default_value='0.015',
            description='Extra clearance in meters above ceiling.'),
        DeclareLaunchArgument(
            'voxel_size', default_value='0.01',
            description='xy cell size for the ceiling height map; '
                        'should match cloud_accumulator voxel_size.'),
        DeclareLaunchArgument(
            'capsule_config_file',
            default_value=default_capsule_config,
            description='YAML file describing capsules per side '
                        '(frame, p1, p2, radius).'),
        DeclareLaunchArgument(
            'switch_controller_service',
            default_value='/controller_manager/switch_controller'),
        DeclareLaunchArgument(
            'left_controller', default_value='arm_l_controller'),
        DeclareLaunchArgument(
            'right_controller', default_value='arm_r_controller'),
        DeclareLaunchArgument(
            'cloud_topic', default_value='/safety/accumulated_cloud'),
        DeclareLaunchArgument(
            'robot_description_topic', default_value='/robot_description'),
        DeclareLaunchArgument(
            'joint_states_topic', default_value='/joint_states'),
        DeclareLaunchArgument(
            'left_input_topic',
            default_value='/leader/joint_trajectory_command_broadcaster_left'
                          '/joint_trajectory'),
        DeclareLaunchArgument(
            'right_input_topic',
            default_value='/leader/joint_trajectory_command_broadcaster_right'
                          '/joint_trajectory'),
        DeclareLaunchArgument(
            'marker_topic', default_value='/safety/capsule_markers'),
        DeclareLaunchArgument(
            'marker_publish_rate', default_value='10.0',
            description='Hz for capsule MarkerArray updates in RViz. '
                        '0 to disable.'),
    ]

    filter_node = Node(
        package='ffw_safety',
        executable='leader_safety_filter',
        name='leader_safety_filter',
        output='screen',
        parameters=[{
            'safety_margin': LaunchConfiguration('safety_margin'),
            'voxel_size': LaunchConfiguration('voxel_size'),
            'capsule_config_file': LaunchConfiguration(
                'capsule_config_file'),
            'switch_controller_service': LaunchConfiguration(
                'switch_controller_service'),
            'left_controller': LaunchConfiguration('left_controller'),
            'right_controller': LaunchConfiguration('right_controller'),
            'cloud_topic': LaunchConfiguration('cloud_topic'),
            'robot_description_topic': LaunchConfiguration(
                'robot_description_topic'),
            'joint_states_topic': LaunchConfiguration('joint_states_topic'),
            'left_input_topic': LaunchConfiguration('left_input_topic'),
            'right_input_topic': LaunchConfiguration('right_input_topic'),
            'marker_topic': LaunchConfiguration('marker_topic'),
            'marker_publish_rate': LaunchConfiguration(
                'marker_publish_rate'),
        }],
    )

    return LaunchDescription(declared_args + [filter_node])
