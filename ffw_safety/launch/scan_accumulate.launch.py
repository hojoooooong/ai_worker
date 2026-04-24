#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            'input_topic',
            default_value='/zedm/zed_node/point_cloud/cloud_registered',
            description='Source PointCloud2 topic from the ZED wrapper.'),
        DeclareLaunchArgument(
            'base_frame', default_value='base_link',
            description='Target frame to accumulate points in.'),
        DeclareLaunchArgument('x_min', default_value='0.2'),
        DeclareLaunchArgument('x_max', default_value='1.3'),
        DeclareLaunchArgument('y_min', default_value='-1.0'),
        DeclareLaunchArgument('y_max', default_value='1.0'),
        DeclareLaunchArgument('z_min', default_value='-0.5'),
        DeclareLaunchArgument('z_max', default_value='1.5'),
        DeclareLaunchArgument(
            'voxel_size', default_value='0.01',
            description='Voxel size in meters for downsampling/dedup.'),
        DeclareLaunchArgument(
            'publish_rate', default_value='2.0',
            description='Hz for the live /safety/accumulated_cloud publish.'),
        DeclareLaunchArgument(
            'min_observations', default_value='2',
            description='Consensus filter: voxels seen in fewer than this '
                        'many frames are dropped on finish.'),
    ]

    accumulator = Node(
        package='ffw_safety',
        executable='cloud_accumulator',
        name='cloud_accumulator',
        output='screen',
        parameters=[{
            'input_topic': LaunchConfiguration('input_topic'),
            'base_frame': LaunchConfiguration('base_frame'),
            'x_min': LaunchConfiguration('x_min'),
            'x_max': LaunchConfiguration('x_max'),
            'y_min': LaunchConfiguration('y_min'),
            'y_max': LaunchConfiguration('y_max'),
            'z_min': LaunchConfiguration('z_min'),
            'z_max': LaunchConfiguration('z_max'),
            'voxel_size': LaunchConfiguration('voxel_size'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'min_observations': LaunchConfiguration('min_observations'),
        }],
    )

    return LaunchDescription(declared_args + [accumulator])
