# Copyright 2023 ROBOTIS CO., LTD.
# Authors: Sungho Woo

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    param_dir = LaunchConfiguration(
      'param_dir',
      default=os.path.join(
        get_package_share_directory('v_marker_estimation'),
        'param',
        'v_marker_estimation_config.yaml'
      )
    )

    return LaunchDescription([
      DeclareLaunchArgument(
        'param_dir',
        default_value=param_dir,
        description='Full path of parameter file'
      ),

      Node(
        package='v_marker_estimation',
        executable='lidar_cluster',
        name='lidar_cluster',
        parameters=[param_dir],
        output='screen'
      ),

      Node(
        package='v_marker_estimation',
        executable='v_marker_localization',
        name='v_marker_localization',
        parameters=[param_dir],
        output='screen'
      ),

      Node(
        package='v_marker_estimation',
        executable='holonomic_drive_controller',
        name='holonomic_drive_controller',
        parameters=[param_dir],
        output='screen'
      ),

      Node(
        package='v_marker_estimation',
        executable='marker_ab_goal_router',
        name='marker_ab_goal_router',
        parameters=[param_dir],
        output='screen'
      )

    ])
