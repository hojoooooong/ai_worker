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
#
# Launch file to test arm collision by publishing joint trajectory commands

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, PythonExpression


def generate_launch_description():
    # Get the path to the script
    script_path = PathJoinSubstitution([
        FindPackageShare('ffw_collision'),
        'scripts',
        'test_arm_collision.py'
    ])

    return LaunchDescription([
        ExecuteProcess(
            cmd=['python3', script_path],
            output='screen',
            name='arm_collision_test'
        )
    ])

