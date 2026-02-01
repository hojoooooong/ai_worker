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
# This script publishes joint trajectory commands to make both arms collide
# for testing the self-collision detection system.

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time


class ArmCollisionTest(Node):
    """Publishes joint trajectory commands to make both arms collide."""

    def __init__(self):
        super().__init__('arm_collision_test')

        # Publishers for left and right arm
        self.left_arm_pub = self.create_publisher(
            JointTrajectory,
            '/leader/joint_trajectory_command_broadcaster_left/joint_trajectory',
            10
        )

        self.right_arm_pub = self.create_publisher(
            JointTrajectory,
            '/leader/joint_trajectory_command_broadcaster_right/joint_trajectory',
            10
        )

        # Joint names for both arms (8 joints each, including gripper)
        # Following ffw_teleop keyboard_control.py structure
        self.left_joints = [
            'arm_l_joint1', 'arm_l_joint2', 'arm_l_joint3',
            'arm_l_joint4', 'arm_l_joint5', 'arm_l_joint6', 'arm_l_joint7',
            'gripper_l_joint1'
        ]

        self.right_joints = [
            'arm_r_joint1', 'arm_r_joint2', 'arm_r_joint3',
            'arm_r_joint4', 'arm_r_joint5', 'arm_r_joint6', 'arm_r_joint7',
            'gripper_r_joint1'
        ]

        # Current joint positions (will be updated from joint_states)
        self.left_current_positions = [0.0] * 8
        self.right_current_positions = [0.0] * 8
        self.joint_states_received = False

        # Subscribe to joint states to get current positions
        self.joint_states_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Collision positions: arms moving toward each other
        # Left arm: rotate right (positive joint1), extend forward
        # Right arm: rotate left (negative joint1), extend forward
        self.left_collision_positions = [
            1.0,   # joint1: rotate right
            0.5,   # joint2: lift up
            -0.5,  # joint3: extend forward
            1.0,   # joint4
            0.0,   # joint5
            0.0,   # joint6
            0.0,   # joint7
            0.0    # gripper
        ]

        self.right_collision_positions = [
            -1.0,  # joint1: rotate left
            0.5,   # joint2: lift up
            1.0,  # joint3: extend forward
            1.0,   # joint4
            0.0,   # joint5
            0.0,   # joint6
            0.0,   # joint7
            0.0    # gripper
        ]

        # Declare parameters for customizable positions
        self.declare_parameter('left_joint1', -1.0)
        self.declare_parameter('left_joint2', -1.0)
        self.declare_parameter('left_joint3', 1.3)
        self.declare_parameter('right_joint1', -1.0)
        self.declare_parameter('right_joint2', 1.0)
        self.declare_parameter('right_joint3', -1.3)
        self.declare_parameter('duration', 3.0)
        self.declare_parameter('repeat', False)
        self.declare_parameter('repeat_interval', 5.0)

        # Get parameters
        self.left_collision_positions[0] = self.get_parameter('left_joint1').value
        self.left_collision_positions[1] = self.get_parameter('left_joint2').value
        self.left_collision_positions[2] = self.get_parameter('left_joint3').value
        self.right_collision_positions[0] = self.get_parameter('right_joint1').value
        self.right_collision_positions[1] = self.get_parameter('right_joint2').value
        self.right_collision_positions[2] = self.get_parameter('right_joint3').value
        self.duration = self.get_parameter('duration').value
        self.repeat = self.get_parameter('repeat').value
        self.repeat_interval = self.get_parameter('repeat_interval').value

        # Number of points in trajectory for smooth motion
        self.num_points = 100

        self.get_logger().info('Arm collision test node started')
        self.get_logger().info('Starting from attention pose (0.0) and moving to collision pose after 3 seconds')

        # Wait for publishers to be ready (check subscription count)
        self.get_logger().info('Waiting for publishers to be ready...')
        for _ in range(20):  # Wait up to 2 seconds
            left_count = self.left_arm_pub.get_subscription_count()
            right_count = self.right_arm_pub.get_subscription_count()
            if left_count > 0 or right_count > 0:
                self.get_logger().info(f'Publishers ready: left={left_count}, right={right_count}')
                break
            time.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0.1)
        else:
            self.get_logger().warn('No subscribers detected, but publishing anyway...')

        if not self.repeat:
            # First, move to attention pose (0.0) if not already there
            # Then move to collision pose
            self.get_logger().info('Step 1: Moving to attention pose (0.0)...')
            self.publish_attention_pose()
            time.sleep(0.5)  # Give time for attention pose command to be sent

            # Wait a bit before moving to collision pose
            self.get_logger().info('Step 2: Waiting 3 seconds, then moving to collision pose...')
            time.sleep(3.0)

            # Now move to collision pose
            self.get_logger().info('Step 3: Moving to collision pose...')
            self.publish_collision_positions()

            # Publish multiple times to ensure it's received
            for i in range(2):
                time.sleep(0.2)
                self.publish_collision_positions()
                self.get_logger().info(f'Published collision trajectory {i+1}/2')
        else:
            # If repeat is enabled, set up a timer
            self.timer = self.create_timer(self.repeat_interval, self.publish_collision_positions)
            self.get_logger().info(f'Repeating every {self.repeat_interval} seconds')

    def joint_state_callback(self, msg):
        """Update current joint positions from joint_states."""
        for i, joint in enumerate(self.left_joints):
            if joint in msg.name:
                idx = msg.name.index(joint)
                self.left_current_positions[i] = msg.position[idx]

        for i, joint in enumerate(self.right_joints):
            if joint in msg.name:
                idx = msg.name.index(joint)
                self.right_current_positions[i] = msg.position[idx]

        self.joint_states_received = True

    def create_smooth_trajectory(self, joint_names, start_pos, end_pos):
        """Create smooth trajectory using quintic polynomial (same as ffw_teleop)."""
        traj = JointTrajectory()
        traj.joint_names = joint_names
        # Note: header.stamp is not set in ffw_teleop, leaving it unset

        times = np.linspace(0, self.duration, self.num_points)

        for i in range(self.num_points):
            point = JointTrajectoryPoint()
            t = times[i]

            t_norm = t / self.duration
            t_norm2 = t_norm * t_norm
            t_norm3 = t_norm2 * t_norm
            t_norm4 = t_norm3 * t_norm
            t_norm5 = t_norm4 * t_norm

            # Quintic polynomial for smooth motion
            pos_coeff = 10 * t_norm3 - 15 * t_norm4 + 6 * t_norm5
            vel_coeff = (30 * t_norm2 - 60 * t_norm3 + 30 * t_norm4) / self.duration
            acc_coeff = (60 * t_norm - 180 * t_norm2 + 120 * t_norm3) / (self.duration * self.duration)

            positions = []
            velocities = []
            accelerations = []

            for j in range(len(joint_names)):
                pos = start_pos[j] + (end_pos[j] - start_pos[j]) * pos_coeff
                vel = (end_pos[j] - start_pos[j]) * vel_coeff
                acc = (end_pos[j] - start_pos[j]) * acc_coeff

                positions.append(pos)
                velocities.append(vel)
                accelerations.append(acc)

            point.positions = positions
            point.velocities = velocities
            point.accelerations = accelerations
            point.time_from_start.sec = int(times[i])
            point.time_from_start.nanosec = int((times[i] % 1) * 1e9)

            traj.points.append(point)

        return traj

    def publish_attention_pose(self):
        """Publish joint trajectory commands to move arms to attention pose (all zeros)."""
        attention_positions = [0.0] * 8

        # Use current positions as start if available, otherwise use zeros
        left_start = self.left_current_positions if self.joint_states_received else [0.0] * 8
        right_start = self.right_current_positions if self.joint_states_received else [0.0] * 8

        # Create smooth trajectories to attention pose
        left_trajectory = self.create_smooth_trajectory(
            self.left_joints,
            left_start,
            attention_positions
        )

        right_trajectory = self.create_smooth_trajectory(
            self.right_joints,
            right_start,
            attention_positions
        )

        self.left_arm_pub.publish(left_trajectory)
        self.right_arm_pub.publish(right_trajectory)

        self.get_logger().info('Published attention pose trajectories (all 0.0)')
        self.get_logger().info(f'Left arm: {left_start[:3]} -> {attention_positions[:3]}')
        self.get_logger().info(f'Right arm: {right_start[:3]} -> {attention_positions[:3]}')

    def publish_collision_positions(self):
        """Publish joint trajectory commands to make arms collide."""

        # Always start from attention pose (0.0) regardless of current position
        attention_positions = [0.0] * 8

        # Create smooth trajectories from attention pose to collision pose
        left_trajectory = self.create_smooth_trajectory(
            self.left_joints,
            attention_positions,
            self.left_collision_positions
        )

        right_trajectory = self.create_smooth_trajectory(
            self.right_joints,
            attention_positions,
            self.right_collision_positions
        )

        # Publish trajectories
        self.get_logger().info(f'Publishing left arm trajectory with {len(left_trajectory.points)} points')
        self.get_logger().info(f'Publishing right arm trajectory with {len(right_trajectory.points)} points')
        self.get_logger().info(f'Left joints: {left_trajectory.joint_names}')
        self.get_logger().info(f'Right joints: {right_trajectory.joint_names}')

        self.left_arm_pub.publish(left_trajectory)
        self.right_arm_pub.publish(right_trajectory)

        self.get_logger().info('Published collision trajectories')
        self.get_logger().info(f'Left arm: {attention_positions[:3]} -> {self.left_collision_positions[:3]}')
        self.get_logger().info(f'Right arm: {attention_positions[:3]} -> {self.right_collision_positions[:3]}')
        self.get_logger().info(f'Left topic: {self.left_arm_pub.topic_name}')
        self.get_logger().info(f'Right topic: {self.right_arm_pub.topic_name}')


def main(args=None):
    rclpy.init(args=args)

    node = ArmCollisionTest()

    try:
        if node.repeat:
            # Keep running and republishing
            rclpy.spin(node)
        else:
            # Keep node alive for a longer time to ensure messages are sent
            # The trajectory duration is typically 3 seconds, so wait at least that long
            node.get_logger().info('Keeping node alive to ensure trajectory is processed...')
            end_time = time.time() + node.duration + 2.0  # Wait duration + 2 seconds
            while time.time() < end_time:
                rclpy.spin_once(node, timeout_sec=0.1)
            node.get_logger().info('Trajectory should be complete, shutting down...')
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

