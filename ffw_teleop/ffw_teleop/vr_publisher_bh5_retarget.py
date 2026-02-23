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
# Authors: Wonho Yun

import asyncio
import math
import os
import socket
import threading
import traceback

from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, Quaternion, Point32
import nest_asyncio
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from vuer import Vuer
from vuer.base import Server
from vuer.schemas import Hands, Body, DefaultScene
from std_msgs.msg import Bool
from ffw_interfaces.msg import HandJoints

# Allow nested asyncio execution
nest_asyncio.apply()

# WebXR Body Tracking joint order (XRBodyJoint enum)
BODY_JOINT_KEYS = [
    "hips",
    "spine-lower",
    "spine-middle",
    "spine-upper",
    "chest",
    "neck",
    "head",
    "left-shoulder",
    "left-scapula",
    "left-arm-upper",
    "left-arm-lower",
    "left-hand-wrist-twist",
    "right-shoulder",
    "right-scapula",
    "right-arm-upper",
    "right-arm-lower",
    "right-hand-wrist-twist",
    "left-hand-palm",
    "left-hand-wrist",
    "left-hand-thumb-metacarpal",
    "left-hand-thumb-phalanx-proximal",
    "left-hand-thumb-phalanx-distal",
    "left-hand-thumb-tip",
    "left-hand-index-metacarpal",
    "left-hand-index-phalanx-proximal",
    "left-hand-index-phalanx-intermediate",
    "left-hand-index-phalanx-distal",
    "left-hand-index-tip",
    "left-hand-middle-phalanx-metacarpal",
    "left-hand-middle-phalanx-proximal",
    "left-hand-middle-phalanx-intermediate",
    "left-hand-middle-phalanx-distal",
    "left-hand-middle-tip",
    "left-hand-ring-metacarpal",
    "left-hand-ring-phalanx-proximal",
    "left-hand-ring-phalanx-intermediate",
    "left-hand-ring-phalanx-distal",
    "left-hand-ring-tip",
    "left-hand-little-metacarpal",
    "left-hand-little-phalanx-proximal",
    "left-hand-little-phalanx-intermediate",
    "left-hand-little-phalanx-distal",
    "left-hand-little-tip",
    "right-hand-palm",
    "right-hand-wrist",
    "right-hand-thumb-metacarpal",
    "right-hand-thumb-phalanx-proximal",
    "right-hand-thumb-phalanx-distal",
    "right-hand-thumb-tip",
    "right-hand-index-metacarpal",
    "right-hand-index-phalanx-proximal",
    "right-hand-index-phalanx-intermediate",
    "right-hand-index-phalanx-distal",
    "right-hand-index-tip",
    "right-hand-middle-metacarpal",
    "right-hand-middle-phalanx-proximal",
    "right-hand-middle-phalanx-intermediate",
    "right-hand-middle-phalanx-distal",
    "right-hand-middle-tip",
    "right-hand-ring-metacarpal",
    "right-hand-ring-phalanx-proximal",
    "right-hand-ring-phalanx-intermediate",
    "right-hand-ring-phalanx-distal",
    "right-hand-ring-tip",
    "right-hand-little-metacarpal",
    "right-hand-little-phalanx-proximal",
    "right-hand-little-phalanx-intermediate",
    "right-hand-little-phalanx-distal",
    "right-hand-little-tip",
    "left-upper-leg",
    "left-lower-leg",
    "left-foot-ankle-twist",
    "left-foot-ankle",
    "left-foot-subtalar",
    "left-foot-transverse",
    "left-foot-ball",
    "right-upper-leg",
    "right-lower-leg",
    "right-foot-ankle-twist",
    "right-foot-ankle",
    "right-foot-subtalar",
    "right-foot-transverse",
    "right-foot-ball",
]


class VRTrajectoryPublisher(Node):

    def __init__(self):
        super().__init__('vr_trajectory_publisher')
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)

        # VR publishing control flag
        self.vr_publishing_enabled = False  # Default: disabled

        # VR Server setup
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cert_file = os.path.join(current_dir, 'cert.pem')
        key_file = os.path.join(current_dir, 'key.pem')
        hostname = socket.gethostbyname(socket.gethostname())
        ws_url = f'ws://{hostname}:8012'

        self.vuer = Vuer(
            host='0.0.0.0',
            port=8012,
            cert=cert_file,
            key=key_file,
            ws=ws_url,
            queries=dict(grid=False, reconnect=True),
            queue_len=3
        )

        self.fps = 30
        self.get_logger().info(f'VR Trajectory server available at: https://{hostname}:8012')

        # VR event handlers
        self.vuer.add_handler('HAND_MOVE')(self.on_hand_move)
        self.vuer.add_handler('CAMERA_MOVE')(self.on_camera_move)
        self.vuer.add_handler('BODY_MOVE')(self.on_body_tracking_move)

        # Publishers - Direct trajectory publishing without PoseArray
        self.left_hand_pos_pub = self.create_publisher(
            HandJoints,
            '/left_hand/hand_joint_pos',
            10
        )
        self.right_hand_pos_pub = self.create_publisher(
            HandJoints,
            '/right_hand/hand_joint_pos',
            10
        )

        self.left_wrist_rviz_pub = self.create_publisher(PoseStamped, '/l_goal_pose', 10)
        self.right_wrist_rviz_pub = self.create_publisher(PoseStamped, '/r_goal_pose', 10)
        self.left_elbow_pub = self.create_publisher(PoseStamped, '/l_elbow_pose', 10)
        self.right_elbow_pub = self.create_publisher(PoseStamped, '/r_elbow_pose', 10)

        # Subscriber for VR control toggle
        self.vr_control_sub = self.create_subscription(
            Bool,
            '/vr_control/toggle',
            self.vr_control_callback,
            10
        )

        self.required_vr_frames = [0,
                                   1,2,3,4,
                                   6,7,8,9,
                                   11,12,13,14,
                                   16,17,18,19,
                                   21,22,23,24]

        self.vr_hand_to_urdf = np.array([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1]
        ])

        self.prev_poses_right = np.zeros((21,3))
        self.start_poses_right = False

        self.prev_poses_left = np.zeros((21,3))
        self.start_poses_left = False

        # VR data storage
        self.left_hand_data = None
        self.right_hand_data = None
        self.head_transform_matrix = np.eye(4)
        self.head_inverse_matrix = np.eye(4)
        self.hand_pose_is_head_relative = self.declare_parameter(
            'hand_pose_is_head_relative', True
        ).value
        self.zero_z_on_start = self.declare_parameter(
            'zero_z_on_start', True
        ).value
        self.z_calibrated = False
        self.z_calibration_offset = 0.0

        # Low-pass filter settings
        self.low_pass_filter_alpha = 0.5
        self.pose_filters = {}
        self.max_elbow_wrist_distance = 0.4
        self.max_wrist_angle_step_deg = 30.0

        # Scaling VR data
        self.scaling_vr = 1.1
        self.wrist_vr_scale = 1.4
        self.elbow_vr_scale = 1.4
        self.wrist_offsets = {'x': -0.14, 'y': 0.0, 'z': 1.0}
        self.elbow_offsets = {'x': -0.14, 'y': 0.0, 'z': 1.0}

        # Head pitch offset configuration
        self.pitch_offset = -0.5  # Adjustable pitch offset in radians

        self.hand_log_counter = 0

        # Status monitoring timer (every 5 seconds)
        self.status_timer = self.create_timer(5.0, self.log_status)

        # Logging counters
        self.head_log_counter = 0
        self.log_every_n = self.fps

        # Start Vuer server in background thread
        self.start_vuer_server()

        self.get_logger().info('VR Trajectory Publisher node has been started')
        self.get_logger().info('VR publishing is DISABLED by default. Send /vr_control/toggle message (True=enable, False=disable).')

    def vr_control_callback(self, msg):
        """Callback to enable/disable VR publishing based on message content."""
        new_state = bool(msg.data)  # Read message content

        # Only log if state actually changed
        if new_state != self.vr_publishing_enabled:
            self.vr_publishing_enabled = new_state
            status = "ENABLED" if self.vr_publishing_enabled else "DISABLED"
            self.get_logger().info(f'VR publishing changed to: {status} (message value: {msg.data})')

            if not self.vr_publishing_enabled:
                # Reset joint positions to zero when disabled
                self.left_joint_positions = [0.0] * 20
                self.right_joint_positions = [0.0] * 20
                self.get_logger().info('Joint positions reset to zero')

    def log_status(self):
        """Log current system status for debugging."""
        vr_status = "ENABLED" if self.vr_publishing_enabled else "DISABLED"

        self.get_logger().info(f'Status: VR={vr_status}')

        # Check if we have hand data
        left_data_status = "Available" if self.left_hand_data is not None else "None"
        right_data_status = "Available" if self.right_hand_data is not None else "None"

        # self.get_logger().info(f'Hand data: Left={left_data_status}, Right={right_data_status}')

        # Check joint positions
        # if self.left_joint_positions and any(pos != 0.0 for pos in self.left_joint_positions):
        #     self.get_logger().info(f'Left joint positions: {self.left_joint_positions[:5]}...')
        # if self.right_joint_positions and any(pos != 0.0 for pos in self.right_joint_positions):
        #     self.get_logger().info(f'Right joint positions: {self.right_joint_positions[:5]}...')


    def is_valid_float(self, value):
        """Check if value is valid float (excluding NaN, inf)."""
        return isinstance(value, (int, float)) and np.isfinite(value)

    def safe_point(self, x, y, z):
        """Create safe Point (filtering NaN/inf values)."""
        safe_x = float(x) if self.is_valid_float(x) else 0.0
        safe_y = float(y) if self.is_valid_float(y) else 0.0
        safe_z = float(z) if self.is_valid_float(z) else 0.0
        return Point(x=safe_x, y=safe_y, z=safe_z)

    def safe_quaternion(self, x, y, z, w):
        """Create safe Quaternion (filtering NaN/inf values)."""
        safe_x = float(x) if self.is_valid_float(x) else 0.0
        safe_y = float(y) if self.is_valid_float(y) else 0.0
        safe_z = float(z) if self.is_valid_float(z) else 0.0
        safe_w = float(w) if self.is_valid_float(w) else 1.0
        return Quaternion(x=safe_x, y=safe_y, z=safe_z, w=safe_w)

    def matrix_to_pose(self, mat):
        """Convert 4x4 transformation matrix to (position, quaternion)."""
        pos = mat[:3, 3]
        rot = mat[:3, :3]

        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(rot)):
            self.get_logger().warn('Invalid matrix data detected, using default pose')
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])

        trace = rot[0, 0] + rot[1, 1] + rot[2, 2]

        if 1 + trace <= 0:
            quat = np.array([0.0, 0.0, 0.0, 1.0])
            return pos, quat

        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * s
            qx = (rot[2, 1] - rot[1, 2]) / s
            qy = (rot[0, 2] - rot[2, 0]) / s
            qz = (rot[1, 0] - rot[0, 1]) / s
        elif ((rot[0, 0] > rot[1, 1]) and (rot[0, 0] > rot[2, 2])):
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s

        quat = np.array([qx, qy, qz, qw])

        if not np.all(np.isfinite(quat)):
            quat = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            norm = np.linalg.norm(quat)
            if norm > 0:
                quat = quat / norm
            else:
                quat = np.array([0.0, 0.0, 0.0, 1.0])

        return pos, quat

    def vr_to_ros_transform(self, vr_pos, vr_quat):
        """Transform from VR coordinate system to ROS coordinate system."""
        vr_to_ros_matrix = np.array([
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0]
        ])

        ros_pos = vr_to_ros_matrix @ vr_pos

        vr_rotation = R.from_quat([vr_quat[0], vr_quat[1], vr_quat[2], vr_quat[3]])
        vr_rot_matrix = vr_rotation.as_matrix()
        ros_rot_matrix = vr_to_ros_matrix @ vr_rot_matrix
        ros_rotation = R.from_matrix(ros_rot_matrix)
        ros_quat = ros_rotation.as_quat()

        return ros_pos, ros_quat

    def transform_and_publish_pose(self, pose_array_msg, publisher, hand_name, vr_scale=1.0):
        """Transform wrist pose from head relative coordinates to base_link and publish."""
        if not pose_array_msg.poses:
            return

        # Assume the first pose in the array is the wrist pose (head relative, ROS coordinates)
        wrist_pose_relative = pose_array_msg.poses[0]

        camera_relative_position = np.array([
            wrist_pose_relative.position.x,
            wrist_pose_relative.position.y,
            wrist_pose_relative.position.z
        ], dtype=np.float64)

        camera_relative_quaternion = np.array([
            wrist_pose_relative.orientation.x,
            wrist_pose_relative.orientation.y,
            wrist_pose_relative.orientation.z,
            wrist_pose_relative.orientation.w
        ], dtype=np.float64)

        apply_right_z_flip = (hand_name == 'right')
        self.publish_relative_pose(
            camera_relative_position,
            camera_relative_quaternion,
            publisher,
            vr_scale=vr_scale,
            x_offset=self.wrist_offsets['x'],
            y_offset=self.wrist_offsets['y'],
            z_offset=self.wrist_offsets['z'],
            apply_right_z_flip=apply_right_z_flip,
            pose_role='wrist',
            side=hand_name,
        )

    def publish_relative_pose(
        self,
        camera_relative_position,
        camera_relative_quaternion,
        publisher,
        vr_scale=1.0,
        x_offset=0.0,
        y_offset=0.0,
        z_offset=0.0,
        apply_right_z_flip=False,
        pose_role='wrist',
        side='',
    ):
        """Publish a pose using the same base_link transform as wrists."""
        # Fixed offset: zedm_camera_center → base_link
        zedm_to_base_offset = np.array([
            0.0 - 0.0238122 - 0.040 - 0.049483 - 0.0055,  # x: -0.1187952
            0.0 + 0.0 + 0.0 + 0.0 + 0.0,                  # y: 0.0
            -0.01325 + 0.0242094 - 0.054 - 0.102130 - 1.4316  # z: -1.5767706
        ], dtype=np.float64)

        base_position = (camera_relative_position * vr_scale) - zedm_to_base_offset
        if self.zero_z_on_start:
            if (not self.z_calibrated) and pose_role == 'wrist':
                self.z_calibration_offset = base_position[2]
                self.z_calibrated = True
            if self.z_calibrated:
                base_position = base_position.copy()
                base_position[2] -= self.z_calibration_offset

        camera_relative_rotation = R.from_quat(camera_relative_quaternion)
        if apply_right_z_flip:
            rot_z_180 = R.from_euler('z', 180, degrees=True)
            camera_relative_rotation = camera_relative_rotation * rot_z_180
        arm_quaternion = camera_relative_rotation.as_quat()  # [x, y, z, w]

        pose_key = f'{side}_{pose_role}' if side else pose_role
        base_position, arm_quaternion = self.low_pass_filter_pose(
            pose_key,
            base_position,
            arm_quaternion,
            max_angle_deg=self.max_wrist_angle_step_deg if pose_role == 'wrist' else None,
        )
        if side:
            base_position = self.apply_elbow_wrist_safety(side, pose_role, base_position)

        target_pose = PoseStamped()
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.header.frame_id = 'base_link'

        target_pose.pose.position.x = base_position[0] + x_offset
        target_pose.pose.position.y = base_position[1] + y_offset
        target_pose.pose.position.z = base_position[2] + z_offset

        target_pose.pose.orientation.x = arm_quaternion[0]
        target_pose.pose.orientation.y = arm_quaternion[1]
        target_pose.pose.orientation.z = arm_quaternion[2]
        target_pose.pose.orientation.w = arm_quaternion[3]

        publisher.publish(target_pose)

    def get_joint_matrix(self, hand_data, joint_index):
        """Extract joint transformation matrix from hand data."""
        arr = np.array(hand_data)
        start_idx = joint_index * 16
        end_idx = start_idx + 16
        matrix_data = arr[start_idx:end_idx]
        return matrix_data.reshape(4, 4, order='F')

    def quat_inverse(self, q):
        """Returns the inverse of a quaternion."""
        norm = q.x**2 + q.y**2 + q.z**2 + q.w**2
        if norm == 0:
            return Quaternion()
        inv_norm = 1.0 / norm
        msg = Quaternion()
        msg.x = -q.x * inv_norm
        msg.y = -q.y * inv_norm
        msg.z = -q.z * inv_norm
        msg.w = q.w * inv_norm
        return msg

    def quat_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
        x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
        y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x
        z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
        msg = Quaternion()
        msg.x = x
        msg.y = y
        msg.z = z
        msg.w = w
        return msg

    def get_roll_pitch_yaw(self, q1, q2, cmd=''):
        """Calculate roll, pitch, yaw from two quaternions."""
        q_combined = self.quat_multiply(q1, q2)
        w, x, y, z = q_combined.w, q_combined.x, q_combined.y, q_combined.z

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x**2 + y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        sinp = np.clip(sinp, -1, 1)
        pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y**2 + z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        if cmd == 'r':
            return roll
        elif cmd == 'p':
            return pitch
        elif cmd == 'y':
            return yaw
        else:
            return roll, pitch, yaw

    def wrap_pi(self, angle):
        """Wrap angle to [-pi, pi] range."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix."""
        x, y, z, w = q[0], q[1], q[2], q[3]
        rot_matrix = np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
        ])
        return rot_matrix

    def process_hand_joints(self, hand_data, side='left'):
        """Process VR hand data and calculate joint positions with low-pass filtering."""
        if hand_data is None or len(hand_data) != 400:
            return

        # Extract hand pose quaternions
        hand_joints = HandJoints()
        hand_joints.header.stamp = self.get_clock().now().to_msg()
        hand_joints.header.frame_id = ''
        hand_joints.joints = []
        temp_joints = np.zeros((21,3))
        pose_counter = 0
        wrist_quat = np.zeros(4)
        wrist_rot = np.eye(3)

        # For wrist
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = ''
        pose_array.poses = []

        for i in self.required_vr_frames:
            try:
                world_joint_matrix = self.get_joint_matrix(hand_data, i)
                if self.hand_pose_is_head_relative:
                    relative_joint_matrix = world_joint_matrix
                else:
                    relative_joint_matrix = self.head_inverse_matrix @ world_joint_matrix

                relative_pos, relative_quat = self.matrix_to_pose(relative_joint_matrix)
                temp_joints[pose_counter,:] = relative_pos
                pose_counter += 1

                if i == 0:
                    wrist_quat = relative_quat
                    wrist_rot = self.quaternion_to_rotation_matrix(wrist_quat)

                    relative_pos_ros, relative_quat_ros = self.vr_to_ros_transform(relative_pos, relative_quat)

                    quat = Quaternion()
                    quat.x = relative_quat_ros[0]
                    quat.y = relative_quat_ros[1]
                    quat.z = relative_quat_ros[2]
                    quat.w = relative_quat_ros[3]

                    pose_msg = Pose()
                    pose_msg.position = self.safe_point(relative_pos_ros[0], relative_pos_ros[1], relative_pos_ros[2])
                    pose_msg.orientation = quat
                    pose_array.poses.append(pose_msg)

            except Exception as e:
                self.get_logger().warn(f'Error processing hand joint {i}: {e}')
                return

        if side == 'left':
            # Transform and publish left wrist pose to base_link coordinates
            self.transform_and_publish_pose(
                pose_array, self.left_wrist_rviz_pub, 'left', vr_scale=self.wrist_vr_scale)

            # Apply low-pass filter
            if self.start_poses_left:
                temp_joints = self.low_pass_filter_alpha * temp_joints + (1-self.low_pass_filter_alpha) * self.prev_poses_left
            else:
                self.prev_poses_left = temp_joints

            for i in range(21):
                temp_position = self.vr_hand_to_urdf @ wrist_rot.T @ (temp_joints[i,:] - temp_joints[0,:])
                temp_point = Point32()
                temp_point.x = temp_position[0]
                temp_point.y = temp_position[1]
                temp_point.z = temp_position[2]
                hand_joints.joints.append(temp_point)

            self.left_hand_pos_pub.publish(hand_joints)

        elif side == 'right':
            # Transform and publish left wrist pose to base_link coordinates
            self.transform_and_publish_pose(
                pose_array, self.right_wrist_rviz_pub, 'right', vr_scale=self.wrist_vr_scale)

            # Apply low-pass filter
            if self.start_poses_right:
                temp_joints = self.low_pass_filter_alpha * temp_joints + (1-self.low_pass_filter_alpha) * self.prev_poses_right
            else:
                self.prev_poses_right = temp_joints

            for i in range(21):
                temp_position = self.vr_hand_to_urdf @ wrist_rot.T @ (temp_joints[i,:] - temp_joints[0,:])
                temp_point = Point32()
                temp_point.x = temp_position[0]
                temp_point.y = temp_position[1]
                temp_point.z = temp_position[2]
                hand_joints.joints.append(temp_point)

            self.right_hand_pos_pub.publish(hand_joints)

    # def publish_hand_trajectory(self):
    #     """Publish hand joint trajectory directly only if enabled."""
    #     if not self.vr_publishing_enabled:
    #         return

    #     left_msg = JointTrajectory()
    #     left_msg.joint_names = self.left_joint_names
    #     left_goal_point = JointTrajectoryPoint()
    #     left_goal_point.positions = self.left_joint_positions.copy()
    #     left_goal_point.time_from_start.sec = 0
    #     left_goal_point.time_from_start.nanosec = 0
    #     left_msg.points.append(left_goal_point)
    #     self.left_hand_trajectory_pub.publish(left_msg)

    #     right_msg = JointTrajectory()
    #     right_msg.joint_names = self.right_joint_names
    #     right_goal_point = JointTrajectoryPoint()
    #     right_goal_point.positions = self.right_joint_positions.copy()
    #     right_goal_point.time_from_start.sec = 0
    #     right_goal_point.time_from_start.nanosec = 0
    #     right_msg.points.append(right_goal_point)
    #     self.right_hand_trajectory_pub.publish(right_msg)

    def start_vuer_server(self):
        """Start the VR server in a separate thread."""
        def run_server():
            try:
                self.vuer.spawn(self.main_hand_tracking)
                self.get_logger().info('Starting VR server...')
                self.start_vuer_server_no_super()
            except Exception as e:
                self.get_logger().error(f'Failed to start VR server: {e}')
                self.get_logger().error(traceback.format_exc())

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

    def start_vuer_server_no_super(self):
        """Start Vuer server without zero-arg super() usage."""
        # Replicate vuer.server.Vuer.start setup, but call Server.start directly.
        self.vuer._add_route("", self.vuer.socket_index, method="GET")
        self.vuer._add_static("/assets", self.vuer.client_root / "assets")
        self.vuer._static_file("/editor", self.vuer.client_root, "editor/index.html")
        self.vuer._add_static("/static", self.vuer.static_root)
        self.vuer._add_route("/relay", self.vuer.relay, method="POST")

        # Start base Server without Vuer.start() super() call.
        Server.start(self.vuer)

    async def main_hand_tracking(self, session):
        """Main hand tracking session."""
        try:
            fps = self.fps
            self.get_logger().info('Starting hand tracking session')
            session.set @ DefaultScene(
                grid=False,
            )
            session.upsert(
                Hands(
                    fps=fps,
                    stream=True,
                    key='hands',
                    hideLeft=False,
                    hideRight=False,
                ),
                to='bgChildren',
            )
            session.upsert(
                Body(
                    key='body_tracking',
                    stream=True,
                    fps=fps,
                    leftHand=False,
                    rightHand=False,
                    hideIndicate=False,
                    showFrame=True,
                    showBody=True,
                    frameScale=0.02,
                ),
                to='children',
            )
            self.get_logger().info('Hand tracking enabled')
            while True:
                await asyncio.sleep(1/fps)
        except Exception as e:
            self.get_logger().error(f'Error in hand tracking session: {e}')

    async def on_camera_move(self, event, session):
        """Handle camera movement events."""
        try:
            if not self.vr_publishing_enabled:
                return

            if getattr(event, 'key', None) != 'defaultCamera':
                return

            matrix = None
            if isinstance(event.value, dict):
                if 'matrix' in event.value:
                    matrix = event.value['matrix']
                elif 'camera' in event.value and isinstance(event.value['camera'], dict):
                    matrix = event.value['camera'].get('matrix')

            if isinstance(matrix, (list, np.ndarray)) and len(matrix) == 16:
                self.head_transform_matrix = np.array(matrix).reshape(4, 4, order='F')
                self.head_inverse_matrix = np.linalg.inv(self.head_transform_matrix)
                pos, quat = self.matrix_to_pose(self.head_transform_matrix)

                # Publish head joint trajectory
                r = R.from_quat(quat)
                roll, pitch, yaw = r.as_euler('zxy')

                if self.is_valid_float(pitch) and self.is_valid_float(yaw):
                    msg = JointTrajectory()
                    msg.joint_names = ['head_joint1', 'head_joint2']
                    point = JointTrajectoryPoint()
                    # Apply pitch offset
                    adjusted_pitch = pitch + self.pitch_offset
                    point.positions = [-float(adjusted_pitch), float(yaw)]
                    point.velocities = [0.0, 0.0]
                    point.accelerations = [0.0, 0.0]
                    point.effort = []
                    msg.points.append(point)
                    # self.head_joint_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f'Error in camera move event: {e}')

    async def on_hand_move(self, event, session):
        """Handle hand movement events."""
        try:
            if not self.vr_publishing_enabled:
                return

            # Process hand data
            if 'left' in event.value:
                left_data = event.value['left']
                if isinstance(left_data, (list, np.ndarray)) and len(left_data) == 400:
                    self.left_hand_data = np.array(left_data)
                    # Process left hand joints directly
                    self.process_hand_joints(self.left_hand_data,'left')

            # Process hand data
            if 'right' in event.value:
                right_data = event.value['right']
                if isinstance(right_data, (list, np.ndarray)) and len(right_data) == 400:
                    self.right_hand_data = np.array(right_data)
                    # Process right hand joints directly
                    self.process_hand_joints(self.right_hand_data,'right')

            # Debug logging
            # self.hand_log_counter += 1
            # if self.hand_log_counter % self.log_every_n == 0:
            #     self.get_logger().info('Hand tracking data processed')

        except Exception as e:
            self.get_logger().error(f'Error in hand move event: {e}')

    async def on_body_tracking_move(self, event, session):
        """Handle body tracking events (elbow poses)."""
        try:
            if not self.vr_publishing_enabled:
                return

            if not isinstance(event.value, dict) or not event.value:
                return

            body_data = event.value.get('body') if isinstance(event.value, dict) else None
            if not isinstance(body_data, (list, tuple, np.ndarray)):
                return

            # WebXR Body Tracking spec joints use left-arm-lower/right-arm-lower
            left_matrix = self.get_body_joint_matrix_from_flat(
                body_data,
                'left-arm-lower'
            )
            right_matrix = self.get_body_joint_matrix_from_flat(
                body_data,
                'right-arm-lower'
            )

            if left_matrix is not None:
                self.publish_body_joint_pose(left_matrix, self.left_elbow_pub, side='left')
            if right_matrix is not None:
                self.publish_body_joint_pose(right_matrix, self.right_elbow_pub, side='right')

        except Exception as e:
            self.get_logger().error(f'Error in body tracking event: {e}')

    def get_body_joint_matrix_from_flat(self, body_array, joint_name):
        """Extract a 4x4 matrix for a body joint from flattened body array."""
        if joint_name not in BODY_JOINT_KEYS:
            return None
        index = BODY_JOINT_KEYS.index(joint_name)
        start = index * 16
        end = start + 16
        if len(body_array) < end:
            return None
        matrix = np.array(body_array[start:end], dtype=np.float64)
        return matrix.reshape(4, 4, order='F')

    def publish_body_joint_pose(self, joint_matrix, publisher, side=''):
        """Publish PoseStamped for a body joint."""
        relative_joint_matrix = self.head_inverse_matrix @ joint_matrix
        pos, quat = self.matrix_to_pose(relative_joint_matrix)
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
            return

        pos_ros, quat_ros = self.vr_to_ros_transform(pos, quat)
        self.publish_relative_pose(
            pos_ros,
            quat_ros,
            publisher,
            vr_scale=self.elbow_vr_scale,
            x_offset=self.elbow_offsets['x'],
            y_offset=self.elbow_offsets['y'],
            z_offset=self.elbow_offsets['z'],
            apply_right_z_flip=False,
            pose_role='elbow',
            side=side,
        )

    def low_pass_filter_pose(self, key, position, quaternion, max_angle_deg=None):
        """Apply low-pass filter to position and quaternion."""
        quat = np.array(quaternion, dtype=np.float64)
        quat_norm = np.linalg.norm(quat)
        if not np.isfinite(quat_norm) or quat_norm <= 0.0:
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            quat = quat / quat_norm
        if key not in self.pose_filters:
            self.pose_filters[key] = {
                'pos': np.array(position, dtype=np.float64),
                'quat': quat,
            }
            return position, quat

        prev = self.pose_filters[key]
        alpha = 0.5
        filtered_pos = alpha * position + (1.0 - alpha) * prev['pos']
        prev_quat = prev['quat']
        if np.dot(prev_quat, quat) < 0.0:
            quat = -quat
        filtered_quat = self.slerp_quaternion(prev_quat, quat, alpha)
        if max_angle_deg is not None:
            filtered_quat = self.limit_quaternion_spike(prev_quat, filtered_quat, max_angle_deg)

        self.pose_filters[key]['pos'] = filtered_pos
        self.pose_filters[key]['quat'] = filtered_quat
        return filtered_pos, filtered_quat

    def limit_quaternion_spike(self, prev_quat, current_quat, max_angle_deg):
        """Clamp quaternion step by max angle in degrees."""
        prev_quat = np.array(prev_quat, dtype=np.float64)
        curr_quat = np.array(current_quat, dtype=np.float64)
        if np.dot(prev_quat, curr_quat) < 0.0:
            curr_quat = -curr_quat
        dot = float(np.clip(np.dot(prev_quat, curr_quat), -1.0, 1.0))
        angle = 2.0 * math.acos(dot)
        max_angle = math.radians(max_angle_deg)
        if angle <= max_angle or angle <= 1.0e-6:
            return curr_quat
        t = max_angle / angle
        return self.slerp_quaternion(prev_quat, curr_quat, t)

    def slerp_quaternion(self, q0, q1, t):
        """Spherical linear interpolation for quaternions."""
        q0 = np.array(q0, dtype=np.float64)
        q1 = np.array(q1, dtype=np.float64)
        dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            norm = np.linalg.norm(result)
            return result / norm if norm > 0.0 else q0
        theta_0 = math.acos(dot)
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return s0 * q0 + s1 * q1

    def apply_elbow_wrist_safety(self, side, pose_role, position):
        """Clamp wrist-elbow distance to safety limit (wrist has priority)."""
        if pose_role not in ('wrist', 'elbow'):
            return position
        if pose_role == 'wrist':
            return position
        other_key = f'{side}_wrist'
        if other_key not in self.pose_filters:
            return position
        other_pos = self.pose_filters[other_key]['pos']
        delta = position - other_pos
        dist = np.linalg.norm(delta)
        if dist > self.max_elbow_wrist_distance and dist > 0.0:
            position = other_pos + delta * (self.max_elbow_wrist_distance / dist)
        return position

    def __del__(self):
        try:
            if hasattr(self, 'vuer') and hasattr(self.vuer, 'stop'):
                self.vuer.stop()
        except Exception as e:
            self.get_logger().error(f'Error in cleanup: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = VRTrajectoryPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
