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

from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, Quaternion, Twist
from sensor_msgs.msg import JointState
import nest_asyncio
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from vuer import Vuer, VuerSession
from vuer.schemas import Hands, MotionControllers, DefaultScene
from std_msgs.msg import Bool

# Allow nested asyncio execution
nest_asyncio.apply()


class VRTrajectoryPublisher(Node):

    def __init__(self):
        super().__init__('vr_trajectory_publisher')
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)

        # VR publishing control flag
        self.vr_publishing_enabled = True #False  # Default: disabled

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
        self.vuer.add_handler('CAMERA_MOVE')(self.on_camera_move)
        self.vuer.add_handler('CONTROLLER_MOVE')(self.on_controller_move)

        # Controller data storage
        self.left_controller_data = None
        self.right_controller_data = None
        self.left_controller_state = None
        self.right_controller_state = None

        # Squeeze tracking for enabling publishing
        self.left_squeeze_start_time = None
        self.right_squeeze_start_time = None
        self.squeeze_hold_duration = 2.0  # seconds
        self.publishing_enabled_by_squeeze = False

        # Thumbstick mode tracking
        # joystick_mode: True = lift+head mode, False = lift+cmd_vel mode
        # Arm control always works regardless of mode
        self.joystick_mode = True  # True: lift+head, False: lift+cmd_vel
        self.prev_left_thumbstick_pressed = False
        self.prev_right_thumbstick_pressed = False

        # Cmd_vel scaling factors (similar to joystick_controller.cpp)
        self.linear_x_scale = 2.0
        self.linear_y_scale = 2.0
        self.angular_z_scale = 2.0

        # Joystick control parameters (similar to joystick_controller.cpp)
        self.jog_scale = 0.015  # Default jog scale (DEFAULT_JOG_SCALE)
        self.deadzone = 0.25  # Deadzone for thumbstick values (0.0 to 1.0)
        self.current_joint_states = None
        self.lift_joint_current_position = 0.0
        self.head_joint1_current_position = 0.0
        self.head_joint2_current_position = 0.0

        # VR data storage
        self.left_hand_data = None
        self.right_hand_data = None
        self.head_transform_matrix = np.eye(4)
        self.head_inverse_matrix = np.eye(4)

        # Logging counters
        self.hand_log_counter = 0
        self.log_every_n = self.fps

        # Publishers for controller poses (matching vr_publisher_bh5.py naming)
        self.left_controller_wrist_pub = self.create_publisher(PoseStamped, '/vr_hand/left_wrist', 10)
        self.right_controller_wrist_pub = self.create_publisher(PoseStamped, '/vr_hand/right_wrist', 10)

        # Publishers for joystick control
        self.left_joystick_pub = self.create_publisher(JointTrajectory, '/leader/joystick_controller_left/joint_trajectory', 10)
        self.right_joystick_pub = self.create_publisher(JointTrajectory, '/leader/joystick_controller_right/joint_trajectory', 10)

        # Publisher for swerve mode (base velocity)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for joint states (needed for joystick control)
        self.joint_states_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )

        # Subscriber for VR control toggle
        self.vr_control_sub = self.create_subscription(
            Bool,
            '/vr_control/toggle',
            self.vr_control_callback,
            10
        )

        # Scaling VR data
        self.scaling_vr = 1.1

        # Async setup
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.start_vuer_server()

        self.get_logger().info('VR Trajectory Publisher node has been started')
        self.get_logger().info('VR publishing is DISABLED by default. Send /vr_control/toggle message (True=enable, False=disable).')



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

    def calculate_base_position(self, controller_matrix):
        """Calculate base_link position from controller matrix (similar to transform_and_publish_pose in vr_publisher_bh5.py)."""
        # Transform to head relative coordinates
        if self.head_inverse_matrix is not None and np.all(np.isfinite(self.head_inverse_matrix)):
            relative_matrix = self.head_inverse_matrix @ controller_matrix
            relative_pos, relative_quat = self.matrix_to_pose(relative_matrix)
            relative_pos_ros, relative_quat_ros = self.vr_to_ros_transform(relative_pos, relative_quat)
        else:
            vr_world_pos, vr_world_quat = self.matrix_to_pose(controller_matrix)
            relative_pos_ros, relative_quat_ros = self.vr_to_ros_transform(vr_world_pos, vr_world_quat)

        # Scale VR position
        camera_relative_position = relative_pos_ros * self.scaling_vr

        # Fixed offset: zedm_camera_center → base_link
        zedm_to_base_offset = np.array([
            0.0 - 0.0238122 - 0.040 - 0.049483 - 0.0055,  # x: -0.1187952
            0.0 + 0.0 + 0.0 + 0.0 + 0.0,                  # y: 0.0
            -0.01325 + 0.0242094 - 0.054 - 0.102130 - 1.4316  # z: -1.5767706
        ], dtype=np.float64)

        # Transform to base_link coordinates
        base_position = camera_relative_position - zedm_to_base_offset

        return base_position, relative_quat_ros

    def apply_deadzone(self, value):
        """Apply deadzone to thumbstick value (similar to joystick_controller.cpp)."""
        abs_value = abs(value)
        
        # If value is within deadzone, return 0
        if abs_value < self.deadzone:
            return 0.0
        
        # Normalize after deadzone
        sign = 1.0 if value >= 0.0 else -1.0
        normalized_value = (abs_value - self.deadzone) / (1.0 - self.deadzone)
        
        return sign * normalized_value

    def joint_states_callback(self, msg):
        """Callback to receive joint states."""
        self.current_joint_states = msg
        # Update joint current positions
        if 'lift_joint' in msg.name:
            idx = msg.name.index('lift_joint')
            old_position = self.lift_joint_current_position
            self.lift_joint_current_position = msg.position[idx]
            # Log if position changed significantly
            if abs(old_position - self.lift_joint_current_position) > 0.01:
                self.get_logger().debug(
                    f'[JOINT_STATES] lift_joint: {old_position:.3f} -> {self.lift_joint_current_position:.3f}'
                )
        if 'head_joint1' in msg.name:
            idx = msg.name.index('head_joint1')
            self.head_joint1_current_position = msg.position[idx]
        if 'head_joint2' in msg.name:
            idx = msg.name.index('head_joint2')
            self.head_joint2_current_position = msg.position[idx]

    def vr_control_callback(self, msg):
        """Callback to enable/disable VR publishing based on message content."""
        new_state = bool(msg.data)  # Read message content

        # Only log if state actually changed
        if new_state != self.vr_publishing_enabled:
            self.vr_publishing_enabled = new_state
            status = "ENABLED" if self.vr_publishing_enabled else "DISABLED"
            self.get_logger().info(f'VR publishing changed to: {status} (message value: {msg.data})')

    def check_squeeze_condition(self):
        """Check if both squeezes are held for 2 seconds or more."""
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Extract squeeze values from controller states
        left_squeeze = 0.0
        right_squeeze = 0.0

        if isinstance(self.left_controller_state, dict) and 'squeezeValue' in self.left_controller_state:
            left_squeeze = float(self.left_controller_state.get('squeezeValue', 0.0))
        elif isinstance(self.left_controller_state, (int, float)):
            left_squeeze = float(self.left_controller_state)

        if isinstance(self.right_controller_state, dict) and 'squeezeValue' in self.right_controller_state:
            right_squeeze = float(self.right_controller_state.get('squeezeValue', 0.0))
        elif isinstance(self.right_controller_state, (int, float)):
            right_squeeze = float(self.right_controller_state)

        # Check if both squeezes are pressed (non-zero)
        both_squeezed = left_squeeze > 0.0 and right_squeeze > 0.0

        if both_squeezed:
            # Start tracking time if not already started
            if self.left_squeeze_start_time is None or self.right_squeeze_start_time is None:
                self.left_squeeze_start_time = current_time
                self.right_squeeze_start_time = current_time
                self.get_logger().info(f'[SQUEEZE] Both squeezes pressed: left={left_squeeze:.3f}, right={right_squeeze:.3f}, starting timer')

            # Check if 2 seconds have passed
            elapsed = current_time - max(self.left_squeeze_start_time, self.right_squeeze_start_time)
            if elapsed >= self.squeeze_hold_duration:
                if not self.publishing_enabled_by_squeeze:
                    self.publishing_enabled_by_squeeze = True
                    self.get_logger().info(f'[SQUEEZE] Publishing ENABLED: Both squeezes held for {elapsed:.2f} seconds')
            # Don't log debug messages too frequently
        else:
            # Reset if either squeeze is released
            if self.left_squeeze_start_time is not None or self.right_squeeze_start_time is not None:
                self.left_squeeze_start_time = None
                self.right_squeeze_start_time = None
                if self.publishing_enabled_by_squeeze:
                    self.publishing_enabled_by_squeeze = False
                    self.get_logger().info(f'[SQUEEZE] Publishing DISABLED: Squeeze released (left={left_squeeze:.3f}, right={right_squeeze:.3f})')

        return self.publishing_enabled_by_squeeze

    def process_thumbstick(self):
        """Process thumbstick input for mode switching and joystick control."""
        try:
            # Extract thumbstick values
            left_thumbstick_pressed = False
            right_thumbstick_pressed = False
            left_thumbstick_value = [0.0, 0.0]
            right_thumbstick_value = [0.0, 0.0]

            if isinstance(self.left_controller_state, dict):
                if 'thumbstick' in self.left_controller_state:
                    left_thumbstick_pressed = bool(self.left_controller_state.get('thumbstick', False))
                if 'thumbstickValue' in self.left_controller_state:
                    thumbstick_val = self.left_controller_state.get('thumbstickValue', [0.0, 0.0])
                    if isinstance(thumbstick_val, (list, tuple)) and len(thumbstick_val) >= 2:
                        left_thumbstick_value = [-float(thumbstick_val[0]), float(thumbstick_val[1])]

            if isinstance(self.right_controller_state, dict):
                if 'thumbstick' in self.right_controller_state:
                    right_thumbstick_pressed = bool(self.right_controller_state.get('thumbstick', False))
                if 'thumbstickValue' in self.right_controller_state:
                    thumbstick_val = self.right_controller_state.get('thumbstickValue', [0.0, 0.0])
                    if isinstance(thumbstick_val, (list, tuple)) and len(thumbstick_val) >= 2:
                        right_thumbstick_value = [float(thumbstick_val[0]), float(thumbstick_val[1])]

            # Check for mode switch: both thumbsticks pressed simultaneously -> toggle between lift+head and lift+cmd_vel
            if left_thumbstick_pressed and right_thumbstick_pressed:
                # Only switch on rising edge (when both are newly pressed)
                if not self.prev_left_thumbstick_pressed or not self.prev_right_thumbstick_pressed:
                    self.joystick_mode = not self.joystick_mode
                    mode_name = "LIFT+HEAD" if self.joystick_mode else "LIFT+CMD_VEL"
                    self.get_logger().info(f'[THUMBSTICK] Mode switched to: {mode_name}')

            self.prev_left_thumbstick_pressed = left_thumbstick_pressed
            self.prev_right_thumbstick_pressed = right_thumbstick_pressed

            # Publish lift based on right thumbstick (works in both modes)
            # Deadzone is applied inside publish_right_joystick
            if abs(right_thumbstick_value[1]) > 0.0:
                self.publish_right_joystick(-right_thumbstick_value[1])

            # Publish head based on left thumbstick (only in lift+head mode)
            # Deadzone is applied inside publish_left_joystick_from_thumbstick
            if self.joystick_mode and (abs(left_thumbstick_value[0]) > 0.0 or abs(left_thumbstick_value[1]) > 0.0):
                self.publish_left_joystick_from_thumbstick(left_thumbstick_value)

            # Publish cmd_vel in lift+cmd_vel mode (based on thumbstick values)
            # Deadzone is applied inside publish_cmd_vel_from_thumbstick
            if not self.joystick_mode:
                self.publish_cmd_vel_from_thumbstick(left_thumbstick_value, right_thumbstick_value)

        except Exception as e:
            self.get_logger().error(f'Error processing thumbstick: {e}')
            import traceback
            traceback.print_exc()

    def publish_right_joystick(self, thumbstick_value):
        """Publish right joystick command to lift_joint (similar to joystick_controller.cpp)."""
        try:
            # Apply deadzone to thumbstick value
            deadzone_applied_value = self.apply_deadzone(float(thumbstick_value))
            
            # Calculate new position: current_position + thumbstick_value * jog_scale
            # Similar to joystick_controller.cpp: new_position = current_position + sensorxel_joy_value * sensor_jog_scale
            new_lift_position = self.lift_joint_current_position + deadzone_applied_value * self.jog_scale
            
            # Log for debugging
            self.get_logger().debug(
                f'[LIFT] current={self.lift_joint_current_position:.3f}, '
                f'thumbstick={thumbstick_value:.3f}, '
                f'deadzone_applied={deadzone_applied_value:.3f}, '
                f'new={new_lift_position:.3f}'
            )

            msg = JointTrajectory()
            msg.header.stamp.sec = 0
            msg.header.stamp.nanosec = 0
            msg.header.frame_id = ''
            msg.joint_names = ['lift_joint']

            point = JointTrajectoryPoint()
            point.positions = [new_lift_position]
            point.velocities = [0.0]
            point.accelerations = [0.0]
            point.effort = []
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = 0

            msg.points.append(point)
            self.right_joystick_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing right joystick: {e}')
            import traceback
            traceback.print_exc()

    def publish_left_joystick_from_thumbstick(self, thumbstick_value):
        """Publish left joystick command to head joints from thumbstick (similar to joystick_controller.cpp)."""
        try:
            # Apply deadzone to thumbstick values
            deadzone_applied_x = self.apply_deadzone(float(thumbstick_value[0]))
            deadzone_applied_y = self.apply_deadzone(float(thumbstick_value[1]))
            
            # Calculate new positions: current_position + thumbstick_value * jog_scale
            # Similar to joystick_controller.cpp
            # thumbstick_value[0] -> head_joint1, thumbstick_value[1] -> head_joint2
            new_head_joint1_position = self.head_joint1_current_position + deadzone_applied_x * self.jog_scale
            new_head_joint2_position = self.head_joint2_current_position + deadzone_applied_y * self.jog_scale

            msg = JointTrajectory()
            msg.joint_names = ['head_joint1', 'head_joint2']
            point = JointTrajectoryPoint()
            point.positions = [new_head_joint1_position, new_head_joint2_position]
            point.velocities = [0.0, 0.0]
            point.accelerations = [0.0, 0.0]
            point.effort = []
            msg.points.append(point)
            self.left_joystick_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing left joystick from thumbstick: {e}')
            import traceback
            traceback.print_exc()

    def publish_cmd_vel_from_thumbstick(self, left_thumbstick_value, right_thumbstick_value):
        """Publish cmd_vel based on thumbstick values (similar to joystick_controller.cpp)."""
        try:
            if not self.vr_publishing_enabled:
                return

            # Apply deadzone to thumbstick values
            left_x_deadzone = self.apply_deadzone(float(left_thumbstick_value[0]))
            left_y_deadzone = self.apply_deadzone(float(left_thumbstick_value[1]))
            right_x_deadzone = self.apply_deadzone(float(right_thumbstick_value[0]))
            right_y_deadzone = self.apply_deadzone(float(right_thumbstick_value[1]))

            # Calculate cmd_vel from thumbstick values (similar to joystick_controller.cpp)
            # linear.x = -left_y / LINEAR_X_SCALE
            # linear.y = left_x / LINEAR_Y_SCALE
            # angular.z = -right_x / ANGULAR_Z_SCALE
            twist_msg = Twist()
            twist_msg.linear.x = -left_y_deadzone / self.linear_x_scale
            twist_msg.linear.y = left_x_deadzone / self.linear_y_scale
            twist_msg.linear.z = 0.0
            twist_msg.angular.x = 0.0
            twist_msg.angular.y = 0.0
            twist_msg.angular.z = -right_x_deadzone / self.angular_z_scale

            self.cmd_vel_pub.publish(twist_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing cmd_vel from thumbstick: {e}')
            import traceback
            traceback.print_exc()

    def publish_controller_pose(self, controller_matrix, side='left'):
        """Transform controller pose from VR coordinates to ROS base_link and publish."""
        try:
            if not self.vr_publishing_enabled:
                self.get_logger().debug('VR publishing is disabled, skipping controller pose publish')
                return

            # Arm control always works (no mode check needed)

            # Check squeeze condition - both squeezes must be held for 2 seconds
            if not self.check_squeeze_condition():
                return

            # Calculate base position using shared function
            base_position, relative_quat_ros = self.calculate_base_position(controller_matrix)

            # Process orientation
            camera_relative_rotation = R.from_quat(relative_quat_ros)

            # Apply rotation around Z-axis (blue axis) based on hand side
            # Right hand: counter-clockwise 90 degrees (positive)
            # Left hand: clockwise 90 degrees (negative)
            if side == 'right':
                rot_z_90 = R.from_euler('z', 90, degrees=True)
                camera_relative_rotation = camera_relative_rotation * rot_z_90
            elif side == 'left':
                rot_z_90 = R.from_euler('z', 90, degrees=True)
                camera_relative_rotation = camera_relative_rotation * rot_z_90

            arm_quaternion = camera_relative_rotation.as_quat()  # [x, y, z, w]

            # Create and publish pose message
            target_pose = PoseStamped()
            target_pose.header.stamp = self.get_clock().now().to_msg()
            target_pose.header.frame_id = 'base_link'

            target_pose.pose.position.x = base_position[0] - 0.15  # Small offset for better visualization
            target_pose.pose.position.y = base_position[1]
            target_pose.pose.position.z = base_position[2]

            target_pose.pose.orientation.x = arm_quaternion[0]
            target_pose.pose.orientation.y = arm_quaternion[1]
            target_pose.pose.orientation.z = arm_quaternion[2]
            target_pose.pose.orientation.w = arm_quaternion[3]

            # Publish to appropriate topic
            if side == 'left':
                self.left_controller_wrist_pub.publish(target_pose)
            elif side == 'right':
                self.right_controller_wrist_pub.publish(target_pose)
                self.get_logger().info(
                    f'Published right controller pose: pos=[{base_position[0]:.3f}, {base_position[1]:.3f}, {base_position[2]:.3f}]'
                )

        except Exception as e:
            self.get_logger().error(f'Error publishing {side} controller pose: {e}')
            import traceback
            traceback.print_exc()

    def start_vuer_server(self):
        """Start the VR server in a separate thread."""
        async def start_server():
            try:
                self.vuer.spawn(start=True)(self.main_hand_tracking)
                self.get_logger().info('Starting VR server...')
                # Check if start() returns something awaitable
                start_result = self.vuer.start()
                if start_result is not None:
                    await start_result
            except Exception as e:
                self.get_logger().error(f'Failed to start VR server: {e}')

        def run_server():
            try:
                self.loop.run_until_complete(start_server())
                self.loop.run_forever()
            except Exception as e:
                self.get_logger().error(f'Error in VR server thread: {e}')

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

    async def main_hand_tracking(self, session):
        """Main hand tracking session."""
        try:
            fps = self.fps
            self.get_logger().info('Starting hand tracking and controller session')

            # Set default scene with VR enabled
            session.set @ DefaultScene(
                frameloop="always",
                handTracking=True,  # Enable hand tracking
                handTrackingOptions={
                    "enable": True,
                    "maxHands": 2,
                    "modelComplexity": 1
                }
            )

            # Enable hand tracking
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

            # Enable motion controllers streaming
            session.upsert @ MotionControllers(
                stream=True,
                key="motion-controller",
                left=True,
                right=True
            )

            self.get_logger().info('Hand tracking and motion controllers enabled')
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

                # Calculate relative position and quaternion for swerve mode
                relative_pos_ros, relative_quat_ros = self.vr_to_ros_transform(pos, quat)

                # Head is published in process_thumbstick() based on thumbstick values
                # Cmd_vel is also published in process_thumbstick() based on thumbstick values

        except Exception as e:
            self.get_logger().error(f'Error in camera move event: {e}')

    async def on_controller_move(self, event, session: VuerSession):
        """Handle controller movement events."""
        try:
            if not self.vr_publishing_enabled:
                return

            if not isinstance(event.value, dict):
                return

            # Process left controller (hand data)
            if 'left' in event.value:
                left_matrix_data = event.value['left']
                if isinstance(left_matrix_data, (list, np.ndarray)) and len(left_matrix_data) == 16:
                    left_matrix = np.array(left_matrix_data).reshape(4, 4, order='F')
                    self.left_controller_data = left_matrix

                    # Calculate base position using shared function
                    base_position, _ = self.calculate_base_position(left_matrix)

                    # Check if controller or hand tracking based on state value
                    # Controller has state values, hand tracking typically has 0 or None
                    is_controller = (self.left_controller_state is not None and
                                    self.left_controller_state != 0 and
                                    self.left_controller_state != [0, 0] and
                                    self.left_controller_state != {})

                    # Publish to topic (no logging for left)
                    self.publish_controller_pose(left_matrix, 'left')

            # Process right controller (hand data)
            if 'right' in event.value:
                right_matrix_data = event.value['right']
                if isinstance(right_matrix_data, (list, np.ndarray)) and len(right_matrix_data) == 16:
                    right_matrix = np.array(right_matrix_data).reshape(4, 4, order='F')
                    self.right_controller_data = right_matrix

                    # Calculate base position using shared function
                    base_position, _ = self.calculate_base_position(right_matrix)

                    # Check if controller or hand tracking based on state value
                    # Controller has state values, hand tracking typically has 0 or None
                    is_controller = (self.right_controller_state is not None and
                                    self.right_controller_state != 0 and
                                    self.right_controller_state != [0, 0] and
                                    self.right_controller_state != {})

                    detection_type = '[CONTROLLER]' if is_controller else '[HAND TRACKING]'
                    has_finger_data = 'False' if is_controller else 'True (estimated)'

                    # Log right data with coordinates and detection type
                    state_info = f', state={self.right_controller_state}' if self.right_controller_state is not None else ', state=None'
                    self.get_logger().info(
                        f'{detection_type} right: pos=[{base_position[0]:.3f}, {base_position[1]:.3f}, {base_position[2]:.3f}]{state_info}, has_finger_data={has_finger_data}'
                    )

                    # Publish to topic
                    self.publish_controller_pose(right_matrix, 'right')

            # Process controller states
            if 'leftState' in event.value:
                self.left_controller_state = event.value['leftState']
                # No logging for left state

            if 'rightState' in event.value:
                self.right_controller_state = event.value['rightState']
                # Extract squeeze value for logging
                right_squeeze = 0.0
                if isinstance(self.right_controller_state, dict) and 'squeezeValue' in self.right_controller_state:
                    right_squeeze = self.right_controller_state.get('squeezeValue', 0.0)
                self.get_logger().info(f'right_controller_state: {self.right_controller_state}, squeeze={right_squeeze:.3f}')

            # Process thumbstick for mode switching and joystick control
            self.process_thumbstick()

            # Debug logging
            self.hand_log_counter += 1
            if self.hand_log_counter % self.log_every_n == 0:
                self.get_logger().info('Controller tracking data processed')

        except Exception as e:
            self.get_logger().error(f'Error in controller move event: {e}')
            import traceback
            traceback.print_exc()

    def __del__(self):
        try:
            if hasattr(self, 'vuer'):
                self.loop.run_until_complete(self.vuer.stop())
            if hasattr(self, 'loop'):
                self.loop.close()
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
