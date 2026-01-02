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

from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, Quaternion
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
        # self.vuer.add_handler('HAND_MOVE')(self.on_hand_move)
        self.vuer.add_handler('CAMERA_MOVE')(self.on_camera_move)
        # self.vuer.add_handler('BODY_TRACKING_MOVE')(self.on_body_move)
        self.vuer.add_handler('CONTROLLER_MOVE')(self.on_controller_move)

        # Controller data storage
        self.left_controller_data = None
        self.right_controller_data = None
        self.left_controller_state = None
        self.right_controller_state = None

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

        # Scaling VR data
        self.scaling_vr = 1.1

        # Async setup
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.start_vuer_server()

        self.get_logger().info('VR Trajectory Publisher node has been started')



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

    def publish_controller_pose(self, controller_matrix, side='left'):
        """Transform controller pose from VR coordinates to ROS base_link and publish."""
        try:
            if not self.vr_publishing_enabled:
                self.get_logger().debug('VR publishing is disabled, skipping controller pose publish')
                return

            # Controller matrix is in VR world coordinates
            # First transform to head/camera relative coordinates (like hand tracking)
            if self.head_inverse_matrix is not None and np.all(np.isfinite(self.head_inverse_matrix)):
                # Transform world matrix to head relative matrix
                relative_matrix = self.head_inverse_matrix @ controller_matrix
                
                # Convert relative matrix to pose
                relative_pos, relative_quat = self.matrix_to_pose(relative_matrix)
                
                # Transform from VR coordinate system to ROS coordinate system
                relative_pos_ros, relative_quat_ros = self.vr_to_ros_transform(relative_pos, relative_quat)
            else:
                # Fallback: use world coordinates directly if head matrix not available
                self.get_logger().warn('Head inverse matrix not available, using world coordinates')
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

            # Process orientation
            camera_relative_rotation = R.from_quat(relative_quat_ros)

            # Additional: Apply 180-degree rotation around Z-axis (right hand only)
            if side == 'right':
                rot_z_180 = R.from_euler('z', 180, degrees=True)
                camera_relative_rotation = camera_relative_rotation * rot_z_180

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
                self.get_logger().info(
                    f'Published left controller pose: pos=[{base_position[0]:.3f}, {base_position[1]:.3f}, {base_position[2]:.3f}]'
                )
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

            # print(f'matrix: {matrix}')

            if isinstance(matrix, (list, np.ndarray)) and len(matrix) == 16:
                self.head_transform_matrix = np.array(matrix).reshape(4, 4, order='F')
                self.head_inverse_matrix = np.linalg.inv(self.head_transform_matrix)
                pos, quat = self.matrix_to_pose(self.head_transform_matrix)

                # Publish head joint trajectory
                r = R.from_quat(quat)
                roll, pitch, yaw = r.as_euler('zxy')

                # print(f'roll: {roll}, pitch: {pitch}, yaw: {yaw}')

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
                    # Store hand data for processing
                    # TODO: Implement process_hand_joints if needed

            # Process hand data
            if 'right' in event.value:
                right_data = event.value['right']
                if isinstance(right_data, (list, np.ndarray)) and len(right_data) == 400:
                    self.right_hand_data = np.array(right_data)
                    # Store hand data for processing
                    # TODO: Implement process_hand_joints if needed

            # Debug logging
            self.hand_log_counter += 1
            if self.hand_log_counter % self.log_every_n == 0:
                self.get_logger().info('Hand tracking data processed')

        except Exception as e:
            self.get_logger().error(f'Error in hand move event: {e}')

    async def on_body_move(self, event, session):
        """
        Handle incoming BODY_TRACKING_MOVE events from the client.
        event.value should be a BodiesData dictionary:
        { jointName: { matrix: Float32Array-like, ... }, ... }
        """
        print(f"BODY_TRACKING_MOVE: key={event.key} ts={getattr(event, 'ts', None)}")
        # print(f'event.value: {event.value}')

        # Example: print only the first joint to avoid large output
        if event.value:
            first_joint, first_data = next(iter(event.value.items()))
            print(
                first_joint,
                "matrix_len=",
                len(first_data.get("matrix", [])) if first_data else None,
            )

    async def on_controller_move(self, event, session: VuerSession):
        """Handle controller movement events."""
        try:
            if not self.vr_publishing_enabled:
                return

            if not isinstance(event.value, dict):
                return

            # Process left controller
            if 'left' in event.value:
                left_matrix_data = event.value['left']
                if isinstance(left_matrix_data, (list, np.ndarray)) and len(left_matrix_data) == 16:
                    # Reshape with column-major order (Fortran order) like vr_publisher_bh5.py
                    left_matrix = np.array(left_matrix_data).reshape(4, 4, order='F')
                    self.left_controller_data = left_matrix

                    # Convert matrix to pose and process (similar to hand tracking)
                    pos, quat = self.matrix_to_pose(left_matrix)

                    # print(f'Left controller: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]')

                    # Transform VR coordinates to ROS and publish
                    self.publish_controller_pose(left_matrix, 'left')

            # Process right controller
            if 'right' in event.value:
                right_matrix_data = event.value['right']
                if isinstance(right_matrix_data, (list, np.ndarray)) and len(right_matrix_data) == 16:
                    # Reshape with column-major order (Fortran order) like vr_publisher_bh5.py
                    right_matrix = np.array(right_matrix_data).reshape(4, 4, order='F')
                    self.right_controller_data = right_matrix

                    # Convert matrix to pose and process (similar to hand tracking)
                    pos, quat = self.matrix_to_pose(right_matrix)
                    # print(f'Right controller: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], quat=[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]')

                    # Transform VR coordinates to ROS and publish
                    self.publish_controller_pose(right_matrix, 'right')

            # Process controller states
            if 'leftState' in event.value:
                self.left_controller_state = event.value['leftState']
                trigger_value = self.left_controller_state.get('triggerValue', 0.0)

                # Optional: Trigger haptic feedback
                if trigger_value > 0.5:
                    session.upsert @ MotionControllers(
                        key="motion-controller",
                        left=True,
                        right=True,
                        pulseLeftStrength=trigger_value,
                        pulseLeftDuration=100,
                        pulseLeftHash=f'{self.get_clock().now().to_msg().sec}_{trigger_value:.2f}',
                    )

            if 'rightState' in event.value:
                self.right_controller_state = event.value['rightState']
                trigger_value = self.right_controller_state.get('triggerValue', 0.0)

                # Optional: Trigger haptic feedback
                if trigger_value > 0.5:
                    session.upsert @ MotionControllers(
                        key="motion-controller",
                        left=True,
                        right=True,
                        pulseRightStrength=trigger_value,
                        pulseRightDuration=100,
                        pulseRightHash=f'{self.get_clock().now().to_msg().sec}_{trigger_value:.2f}',
                    )

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
