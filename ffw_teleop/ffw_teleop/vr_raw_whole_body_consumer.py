#!/usr/bin/env python3

import math

from geometry_msgs.msg import PoseArray, PoseStamped, Twist, Point32
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ffw_interfaces.msg import HandJoints


class VRRawWholeBodyConsumer(Node):
    BODY_HEAD_TO_ROS_POSITION = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64)

    def __init__(self):
        super().__init__('vr_raw_whole_body_consumer')
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)

        self.declare_parameter('enable_lift_publishing', True)
        self.declare_parameter('enable_head_publishing', False)
        self.declare_parameter('enable_base_publishing', True)
        self.declare_parameter('base_linear_kp', 1.7)
        self.declare_parameter('base_angular_kp', 2.0)
        self.declare_parameter('base_linear_deadzone', 0.1)
        self.declare_parameter('base_angular_deadzone', 0.05)
        self.declare_parameter('base_max_linear_velocity', 0.3)
        self.declare_parameter('base_max_angular_velocity', 0.5)
        self.declare_parameter('enable_base_debug_topics', False)
        self.declare_parameter('base_divergence_position_threshold', 0.5)
        self.declare_parameter('base_divergence_yaw_threshold', 0.5)
        self.declare_parameter('wrist_offset_x', 0.0)
        self.declare_parameter('wrist_offset_y', 0.0)
        self.declare_parameter('wrist_offset_z', 0.0)
        self.declare_parameter('elbow_offset_x', 0.0)
        self.declare_parameter('elbow_offset_y', 0.0)
        self.declare_parameter('elbow_offset_z', 0.0)
        self.declare_parameter('hand_pose_is_head_relative', True)
        self.declare_parameter('zero_z_on_start', False)
        self.declare_parameter('apply_head_height_to_arm_z', True)
        self.declare_parameter('lift_low_pass_alpha', 0.3)
        self.declare_parameter('max_lift_velocity', 0.07)
        self.declare_parameter('raw_sync_slop_sec', 0.05)
        self.declare_parameter('low_pass_filter_alpha', 0.9)
        self.declare_parameter('wrist_vr_scale', 1.4)
        self.declare_parameter('elbow_vr_scale', 1.4)
        self.declare_parameter('pitch_offset', -0.5)

        self.enable_lift_publishing = self.get_parameter('enable_lift_publishing').value
        self.enable_head_publishing = self.get_parameter('enable_head_publishing').value
        self.enable_base_publishing = self.get_parameter('enable_base_publishing').value
        self.base_linear_kp = float(self.get_parameter('base_linear_kp').value)
        self.base_angular_kp = float(self.get_parameter('base_angular_kp').value)
        self.base_linear_deadzone = float(self.get_parameter('base_linear_deadzone').value)
        self.base_angular_deadzone = float(self.get_parameter('base_angular_deadzone').value)
        self.base_max_linear_velocity = float(self.get_parameter('base_max_linear_velocity').value)
        self.base_max_angular_velocity = float(self.get_parameter('base_max_angular_velocity').value)
        self.enable_base_debug_topics = self.get_parameter('enable_base_debug_topics').value
        self.base_divergence_position_threshold = float(self.get_parameter('base_divergence_position_threshold').value)
        self.base_divergence_yaw_threshold = float(self.get_parameter('base_divergence_yaw_threshold').value)
        self.hand_pose_is_head_relative = self.get_parameter('hand_pose_is_head_relative').value
        self.zero_z_on_start = self.get_parameter('zero_z_on_start').value
        self.apply_head_height_to_arm_z = self.get_parameter('apply_head_height_to_arm_z').value
        self.lift_low_pass_alpha = float(self.get_parameter('lift_low_pass_alpha').value)
        self.max_lift_velocity = float(self.get_parameter('max_lift_velocity').value)
        self.raw_sync_slop_ns = int(float(self.get_parameter('raw_sync_slop_sec').value) * 1e9)
        self.low_pass_filter_alpha = float(self.get_parameter('low_pass_filter_alpha').value)
        self.wrist_vr_scale = float(self.get_parameter('wrist_vr_scale').value)
        self.elbow_vr_scale = float(self.get_parameter('elbow_vr_scale').value)
        self.pitch_offset = float(self.get_parameter('pitch_offset').value)
        self.wrist_offsets = {
            'x': float(self.get_parameter('wrist_offset_x').value),
            'y': float(self.get_parameter('wrist_offset_y').value),
            'z': float(self.get_parameter('wrist_offset_z').value),
        }
        self.elbow_offsets = {
            'x': float(self.get_parameter('elbow_offset_x').value),
            'y': float(self.get_parameter('elbow_offset_y').value),
            'z': float(self.get_parameter('elbow_offset_z').value),
        }

        if self.base_linear_kp <= 0.0:
            self.base_linear_kp = 2.0
        if self.base_angular_kp <= 0.0:
            self.base_angular_kp = 1.0
        if self.base_linear_deadzone < 0.0:
            self.base_linear_deadzone = 0.05
        if self.base_angular_deadzone < 0.0:
            self.base_angular_deadzone = 0.05

        self.vr_publishing_enabled = True
        self.vr_hand_to_urdf = np.array([
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=np.float64)
        self.body_head_to_ros_rot = R.from_matrix(self.BODY_HEAD_TO_ROS_POSITION)
        self.ros_to_body_head_position = self.BODY_HEAD_TO_ROS_POSITION.T
        self.zedm_to_base_offset = np.array([
            0.0 - 0.0238122 - 0.040 - 0.049483 - 0.0055,
            0.0,
            -0.01325 + 0.0242094 - 0.054 - 0.102130 - 1.4316,
        ], dtype=np.float64)

        self.prev_poses_left = np.zeros((21, 3), dtype=np.float64)
        self.prev_poses_right = np.zeros((21, 3), dtype=np.float64)
        self.start_poses_left = False
        self.start_poses_right = False
        self.pose_filters = {}
        self.max_elbow_wrist_distance = 0.4
        self.max_wrist_angle_step_deg = 30.0

        self.current_odom = None
        self.initial_odom_position = None
        self.initial_odom_yaw = None
        self.initial_camera_height = None
        self.initial_camera_position = None
        self.initial_camera_yaw = None
        self.head_height_offset_for_arms = 0.0
        self.filtered_lift_position = None
        self.last_lift_time = None
        self.z_calibrated = False
        self.z_calibration_offset = 0.0
        self.head_ros_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.cmd_vel_log_counter = 0
        self.cmd_vel_log_every_n = 10
        self.base_divergence_log_counter = 0
        self.base_divergence_log_every_n = 20
        self.hand_log_counter = 0
        self.wrist_debug_log_counter = 0
        self.wrist_debug_log_every_n = 30

        self.latest_left_wrist = None
        self.latest_right_wrist = None
        self.latest_left_hand = None
        self.latest_right_hand = None

        self.left_hand_pos_pub = self.create_publisher(HandJoints, '/left_hand/hand_joint_pos', 10)
        self.right_hand_pos_pub = self.create_publisher(HandJoints, '/right_hand/hand_joint_pos', 10)
        self.head_joint_pub = self.create_publisher(JointTrajectory, '/leader/joystick_controller_left/joint_trajectory', 10)
        self.lift_joint_pub = self.create_publisher(JointTrajectory, '/leader/joystick_controller_right/joint_trajectory', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.base_divergence_pub = self.create_publisher(Bool, '/vr_base_divergence', 10)
        self.vr_camera_goal_pub = self.create_publisher(PoseStamped, '/vr_camera_goal_pose', 10)
        self.left_wrist_rviz_pub = self.create_publisher(PoseStamped, '/l_goal_pose', 10)
        self.right_wrist_rviz_pub = self.create_publisher(PoseStamped, '/r_goal_pose', 10)
        self.left_elbow_pub = self.create_publisher(PoseStamped, '/l_elbow_pose', 10)
        self.right_elbow_pub = self.create_publisher(PoseStamped, '/r_elbow_pose', 10)

        self.create_subscription(Bool, '/vr_control/toggle', self.vr_control_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/vr/raw/head_pose', self.head_pose_callback, 10)
        self.create_subscription(PoseStamped, '/vr/raw/left_wrist_pose', self.left_wrist_callback, 10)
        self.create_subscription(PoseStamped, '/vr/raw/right_wrist_pose', self.right_wrist_callback, 10)
        self.create_subscription(PoseStamped, '/vr/raw/left_elbow_pose', self.left_elbow_callback, 10)
        self.create_subscription(PoseStamped, '/vr/raw/right_elbow_pose', self.right_elbow_callback, 10)
        self.create_subscription(PoseArray, '/vr/raw/left_hand_points', self.left_hand_callback, 10)
        self.create_subscription(PoseArray, '/vr/raw/right_hand_points', self.right_hand_callback, 10)

        self.status_timer = self.create_timer(5.0, self.log_status)

        self.get_logger().info('VR raw whole-body consumer started')
        self.get_logger().info(
            'Subscribers: /vr/raw/head_pose, /vr/raw/*_wrist_pose, /vr/raw/*_elbow_pose, /vr/raw/*_hand_points'
        )

    def odom_callback(self, msg):
        self.current_odom = msg

    def vr_control_callback(self, msg):
        new_state = bool(msg.data)
        if new_state == self.vr_publishing_enabled:
            return

        self.vr_publishing_enabled = new_state
        status = 'ENABLED' if self.vr_publishing_enabled else 'DISABLED'
        self.get_logger().info(f'VR raw consumer changed to: {status}')

        self.start_poses_left = False
        self.start_poses_right = False
        self.prev_poses_left.fill(0.0)
        self.prev_poses_right.fill(0.0)
        self.pose_filters.clear()
        self.filtered_lift_position = None
        self.last_lift_time = None
        self.z_calibrated = False
        self.head_height_offset_for_arms = 0.0
        self.latest_left_wrist = None
        self.latest_right_wrist = None
        self.latest_left_hand = None
        self.latest_right_hand = None

        if self.vr_publishing_enabled:
            self.initial_camera_height = None
            self.initial_camera_position = None
            self.initial_camera_yaw = None
            if self.current_odom is not None:
                pos = self.current_odom.pose.pose.position
                quat = self.current_odom.pose.pose.orientation
                yaw = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler('xyz')[2]
                self.initial_odom_position = np.array([pos.x, pos.y], dtype=np.float64)
                self.initial_odom_yaw = float(yaw)
            else:
                self.initial_odom_position = None
                self.initial_odom_yaw = None
        else:
            self.initial_odom_position = None
            self.initial_odom_yaw = None
            self.initial_camera_height = None
            self.initial_camera_position = None
            self.initial_camera_yaw = None

    def log_status(self):
        status = 'ENABLED' if self.vr_publishing_enabled else 'DISABLED'
        self.get_logger().info(f'Status: VR={status}')

    def head_pose_callback(self, msg):
        if not self.vr_publishing_enabled:
            return

        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=np.float64)
        quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ], dtype=np.float64)
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
            return

        self.head_ros_quat = quat
        current_camera_height = float(pos[2])
        current_camera_position = np.array([pos[0], pos[1]], dtype=np.float64)
        fallback_yaw = float(self.initial_camera_yaw) if self.initial_camera_yaw is not None else 0.0
        current_camera_yaw = self.yaw_from_orientation_horizontal(quat, fallback_yaw)
        head_rot = R.from_quat(quat)

        if self.initial_camera_height is None:
            self.initial_camera_height = current_camera_height
            self.initial_camera_position = current_camera_position.copy()
            self.initial_camera_yaw = current_camera_yaw
        if self.initial_odom_position is None and self.current_odom is not None:
            odom_pos = self.current_odom.pose.pose.position
            odom_quat = self.current_odom.pose.pose.orientation
            odom_yaw = R.from_quat([odom_quat.x, odom_quat.y, odom_quat.z, odom_quat.w]).as_euler('xyz')[2]
            self.initial_odom_position = np.array([odom_pos.x, odom_pos.y], dtype=np.float64)
            self.initial_odom_yaw = float(odom_yaw)

        relative_height = current_camera_height - self.initial_camera_height
        relative_position = current_camera_position - self.initial_camera_position
        relative_yaw = self.wrap_pi(current_camera_yaw - self.initial_camera_yaw)
        self.head_height_offset_for_arms = float(relative_height)

        if self.enable_lift_publishing:
            self.publish_lift(relative_height)
        if self.enable_base_publishing:
            self.publish_base(relative_position, relative_yaw)
        if self.enable_head_publishing:
            self.publish_head_joints(head_rot)

    def left_wrist_callback(self, msg):
        self.latest_left_wrist = msg
        self.try_process_hand('left')

    def right_wrist_callback(self, msg):
        self.latest_right_wrist = msg
        self.try_process_hand('right')

    def left_hand_callback(self, msg):
        self.latest_left_hand = msg
        self.try_process_hand('left')

    def right_hand_callback(self, msg):
        self.latest_right_hand = msg
        self.try_process_hand('right')

    def left_elbow_callback(self, msg):
        self.process_elbow_pose(msg, 'left')

    def right_elbow_callback(self, msg):
        self.process_elbow_pose(msg, 'right')

    def try_process_hand(self, side):
        if not self.vr_publishing_enabled:
            return

        wrist_msg = self.latest_left_wrist if side == 'left' else self.latest_right_wrist
        hand_msg = self.latest_left_hand if side == 'left' else self.latest_right_hand
        if wrist_msg is None or hand_msg is None:
            return
        if not self.stamps_are_close(wrist_msg.header.stamp, hand_msg.header.stamp):
            return

        self.process_hand_messages(wrist_msg, hand_msg, side)

    def process_elbow_pose(self, msg, side):
        if not self.vr_publishing_enabled:
            return
        position, quaternion = self.pose_to_numpy(msg)
        if msg.header.frame_id != 'vr_head' and self.hand_pose_is_head_relative:
            return

        publisher = self.left_elbow_pub if side == 'left' else self.right_elbow_pub
        self.publish_relative_pose(
            position,
            quaternion,
            publisher,
            vr_scale=self.elbow_vr_scale,
            x_offset=self.elbow_offsets['x'],
            y_offset=self.elbow_offsets['y'],
            z_offset=self.elbow_offsets['z'],
            apply_right_z_flip=False,
            pose_role='elbow',
            side=side,
        )

    def process_hand_messages(self, wrist_msg, hand_msg, side):
        if len(hand_msg.poses) != 21:
            return
        if hand_msg.header.frame_id != 'vr_head' and self.hand_pose_is_head_relative:
            return

        wrist_pos_ros, wrist_quat_ros = self.pose_to_numpy(wrist_msg)
        positions_ros = np.array([
            [pose.position.x, pose.position.y, pose.position.z]
            for pose in hand_msg.poses
        ], dtype=np.float64)
        if not (np.all(np.isfinite(positions_ros)) and np.all(np.isfinite(wrist_pos_ros)) and np.all(np.isfinite(wrist_quat_ros))):
            return

        temp_joints = (self.ros_to_body_head_position @ positions_ros.T).T
        wrist_rot_head = (self.body_head_to_ros_rot.inv() * R.from_quat(wrist_quat_ros)).as_matrix()

        if side == 'left':
            if self.start_poses_left:
                temp_joints = (
                    self.low_pass_filter_alpha * temp_joints
                    + (1.0 - self.low_pass_filter_alpha) * self.prev_poses_left
                )
            self.prev_poses_left[:] = temp_joints
            self.start_poses_left = True
            wrist_publisher = self.left_wrist_rviz_pub
            hand_publisher = self.left_hand_pos_pub
        else:
            if self.start_poses_right:
                temp_joints = (
                    self.low_pass_filter_alpha * temp_joints
                    + (1.0 - self.low_pass_filter_alpha) * self.prev_poses_right
                )
            self.prev_poses_right[:] = temp_joints
            self.start_poses_right = True
            wrist_publisher = self.right_wrist_rviz_pub
            hand_publisher = self.right_hand_pos_pub

        rel_points = temp_joints - temp_joints[0]
        retarget_points = (self.vr_hand_to_urdf @ (wrist_rot_head.T @ rel_points.T)).T

        hand_joints = HandJoints()
        hand_joints.header.stamp = hand_msg.header.stamp
        hand_joints.header.frame_id = ''
        hand_joints.joints = []
        for point in retarget_points:
            msg_point = Point32()
            msg_point.x = float(point[0])
            msg_point.y = float(point[1])
            msg_point.z = float(point[2])
            hand_joints.joints.append(msg_point)

        self.publish_relative_pose(
            wrist_pos_ros,
            wrist_quat_ros,
            wrist_publisher,
            vr_scale=self.wrist_vr_scale,
            x_offset=self.wrist_offsets['x'],
            y_offset=self.wrist_offsets['y'],
            z_offset=self.wrist_offsets['z'],
            apply_right_z_flip=(side == 'right'),
            pose_role='wrist',
            side=side,
        )
        hand_publisher.publish(hand_joints)

    def publish_lift(self, relative_height):
        now = self.get_clock().now()
        if self.filtered_lift_position is None:
            self.filtered_lift_position = float(relative_height)
            self.last_lift_time = now
        else:
            target = (
                self.lift_low_pass_alpha * relative_height
                + (1.0 - self.lift_low_pass_alpha) * self.filtered_lift_position
            )
            if self.max_lift_velocity > 0.0 and self.last_lift_time is not None:
                dt = (now.nanoseconds - self.last_lift_time.nanoseconds) / 1e9
                dt = max(0.005, min(0.2, dt))
                max_step = self.max_lift_velocity * dt
                delta = target - self.filtered_lift_position
                if abs(delta) > max_step:
                    self.filtered_lift_position += math.copysign(max_step, delta)
                else:
                    self.filtered_lift_position = target
            else:
                self.filtered_lift_position = target
            self.last_lift_time = now

        lift_msg = JointTrajectory()
        lift_msg.header.stamp.sec = 0
        lift_msg.header.stamp.nanosec = 0
        lift_msg.header.frame_id = ''
        lift_msg.joint_names = ['lift_joint']
        point = JointTrajectoryPoint()
        point.positions = [float(self.filtered_lift_position)]
        point.velocities = [0.0]
        point.accelerations = [0.0]
        point.effort = []
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 0
        lift_msg.points.append(point)
        self.lift_joint_pub.publish(lift_msg)

    def publish_base(self, relative_position, relative_yaw):
        self.cmd_vel_log_counter += 1
        if (
            self.initial_odom_position is None
            or self.initial_odom_yaw is None
            or self.current_odom is None
        ):
            return

        current_odom_pos = self.current_odom.pose.pose.position
        current_odom_quat = self.current_odom.pose.pose.orientation
        current_odom_yaw = R.from_quat([
            current_odom_quat.x,
            current_odom_quat.y,
            current_odom_quat.z,
            current_odom_quat.w,
        ]).as_euler('xyz')[2]
        current_odom_position = np.array([current_odom_pos.x, current_odom_pos.y], dtype=np.float64)
        robot_movement_position = current_odom_position - self.initial_odom_position
        robot_movement_yaw = self.wrap_pi(current_odom_yaw - self.initial_odom_yaw)

        position_error = relative_position - robot_movement_position
        cos_yaw = math.cos(current_odom_yaw)
        sin_yaw = math.sin(current_odom_yaw)
        position_error_base = np.array([
            cos_yaw * position_error[0] + sin_yaw * position_error[1],
            -sin_yaw * position_error[0] + cos_yaw * position_error[1],
        ], dtype=np.float64)
        yaw_error = self.wrap_pi(relative_yaw - robot_movement_yaw)

        linear_x = self.velocity_from_error(
            position_error_base[0], self.base_linear_kp,
            self.base_linear_deadzone, self.base_max_linear_velocity,
        )
        linear_y = self.velocity_from_error(
            position_error_base[1], self.base_linear_kp,
            self.base_linear_deadzone, self.base_max_linear_velocity,
        )
        angular_z = self.velocity_from_error(
            yaw_error, self.base_angular_kp,
            self.base_angular_deadzone, self.base_max_angular_velocity,
        )

        twist_msg = Twist()
        twist_msg.linear.x = linear_x
        twist_msg.linear.y = linear_y
        twist_msg.angular.z = angular_z
        self.cmd_vel_pub.publish(twist_msg)

        position_error_norm = float(np.linalg.norm(position_error))
        is_diverged = (
            position_error_norm > self.base_divergence_position_threshold
            or abs(yaw_error) > self.base_divergence_yaw_threshold
        )
        if self.enable_base_debug_topics:
            self.base_divergence_pub.publish(Bool(data=is_diverged))
            vr_goal_position = self.initial_odom_position + relative_position
            vr_goal_yaw = self.initial_odom_yaw + relative_yaw
            q_goal = R.from_euler('xyz', [0.0, 0.0, vr_goal_yaw]).as_quat()
            goal_msg = PoseStamped()
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.header.frame_id = 'odom'
            goal_msg.pose.position.x = float(vr_goal_position[0])
            goal_msg.pose.position.y = float(vr_goal_position[1])
            goal_msg.pose.orientation.x = float(q_goal[0])
            goal_msg.pose.orientation.y = float(q_goal[1])
            goal_msg.pose.orientation.z = float(q_goal[2])
            goal_msg.pose.orientation.w = float(q_goal[3])
            self.vr_camera_goal_pub.publish(goal_msg)

        if is_diverged:
            self.base_divergence_log_counter += 1
            if self.base_divergence_log_counter % self.base_divergence_log_every_n == 0:
                self.get_logger().warn(
                    f'[BODY] Base divergence: pos_err_norm={position_error_norm:.3f}, yaw_err={yaw_error:+.3f}'
                )

        if self.cmd_vel_log_counter % self.cmd_vel_log_every_n == 0:
            self.get_logger().info(
                f'[BODY] cmd_vel: linear=[{twist_msg.linear.x:+.3f}, {twist_msg.linear.y:+.3f}], '
                f'angular.z={twist_msg.angular.z:+.3f}'
            )

    def publish_head_joints(self, head_rot):
        ros_roll, ros_pitch, ros_yaw = head_rot.as_euler('xyz')
        if not (self.is_valid_float(ros_pitch) and self.is_valid_float(ros_yaw)):
            return
        msg = JointTrajectory()
        msg.joint_names = ['head_joint1', 'head_joint2']
        point = JointTrajectoryPoint()
        point.positions = [float(ros_pitch + self.pitch_offset), float(ros_yaw)]
        point.velocities = [0.0, 0.0]
        point.accelerations = [0.0, 0.0]
        point.effort = []
        msg.points.append(point)
        self.head_joint_pub.publish(msg)

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
        scaled_pos = np.asarray(camera_relative_position, dtype=np.float64) * vr_scale
        base_position = scaled_pos - self.zedm_to_base_offset

        if self.apply_head_height_to_arm_z and pose_role in ('wrist', 'elbow'):
            base_position = base_position.copy()
            base_position[2] += float(self.head_height_offset_for_arms)

        if self.zero_z_on_start:
            if (not self.z_calibrated) and pose_role == 'wrist':
                self.z_calibration_offset = base_position[2]
                self.z_calibrated = True
            if self.z_calibrated:
                base_position = base_position.copy()
                base_position[2] -= self.z_calibration_offset

        camera_relative_rotation = R.from_quat(camera_relative_quaternion)
        if apply_right_z_flip:
            camera_relative_rotation = camera_relative_rotation * R.from_euler('z', 180.0, degrees=True)
        arm_quaternion = camera_relative_rotation.as_quat()

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
        target_pose.pose.position.x = float(base_position[0] + x_offset)
        target_pose.pose.position.y = float(base_position[1] + y_offset)
        target_pose.pose.position.z = float(base_position[2] + z_offset)
        target_pose.pose.orientation.x = float(arm_quaternion[0])
        target_pose.pose.orientation.y = float(arm_quaternion[1])
        target_pose.pose.orientation.z = float(arm_quaternion[2])
        target_pose.pose.orientation.w = float(arm_quaternion[3])
        publisher.publish(target_pose)

    def low_pass_filter_pose(self, key, position, quaternion, max_angle_deg=None):
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
        prev_quat = np.array(prev_quat, dtype=np.float64)
        curr_quat = np.array(current_quat, dtype=np.float64)
        if np.dot(prev_quat, curr_quat) < 0.0:
            curr_quat = -curr_quat
        dot = float(np.clip(np.dot(prev_quat, curr_quat), -1.0, 1.0))
        angle = 2.0 * math.acos(dot)
        max_angle = math.radians(max_angle_deg)
        if angle <= max_angle or angle <= 1.0e-6:
            return curr_quat
        return self.slerp_quaternion(prev_quat, curr_quat, max_angle / angle)

    def slerp_quaternion(self, q0, q1, t):
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
            return other_pos + delta * (self.max_elbow_wrist_distance / dist)
        return position

    def stamps_are_close(self, stamp_a, stamp_b):
        return abs(self.stamp_to_ns(stamp_a) - self.stamp_to_ns(stamp_b)) <= self.raw_sync_slop_ns

    def stamp_to_ns(self, stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def pose_to_numpy(self, msg):
        position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=np.float64)
        quaternion = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ], dtype=np.float64)
        return position, quaternion

    def velocity_from_error(self, err, kp, deadzone, max_vel):
        if abs(err) <= deadzone:
            return 0.0
        vel = kp * err
        return max(-max_vel, min(max_vel, vel))

    def yaw_from_orientation_horizontal(self, ros_quat, fallback_yaw=0.0):
        forward = R.from_quat(ros_quat).apply(np.array([0.0, 0.0, -1.0], dtype=np.float64))
        fx = float(forward[0])
        fy = float(forward[1])
        norm_xy = math.sqrt(fx * fx + fy * fy)
        if norm_xy < 1e-6:
            return fallback_yaw
        return math.atan2(fy, fx)

    def is_valid_float(self, value):
        return isinstance(value, (int, float)) and np.isfinite(value)

    def wrap_pi(self, angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi


def main(args=None):
    rclpy.init(args=args)
    node = VRRawWholeBodyConsumer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
