#!/usr/bin/env python3

from geometry_msgs.msg import Point, PoseStamped, Quaternion, Twist
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Bool, Float32
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


VR_HEAD_TO_ROS = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
], dtype=np.float64)

VR_WORLD_TO_ROS_WORLD = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)

# Convert relative transforms computed in ros_world into the head-relative ROS axes
# used by the original vr_publisher_sg2.py pipeline.
RELATIVE_ROS_WORLD_TO_SG2_ROS = VR_HEAD_TO_ROS @ VR_WORLD_TO_ROS_WORLD.T


class VRRawSG2Consumer(Node):
    def __init__(self):
        super().__init__('vr_raw_sg2_consumer')
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)

        self.declare_parameter('left_wrist_offset_x', 0.0)
        self.declare_parameter('left_wrist_offset_y', 0.0)
        self.declare_parameter('left_wrist_offset_z', 0.0)
        self.declare_parameter('right_wrist_offset_x', 0.0)
        self.declare_parameter('right_wrist_offset_y', 0.0)
        self.declare_parameter('right_wrist_offset_z', 0.0)
        self.declare_parameter('left_wrist_roll_offset_deg', 90.0)
        self.declare_parameter('left_wrist_pitch_offset_deg', 0.0)
        self.declare_parameter('left_wrist_yaw_offset_deg', 0.0)
        self.declare_parameter('right_wrist_roll_offset_deg', 90.0)
        self.declare_parameter('right_wrist_pitch_offset_deg', 0.0)
        self.declare_parameter('right_wrist_yaw_offset_deg', 0.0)
        self.declare_parameter('left_trigger_offset', 0.0)
        self.declare_parameter('right_trigger_offset', 0.0)
        self.declare_parameter('left_trigger_scale', 1.0)
        self.declare_parameter('right_trigger_scale', 1.0)
        self.declare_parameter('goal_pose_position_scale', 1.1)
        self.declare_parameter('pose_publish_hz', 30.0)
        self.declare_parameter('apply_lift_to_arm_z', True)
        self.declare_parameter('lift_to_arm_z_scale', 1.0)
        self.declare_parameter('left_elbow_offset_x', 0.0)
        self.declare_parameter('left_elbow_offset_y', 0.0)
        self.declare_parameter('left_elbow_offset_z', 0.0)
        self.declare_parameter('right_elbow_offset_x', 0.0)
        self.declare_parameter('right_elbow_offset_y', 0.0)
        self.declare_parameter('right_elbow_offset_z', 0.0)
        self.declare_parameter('reactivate_service', '/reactivate')

        self.vr_publishing_enabled = True
        self.head_reference_valid = False
        self.head_transform_matrix = np.eye(4, dtype=np.float64)
        self.head_inverse_matrix = np.eye(4, dtype=np.float64)
        self.relative_ros_world_to_sg2_ros_rot = R.from_matrix(RELATIVE_ROS_WORLD_TO_SG2_ROS)

        self.vr_stream_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.head_joint_pub = self.create_publisher(JointTrajectory, '/leader/joystick_controller_left/joint_trajectory', 10)
        self.lift_joint_pub = self.create_publisher(JointTrajectory, '/leader/joystick_controller_right/joint_trajectory', 10)
        self.left_squeeze_pub = self.create_publisher(Float32, '/vr_controller/left_squeeze', 10)
        self.right_squeeze_pub = self.create_publisher(Float32, '/vr_controller/right_squeeze', 10)
        self.left_trigger_pub = self.create_publisher(Float32, '/vr_controller/left_trigger', 10)
        self.right_trigger_pub = self.create_publisher(Float32, '/vr_controller/right_trigger', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.left_wrist_rviz_pub = self.create_publisher(PoseStamped, '/l_goal_pose', 1)
        self.right_wrist_rviz_pub = self.create_publisher(PoseStamped, '/r_goal_pose', 1)
        self.left_elbow_rviz_pub = self.create_publisher(PoseStamped, '/l_elbow_pose', 1)
        self.right_elbow_rviz_pub = self.create_publisher(PoseStamped, '/r_elbow_pose', 1)

        self.reactivate_service = str(self.get_parameter('reactivate_service').value)
        self.reactivate_client = self.create_client(Trigger, self.reactivate_service)
        self.reactivate_call_in_flight = False
        self.last_reactivate_service_warn_sec = 0.0
        self.both_a_buttons_pressed_prev = False

        self.create_subscription(Bool, '/vr_control/toggle', self.vr_control_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)
        self.create_subscription(PoseStamped, '/vr/raw/controller/head_pose_ros', self.head_pose_callback, self.vr_stream_qos)
        self.create_subscription(Joy, '/vr/raw/controller/left_state', self.left_state_callback, self.vr_stream_qos)
        self.create_subscription(Joy, '/vr/raw/controller/right_state', self.right_state_callback, self.vr_stream_qos)
        self.create_subscription(PoseStamped, '/vr/raw/controller/left_pose_ros', self.left_pose_callback, self.vr_stream_qos)
        self.create_subscription(PoseStamped, '/vr/raw/controller/right_pose_ros', self.right_pose_callback, self.vr_stream_qos)
        self.create_subscription(PoseStamped, '/vr/raw/controller/left_elbow_pose_ros', self.left_elbow_callback, self.vr_stream_qos)
        self.create_subscription(PoseStamped, '/vr/raw/controller/right_elbow_pose_ros', self.right_elbow_callback, self.vr_stream_qos)

        self.left_controller_state = self.default_controller_state()
        self.right_controller_state = self.default_controller_state()
        self.left_squeeze_value = 0.0
        self.right_squeeze_value = 0.0
        self.goal_pose_squeeze_threshold = 0.8

        self.camera_to_base_offset = np.array([
            0.0 - 0.0238122 - 0.040 - 0.049483 - 0.0055,
            0.0,
            -0.01325 + 0.0242094 - 0.054 - 0.102130 - 1.4316,
        ], dtype=np.float64)
        self.wrist_position_offsets = {
            'left': np.array([
                self.get_parameter('left_wrist_offset_x').value,
                self.get_parameter('left_wrist_offset_y').value,
                self.get_parameter('left_wrist_offset_z').value,
            ], dtype=np.float64),
            'right': np.array([
                self.get_parameter('right_wrist_offset_x').value,
                self.get_parameter('right_wrist_offset_y').value,
                self.get_parameter('right_wrist_offset_z').value,
            ], dtype=np.float64),
        }
        self.elbow_position_offsets = {
            'left': np.array([
                self.get_parameter('left_elbow_offset_x').value,
                self.get_parameter('left_elbow_offset_y').value,
                self.get_parameter('left_elbow_offset_z').value,
            ], dtype=np.float64),
            'right': np.array([
                self.get_parameter('right_elbow_offset_x').value,
                self.get_parameter('right_elbow_offset_y').value,
                self.get_parameter('right_elbow_offset_z').value,
            ], dtype=np.float64),
        }
        self.wrist_rotation_offsets = {
            'left': R.from_euler('xyz', [
                self.get_parameter('left_wrist_roll_offset_deg').value,
                self.get_parameter('left_wrist_pitch_offset_deg').value,
                self.get_parameter('left_wrist_yaw_offset_deg').value,
            ], degrees=True),
            'right': R.from_euler('xyz', [
                self.get_parameter('right_wrist_roll_offset_deg').value,
                self.get_parameter('right_wrist_pitch_offset_deg').value,
                self.get_parameter('right_wrist_yaw_offset_deg').value,
            ], degrees=True),
        }
        self.trigger_offsets = {
            'left': float(self.get_parameter('left_trigger_offset').value),
            'right': float(self.get_parameter('right_trigger_offset').value),
        }
        self.trigger_scales = {
            'left': float(self.get_parameter('left_trigger_scale').value),
            'right': float(self.get_parameter('right_trigger_scale').value),
        }
        self.goal_pose_position_scale = float(self.get_parameter('goal_pose_position_scale').value)
        if not np.isfinite(self.goal_pose_position_scale) or self.goal_pose_position_scale <= 0.0:
            self.goal_pose_position_scale = 1.0

        self.pose_publish_hz = float(self.get_parameter('pose_publish_hz').value)
        self.pose_min_period = (1.0 / self.pose_publish_hz) if self.pose_publish_hz > 0.0 else 0.0
        self.last_pose_publish_sec = {
            'left_wrist': 0.0,
            'right_wrist': 0.0,
            'left_elbow': 0.0,
            'right_elbow': 0.0,
        }

        self.joystick_mode = True
        self.prev_left_thumbstick_pressed = False
        self.prev_right_thumbstick_pressed = False
        self.linear_x_scale = 3.0
        self.linear_y_scale = 3.0
        self.angular_z_scale = 2.0
        self.left_jog_scale = 0.06
        self.right_jog_scale = 0.01
        self.deadzone = 0.05
        self.left_reverse_x = False
        self.left_reverse_y = True
        self.left_stick_swap_xy = True
        self.right_stick_swap_xy = True
        self.current_joint_states = None
        self.lift_joint_current_position = 0.0
        self.lift_reference_position_for_pose = None
        self.head_joint1_current_position = 0.0
        self.head_joint2_current_position = 0.0
        self.apply_lift_to_arm_z = bool(self.get_parameter('apply_lift_to_arm_z').value)
        self.lift_to_arm_z_scale = float(self.get_parameter('lift_to_arm_z_scale').value)
        self.control_max_hz = 30.0
        self.control_min_period = 1.0 / self.control_max_hz
        self.last_lift_publish_sec = 0.0
        self.last_head_publish_sec = 0.0
        self.last_cmd_vel_publish_sec = 0.0
        self.last_lift_command = None
        self.last_head_command = None
        self.last_cmd_vel_command = (0.0, 0.0, 0.0)

        self.get_logger().info('VR raw SG2 consumer started')

    def default_controller_state(self):
        return {
            'thumbstick': False,
            'thumbstickValue': [0.0, 0.0],
            'triggerValue': 0.0,
            'squeezeValue': 0.0,
            'aButton': False,
            'bButton': False,
            'xButton': False,
            'yButton': False,
        }

    def vr_control_callback(self, msg):
        self.vr_publishing_enabled = bool(msg.data)
        if not self.vr_publishing_enabled:
            self.head_reference_valid = False
        status = 'ENABLED' if self.vr_publishing_enabled else 'DISABLED'
        self.get_logger().info(f'VR raw SG2 consumer changed to: {status}')

    def joint_states_callback(self, msg):
        self.current_joint_states = msg
        if 'lift_joint' in msg.name:
            idx = msg.name.index('lift_joint')
            self.lift_joint_current_position = msg.position[idx]
            if self.lift_reference_position_for_pose is None:
                self.lift_reference_position_for_pose = self.lift_joint_current_position
        if 'head_joint1' in msg.name:
            idx = msg.name.index('head_joint1')
            self.head_joint1_current_position = msg.position[idx]
        if 'head_joint2' in msg.name:
            idx = msg.name.index('head_joint2')
            self.head_joint2_current_position = msg.position[idx]

    def head_pose_callback(self, msg):
        if msg.header.frame_id != 'ros_world':
            return
        head_matrix = self.pose_to_matrix(msg.pose)
        if not np.all(np.isfinite(head_matrix)):
            return
        if abs(float(np.linalg.det(head_matrix[:3, :3]))) < 1e-6:
            return
        self.head_transform_matrix = head_matrix
        try:
            self.head_inverse_matrix = np.linalg.inv(head_matrix)
        except np.linalg.LinAlgError:
            return
        self.head_reference_valid = True

    def left_state_callback(self, msg):
        self.left_controller_state = self.joy_to_state(msg)
        self.left_squeeze_value = float(self.left_controller_state['squeezeValue'])
        self.publish_controller_scalars('left')
        self.process_thumbstick()
        self.check_reactivate()

    def right_state_callback(self, msg):
        self.right_controller_state = self.joy_to_state(msg)
        self.right_squeeze_value = float(self.right_controller_state['squeezeValue'])
        self.publish_controller_scalars('right')
        self.process_thumbstick()
        self.check_reactivate()

    def left_pose_callback(self, msg):
        self.publish_wrist_pose(msg, 'left')

    def right_pose_callback(self, msg):
        self.publish_wrist_pose(msg, 'right')

    def left_elbow_callback(self, msg):
        self.publish_elbow_pose(msg, 'left')

    def right_elbow_callback(self, msg):
        self.publish_elbow_pose(msg, 'right')

    def joy_to_state(self, msg):
        axes = list(msg.axes)
        buttons = list(msg.buttons)
        def axis(index):
            return float(axes[index]) if len(axes) > index else 0.0
        def button(index):
            return bool(buttons[index]) if len(buttons) > index else False
        return {
            'thumbstick': button(0),
            'thumbstickValue': [axis(0), axis(1)],
            'triggerValue': axis(2),
            'squeezeValue': axis(3),
            'aButton': button(1),
            'bButton': button(2),
            'xButton': button(3),
            'yButton': button(4),
        }

    def publish_controller_scalars(self, side):
        state = self.left_controller_state if side == 'left' else self.right_controller_state
        squeeze_msg = Float32()
        squeeze_msg.data = float(state['squeezeValue'])
        trigger_msg = Float32()
        trigger_msg.data = self.calibrate_trigger(side, state['triggerValue'])
        if side == 'left':
            self.left_squeeze_pub.publish(squeeze_msg)
            self.left_trigger_pub.publish(trigger_msg)
        else:
            self.right_squeeze_pub.publish(squeeze_msg)
            self.right_trigger_pub.publish(trigger_msg)

    def check_reactivate(self):
        left_a = bool(self.left_controller_state.get('aButton', False) or self.left_controller_state.get('xButton', False))
        right_a = bool(self.right_controller_state.get('aButton', False) or self.right_controller_state.get('xButton', False))
        both_a_now = left_a and right_a
        if both_a_now and not self.both_a_buttons_pressed_prev:
            self.call_reactivate()
        self.both_a_buttons_pressed_prev = both_a_now

    def call_reactivate(self):
        if self.reactivate_call_in_flight:
            return
        if not self.reactivate_client.service_is_ready():
            now_sec = self.get_clock().now().nanoseconds / 1e9
            if (now_sec - self.last_reactivate_service_warn_sec) >= 5.0:
                self.get_logger().warn(f'Reactivate service "{self.reactivate_service}" not available')
                self.last_reactivate_service_warn_sec = now_sec
            return
        self.reactivate_call_in_flight = True
        self.reactivate_client.call_async(Trigger.Request()).add_done_callback(self.reactivate_done_callback)

    def reactivate_done_callback(self, future):
        self.reactivate_call_in_flight = False
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Reactivate service called successfully')
            else:
                self.get_logger().warn(f'Reactivate service returned: {response.message}')
        except Exception as exc:
            self.get_logger().error(f'Reactivate service call failed: {exc}')

    def publish_wrist_pose(self, msg, side):
        if not self.can_publish_goal_pose() or msg.header.frame_id != 'ros_world' or not self.head_reference_valid:
            return
        pose_key = f'{side}_wrist'
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if self.pose_min_period > 0.0 and (now_sec - self.last_pose_publish_sec[pose_key]) < self.pose_min_period:
            return

        relative_pose = self.compute_relative_sg2_pose(msg)
        if relative_pose is None:
            return
        relative_pos_ros, relative_quat_ros = relative_pose
        relative_pos_ros = self.scale_goal_position(relative_pos_ros)
        relative_rot_ros = R.from_quat(relative_quat_ros)

        base_position = relative_pos_ros - self.camera_to_base_offset
        base_position = base_position.copy()
        base_position[2] += self.get_lift_z_delta_for_arm_pose()
        base_position, base_rotation = self.apply_wrist_offsets(side, base_position, relative_rot_ros)
        quat = base_rotation.as_quat()

        target_pose = PoseStamped()
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.header.frame_id = 'base_link'
        target_pose.pose.position = self.safe_point(base_position[0], base_position[1], base_position[2])
        target_pose.pose.orientation = self.safe_quaternion(quat[0], quat[1], quat[2], quat[3])

        if side == 'left':
            self.left_wrist_rviz_pub.publish(target_pose)
        else:
            self.right_wrist_rviz_pub.publish(target_pose)
        self.last_pose_publish_sec[pose_key] = now_sec

    def publish_elbow_pose(self, msg, side):
        if not self.can_publish_goal_pose() or msg.header.frame_id != 'ros_world' or not self.head_reference_valid:
            return
        pose_key = f'{side}_elbow'
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if self.pose_min_period > 0.0 and (now_sec - self.last_pose_publish_sec[pose_key]) < self.pose_min_period:
            return

        relative_pose = self.compute_relative_sg2_pose(msg)
        if relative_pose is None:
            return
        relative_pos_ros, relative_quat_ros = relative_pose
        relative_pos_ros = self.scale_goal_position(relative_pos_ros)

        base_position = relative_pos_ros - self.camera_to_base_offset
        base_position = base_position.copy()
        base_position[2] += self.get_lift_z_delta_for_arm_pose()
        base_position = base_position + self.elbow_position_offsets[side]

        elbow_pose = PoseStamped()
        elbow_pose.header.stamp = self.get_clock().now().to_msg()
        elbow_pose.header.frame_id = 'base_link'
        elbow_pose.pose.position = self.safe_point(base_position[0], base_position[1], base_position[2])
        elbow_pose.pose.orientation = self.safe_quaternion(
            relative_quat_ros[0], relative_quat_ros[1], relative_quat_ros[2], relative_quat_ros[3]
        )

        if side == 'left':
            self.left_elbow_rviz_pub.publish(elbow_pose)
        else:
            self.right_elbow_rviz_pub.publish(elbow_pose)
        self.last_pose_publish_sec[pose_key] = now_sec

    def can_publish_goal_pose(self):
        return (
            self.vr_publishing_enabled and
            self.left_squeeze_value >= self.goal_pose_squeeze_threshold and
            self.right_squeeze_value >= self.goal_pose_squeeze_threshold
        )

    def process_thumbstick(self):
        try:
            left_thumbstick_pressed = bool(self.left_controller_state.get('thumbstick', False))
            right_thumbstick_pressed = bool(self.right_controller_state.get('thumbstick', False))

            left_thumbstick_value = self.normalize_left_thumbstick(self.left_controller_state.get('thumbstickValue', [0.0, 0.0]))
            right_thumbstick_value = self.normalize_right_thumbstick(self.right_controller_state.get('thumbstickValue', [0.0, 0.0]))

            if left_thumbstick_pressed and right_thumbstick_pressed:
                if not self.prev_left_thumbstick_pressed or not self.prev_right_thumbstick_pressed:
                    self.joystick_mode = not self.joystick_mode
                    mode_name = 'LIFT+HEAD' if self.joystick_mode else 'LIFT+CMD_VEL'
                    self.get_logger().info(f'[THUMBSTICK] Mode switched to: {mode_name}')
                    if self.joystick_mode:
                        self.publish_cmd_vel_from_thumbstick([0.0, 0.0], [0.0, 0.0])

            self.prev_left_thumbstick_pressed = left_thumbstick_pressed
            self.prev_right_thumbstick_pressed = right_thumbstick_pressed

            if abs(right_thumbstick_value[0]) > 0.0:
                self.publish_right_joystick(right_thumbstick_value[0])

            if self.joystick_mode:
                if abs(left_thumbstick_value[0]) > 0.0 or abs(left_thumbstick_value[1]) > 0.0:
                    self.publish_left_joystick_from_thumbstick(left_thumbstick_value)
            else:
                self.publish_cmd_vel_from_thumbstick(left_thumbstick_value, right_thumbstick_value)
        except Exception as exc:
            self.get_logger().error(f'Error processing thumbstick: {exc}')

    def normalize_left_thumbstick(self, value):
        lx = float(value[0]) if len(value) > 0 else 0.0
        ly = float(value[1]) if len(value) > 1 else 0.0
        if self.left_stick_swap_xy:
            lx, ly = ly, lx
        if self.left_reverse_x:
            lx = -lx
        if self.left_reverse_y:
            ly = -ly
        return [lx, ly]

    def normalize_right_thumbstick(self, value):
        rx = float(value[0]) if len(value) > 0 else 0.0
        ry = float(value[1]) if len(value) > 1 else 0.0
        if self.right_stick_swap_xy:
            rx, ry = ry, rx
        rx = -rx
        return [rx, ry]

    def publish_right_joystick(self, thumbstick_value):
        deadzone_applied_value = self.apply_deadzone(float(thumbstick_value))
        if abs(deadzone_applied_value) <= 1e-6:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        if (now_sec - self.last_lift_publish_sec) < self.control_min_period:
            return

        base_lift_position = (
            self.last_lift_command
            if self.last_lift_command is not None
            else self.lift_joint_current_position
        )
        new_lift_position = base_lift_position + deadzone_applied_value * self.right_jog_scale

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

        self.lift_joint_pub.publish(msg)
        self.last_lift_publish_sec = now_sec
        self.last_lift_command = new_lift_position

    def publish_left_joystick_from_thumbstick(self, thumbstick_value):
        deadzone_applied_x = self.apply_deadzone(float(thumbstick_value[0]))
        deadzone_applied_y = self.apply_deadzone(float(thumbstick_value[1]))
        if abs(deadzone_applied_x) <= 1e-6 and abs(deadzone_applied_y) <= 1e-6:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        if (now_sec - self.last_head_publish_sec) < self.control_min_period:
            return

        if self.last_head_command is not None:
            base_head_joint1_position, base_head_joint2_position = self.last_head_command
        else:
            base_head_joint1_position = self.head_joint1_current_position
            base_head_joint2_position = self.head_joint2_current_position

        new_head_joint1_position = base_head_joint1_position + deadzone_applied_x * self.left_jog_scale
        new_head_joint2_position = base_head_joint2_position + deadzone_applied_y * self.left_jog_scale

        msg = JointTrajectory()
        msg.joint_names = ['head_joint1', 'head_joint2']
        point = JointTrajectoryPoint()
        point.positions = [new_head_joint1_position, new_head_joint2_position]
        point.velocities = [0.0, 0.0]
        point.accelerations = [0.0, 0.0]
        point.effort = []
        msg.points.append(point)

        self.head_joint_pub.publish(msg)
        self.last_head_publish_sec = now_sec
        self.last_head_command = (new_head_joint1_position, new_head_joint2_position)

    def publish_cmd_vel_from_thumbstick(self, left_thumbstick_value, right_thumbstick_value):
        if not self.vr_publishing_enabled:
            return

        left_x_deadzone = self.apply_deadzone(float(left_thumbstick_value[0]))
        left_y_deadzone = self.apply_deadzone(float(left_thumbstick_value[1]))
        right_y_deadzone = self.apply_deadzone(float(right_thumbstick_value[1]))

        twist_msg = Twist()
        twist_msg.linear.x = -left_x_deadzone / self.linear_x_scale
        twist_msg.linear.y = left_y_deadzone / self.linear_y_scale
        twist_msg.angular.z = -right_y_deadzone / self.angular_z_scale

        cmd_tuple = (twist_msg.linear.x, twist_msg.linear.y, twist_msg.angular.z)
        is_same_command = (
            abs(cmd_tuple[0] - self.last_cmd_vel_command[0]) < 1e-5 and
            abs(cmd_tuple[1] - self.last_cmd_vel_command[1]) < 1e-5 and
            abs(cmd_tuple[2] - self.last_cmd_vel_command[2]) < 1e-5
        )

        now_sec = self.get_clock().now().nanoseconds / 1e9
        if is_same_command and abs(cmd_tuple[0]) < 1e-6 and abs(cmd_tuple[1]) < 1e-6 and abs(cmd_tuple[2]) < 1e-6:
            return
        if (now_sec - self.last_cmd_vel_publish_sec) < self.control_min_period and is_same_command:
            return

        self.cmd_vel_pub.publish(twist_msg)
        self.last_cmd_vel_publish_sec = now_sec
        self.last_cmd_vel_command = cmd_tuple

    def get_lift_z_delta_for_arm_pose(self):
        if not self.apply_lift_to_arm_z:
            return 0.0
        if self.lift_reference_position_for_pose is None:
            return 0.0
        return (
            (self.lift_joint_current_position - self.lift_reference_position_for_pose)
            * self.lift_to_arm_z_scale
        )

    def apply_wrist_offsets(self, side, position_ros, rotation_ros):
        position_with_offset = position_ros + self.wrist_position_offsets[side]
        rotation_with_offset = rotation_ros * self.wrist_rotation_offsets[side]
        return position_with_offset, rotation_with_offset

    def scale_goal_position(self, position_ros):
        return np.asarray(position_ros, dtype=np.float64) * self.goal_pose_position_scale

    def apply_deadzone(self, value):
        abs_value = abs(value)
        if abs_value < self.deadzone:
            return 0.0
        sign = 1.0 if value >= 0.0 else -1.0
        normalized_value = (abs_value - self.deadzone) / (1.0 - self.deadzone)
        return sign * normalized_value

    def calibrate_trigger(self, side, raw_value):
        calibrated = (float(raw_value) + self.trigger_offsets[side]) * self.trigger_scales[side]
        return float(np.clip(calibrated, 0.0, 1.0))

    def pose_to_matrix(self, pose):
        rotation = R.from_quat([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ])
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ], dtype=np.float64)
        return matrix

    def matrix_to_pose(self, mat):
        pos = mat[:3, 3]
        quat = R.from_matrix(mat[:3, :3]).as_quat()
        return pos, quat

    def compute_relative_sg2_pose(self, msg):
        world_matrix = self.pose_to_matrix(msg.pose)
        relative_joint_matrix = self.head_inverse_matrix @ world_matrix
        relative_pos_ros_world, relative_quat_ros_world = self.matrix_to_pose(relative_joint_matrix)
        return self.relative_ros_world_to_sg2_ros_transform(
            relative_pos_ros_world, relative_quat_ros_world
        )

    def relative_ros_world_to_sg2_ros_transform(self, ros_world_pos, ros_world_quat):
        sg2_pos = RELATIVE_ROS_WORLD_TO_SG2_ROS @ np.asarray(ros_world_pos, dtype=np.float64)
        ros_world_rotation = R.from_quat(ros_world_quat)
        sg2_rotation = (
            self.relative_ros_world_to_sg2_ros_rot
            * ros_world_rotation
            * self.relative_ros_world_to_sg2_ros_rot.inv()
        )
        return sg2_pos, sg2_rotation.as_quat()

    def is_valid_float(self, value):
        return isinstance(value, (int, float)) and np.isfinite(value)

    def safe_point(self, x, y, z):
        safe_x = float(x) if self.is_valid_float(x) else 0.0
        safe_y = float(y) if self.is_valid_float(y) else 0.0
        safe_z = float(z) if self.is_valid_float(z) else 0.0
        return Point(x=safe_x, y=safe_y, z=safe_z)

    def safe_quaternion(self, x, y, z, w):
        safe_x = float(x) if self.is_valid_float(x) else 0.0
        safe_y = float(y) if self.is_valid_float(y) else 0.0
        safe_z = float(z) if self.is_valid_float(z) else 0.0
        safe_w = float(w) if self.is_valid_float(w) else 1.0
        return Quaternion(x=safe_x, y=safe_y, z=safe_z, w=safe_w)


def main(args=None):
    rclpy.init(args=args)
    node = VRRawSG2Consumer()
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
