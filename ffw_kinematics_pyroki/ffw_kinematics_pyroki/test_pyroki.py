#!/usr/bin/env python3
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
# Author: Wonho Yun

"""Arm IK Solver ROS 2 Node using PyROKI

Inverse Kinematics solver for dual-arm robot using PyROKI.
This node implements the same logic as arm_ik_solver.cpp but using PyROKI instead of KDL.
"""

import io
import math
import contextlib
import os
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import pyroki as pk
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from ament_index_python.packages import get_package_share_directory
from typing import Sequence
import yourdfpy


def solve_ik_with_multiple_targets(
    robot: pk.Robot,
    target_link_indices: Sequence[int],
    target_wxyzs: jaxlie.SO3,
    target_positions: np.ndarray,
) -> np.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_indices: Sequence[int]. List of link indices to be controlled.
        target_wxyzs: jaxlie.SO3. Target orientation (single target).
        target_positions: np.ndarray. Shape: (num_targets, 3). Target positions.

    Returns:
        cfg: np.ndarray. Shape: (robot.joint.actuated_count,).
    """
    num_targets = len(target_link_indices)
    assert target_positions.shape == (num_targets, 3)

    # For single target, extract the first position and link index
    # _solve_ik_jax expects a single position vector (3,) and single link index
    target_position = target_positions[0]  # Shape: (3,)
    target_link_index = target_link_indices[0]  # Single integer

    cfg = _solve_ik_jax(
        robot,
        target_wxyzs,
        jnp.array(target_position),
        jnp.array(target_link_index, dtype=jnp.int32),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    return np.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_wxyz: jaxlie.SO3,
    target_position: jax.Array,
    target_link_index: jax.Array,
) -> jax.Array:
    """JAX-compiled IK solver for single target."""
    JointVar = robot.joint_var_cls

    # Create target pose (no batch dimension for single target)
    target_pose = jaxlie.SE3.from_rotation_and_translation(
       target_wxyz, target_position
    )

    factors = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            JointVar(0),
            target_pose,
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.rest_cost(
            JointVar(0),
            rest_pose=JointVar.default_factory(),
            weight=1.0,
        ),
        pk.costs.limit_cost(
            robot,
            JointVar(0),
            jnp.array([100.0] * robot.joints.num_joints),
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [JointVar(0)])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
        )
    )
    return sol[JointVar(0)]


class FfwArmIKSolver(Node):
    """ROS 2 node for dual-arm inverse kinematics solver using PyROKI."""

    def __init__(self):
        super().__init__('arm_ik_solver')

        # Declare parameters
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('arm_base_link', 'arm_base_link')
        self.declare_parameter('right_end_effector_link', 'arm_r_link7')
        self.declare_parameter('left_end_effector_link', 'arm_l_link7')
        self.declare_parameter('right_target_pose_topic', '/vr_hand/right_wrist')
        self.declare_parameter('left_target_pose_topic', '/vr_hand/left_wrist')
        self.declare_parameter('right_ik_solution_topic',
                                '/leader/joint_trajectory_command_broadcaster_right/joint_trajectory')
        self.declare_parameter('left_ik_solution_topic',
                                '/leader/joint_trajectory_command_broadcaster_left/joint_trajectory')
        self.declare_parameter('right_current_pose_topic', '/right_current_end_effector_pose')
        self.declare_parameter('left_current_pose_topic', '/left_current_end_effector_pose')

        # Coordinate transformation parameters (lift_joint origin from URDF)
        self.declare_parameter('lift_joint_x_offset', 0.0) # 0.0055)
        self.declare_parameter('lift_joint_y_offset', 0.0) # 0.0)
        self.declare_parameter('lift_joint_z_offset', 0.0) # 1.4316)

        # IK solver parameters
        self.declare_parameter('max_joint_step_degrees', 50.0)
        self.declare_parameter('ik_max_iterations', 800)
        self.declare_parameter('ik_tolerance', 1e-2)

        # Hybrid IK parameters
        self.declare_parameter('use_hybrid_ik', True)
        self.declare_parameter('current_position_weight', 0.5)
        self.declare_parameter('previous_solution_weight', 0.5)

        # Low-pass filter between current state and IK target
        self.declare_parameter('lpf_alpha', 0.8)

        # Joint limits parameters
        self.declare_parameter('use_hardcoded_joint_limits', True)

        # Get parameters
        self.base_link_ = self.get_parameter('base_link').value
        self.arm_base_link_ = self.get_parameter('arm_base_link').value
        self.right_end_effector_link_ = self.get_parameter('right_end_effector_link').value
        self.left_end_effector_link_ = self.get_parameter('left_end_effector_link').value
        right_target_pose_topic = self.get_parameter('right_target_pose_topic').value
        left_target_pose_topic = self.get_parameter('left_target_pose_topic').value
        right_ik_solution_topic = self.get_parameter('right_ik_solution_topic').value
        left_ik_solution_topic = self.get_parameter('left_ik_solution_topic').value
        right_current_pose_topic = self.get_parameter('right_current_pose_topic').value
        left_current_pose_topic = self.get_parameter('left_current_pose_topic').value

        self.lift_joint_x_offset_ = self.get_parameter('lift_joint_x_offset').value
        self.lift_joint_y_offset_ = self.get_parameter('lift_joint_y_offset').value
        self.lift_joint_z_offset_ = self.get_parameter('lift_joint_z_offset').value

        self.max_joint_step_degrees_ = self.get_parameter('max_joint_step_degrees').value
        self.ik_max_iterations_ = self.get_parameter('ik_max_iterations').value
        self.ik_tolerance_ = self.get_parameter('ik_tolerance').value

        self.use_hybrid_ik_ = self.get_parameter('use_hybrid_ik').value
        self.current_position_weight_ = self.get_parameter('current_position_weight').value
        self.previous_solution_weight_ = self.get_parameter('previous_solution_weight').value

        self.lpf_alpha_ = self.get_parameter('lpf_alpha').value
        if self.lpf_alpha_ < 0.0:
            self.lpf_alpha_ = 0.0
        if self.lpf_alpha_ > 1.0:
            self.lpf_alpha_ = 1.0

        self.use_hardcoded_joint_limits_ = self.get_parameter('use_hardcoded_joint_limits').value

        # Hardcoded joint limits (7 joints per arm)
        # Right arm limits (index 0-6: joint1-joint7)
        self.right_min_joint_positions_ = np.array([
            -3.14, -3.14, -1.57, -2.9361,
            -1.57, -1.57, -1.5804
        ])

        self.right_max_joint_positions_ = np.array([
            1.57, 0.0, 1.57, 0.0,
            1.57, 1.57, 1.8201
        ])

        # Left arm limits (index 0-6: joint1-joint7)
        # Joint2 and Joint7 have inverted rotation direction compared to right arm
        self.left_min_joint_positions_ = np.array([
            -3.14, 0.0, -1.57, -2.9361,
            -1.57, -1.57, -1.8201
        ])

        self.left_max_joint_positions_ = np.array([
            1.57, 3.14, 1.57, 0.0,
            1.57, 1.57, 1.5804
        ])

        # State variables
        self.setup_complete_ = False
        self.has_joint_states_ = False
        self.has_previous_solution_ = False
        self.robot: Optional[pk.Robot] = None

        # Joint information
        self.right_joint_names_: list[str] = []
        self.left_joint_names_: list[str] = []
        self.right_joint_indices_: list[int] = []  # Indices in robot actuated joints
        self.left_joint_indices_: list[int] = []  # Indices in robot actuated joints

        # Current joint positions
        self.right_current_joint_positions_: np.ndarray = np.array([])
        self.left_current_joint_positions_: np.ndarray = np.array([])
        self.lift_joint_position_: float = 0.0

        # Previous IK solutions for hybrid approach
        self.right_previous_solution_: Optional[np.ndarray] = None
        self.left_previous_solution_: Optional[np.ndarray] = None

        # Joint limits
        self.right_q_min_: Optional[np.ndarray] = None
        self.right_q_max_: Optional[np.ndarray] = None
        self.left_q_min_: Optional[np.ndarray] = None
        self.left_q_max_: Optional[np.ndarray] = None

        # Target link indices for PyROKI
        self.right_target_link_index_: Optional[int] = None
        self.left_target_link_index_: Optional[int] = None

        self.get_logger().info('🚀 Dual-Arm IK Solver starting...')
        self.get_logger().info(f'Base link: {self.base_link_}')
        self.get_logger().info(f'Arm base link: {self.arm_base_link_}')
        self.get_logger().info(f'Right end effector link: {self.right_end_effector_link_}')
        self.get_logger().info(f'Left end effector link: {self.left_end_effector_link_}')

        # Create subscriptions
        qos_transient_local = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.robot_description_sub_ = self.create_subscription(
            String, '/robot_description',
            self.robot_description_callback, qos_transient_local)

        self.joint_state_sub_ = self.create_subscription(
            JointState, '/joint_states',
            self.joint_state_callback, 10)

        self.right_target_pose_sub_ = self.create_subscription(
            PoseStamped, right_target_pose_topic,
            self.right_target_pose_callback, 10)

        self.left_target_pose_sub_ = self.create_subscription(
            PoseStamped, left_target_pose_topic,
            self.left_target_pose_callback, 10)

        # Create publishers
        self.right_joint_solution_pub_ = self.create_publisher(
            JointTrajectory, right_ik_solution_topic, 10)

        self.left_joint_solution_pub_ = self.create_publisher(
            JointTrajectory, left_ik_solution_topic, 10)

        self.right_current_pose_pub_ = self.create_publisher(
            PoseStamped, right_current_pose_topic, 10)

        self.left_current_pose_pub_ = self.create_publisher(
            PoseStamped, left_current_pose_topic, 10)

        # Try to get robot_description from parameter server
        try:
            robot_desc = self.get_parameter('robot_description').value
            if robot_desc:
                self.get_logger().info('Retrieved robot_description from parameter server')
                self.process_robot_description(robot_desc)
        except Exception:
            # Parameter not available, will wait for topic
            self.get_logger().info('Waiting for robot_description topic...')

        self.get_logger().info('✅ Dual-arm IK solver initialized. Waiting for target poses on:')
        self.get_logger().info(f'Right arm: {right_target_pose_topic}')
        self.get_logger().info(f'Left arm: {left_target_pose_topic}')
        self.get_logger().info(f'Publishing IK solutions on:')
        self.get_logger().info(f'Right arm: {right_ik_solution_topic}')
        self.get_logger().info(f'Left arm: {left_ik_solution_topic}')
        self.get_logger().info(f'Publishing current poses on:')
        self.get_logger().info(f'Right arm: {right_current_pose_topic}')
        self.get_logger().info(f'Left arm: {left_current_pose_topic}')

    def robot_description_callback(self, msg: String):
        """Callback for robot_description topic."""
        self.get_logger().info('Received robot_description via topic')
        self.process_robot_description(msg.data)

    def process_robot_description(self, robot_description: str):
        """Process robot description and setup PyROKI robot."""
        self.get_logger().info(f'Processing robot_description ({len(robot_description)} bytes)')

        try:
            # Filename handler that handles package:// paths
            def filename_handler(fname: str) -> str:
                """Handle file paths, skip mesh files."""
                # Skip mesh files - not needed for IK
                mesh_extensions = ('.stl', '.dae', '.obj', '.ply', '.3ds', '.blend')
                if fname.lower().endswith(mesh_extensions):
                    return ''  # Return empty to skip mesh files silently
                # Handle package:// paths
                if fname.startswith('package://'):
                    parts = fname.replace('package://', '').split('/', 1)
                    if len(parts) == 2:
                        package_name, file_path = parts
                        try:
                            package_path = get_package_share_directory(package_name)
                            full_path = os.path.join(package_path, file_path)
                            if os.path.exists(full_path):
                                return full_path
                        except Exception:
                            pass
                    return fname
                return fname

            # Load URDF
            urdf_io = io.StringIO(robot_description)
            with open(os.devnull, 'w') as devnull:
                # Temporarily redirect stderr to suppress "Can't find" messages
                with contextlib.redirect_stderr(devnull):
                    urdf = yourdfpy.URDF.load(urdf_io, filename_handler=filename_handler)

            # Validate URDF
            if not urdf.validate():
                self.get_logger().warn('URDF validation failed. Proceeding anyway...')
            else:
                self.get_logger().info('URDF validation passed.')

            # Create PyROKI robot
            self.robot = pk.Robot.from_urdf(urdf)

            self.get_logger().info('✅ PyROKI robot loaded successfully!')
            self.get_logger().info(f'   Number of actuated joints: {self.robot.joints.num_actuated_joints}')

            # Verify target links exist
            if self.right_end_effector_link_ not in self.robot.links.names:
                self.get_logger().error(f"Error: Target link '{self.right_end_effector_link_}' not found!")
                return
            if self.left_end_effector_link_ not in self.robot.links.names:
                self.get_logger().error(f"Error: Target link '{self.left_end_effector_link_}' not found!")
                return

            # Get target link indices
            self.right_target_link_index_ = self.robot.links.names.index(self.right_end_effector_link_)
            self.left_target_link_index_ = self.robot.links.names.index(self.left_end_effector_link_)

            # Extract joint names
            self.extract_joint_names()

            # Setup joint limits
            self.setup_joint_limits()

            # Initialize previous solution arrays
            self.right_previous_solution_ = np.zeros(len(self.right_joint_names_))
            self.left_previous_solution_ = np.zeros(len(self.left_joint_names_))

            self.setup_complete_ = True
            self.get_logger().info('🎉 IK solver setup complete!')
            self.get_logger().info(f'   Hybrid IK: {"enabled" if self.use_hybrid_ik_ else "disabled"} '
                                  f'(current: {self.current_position_weight_ * 100.0:.1f}%, '
                                  f'previous: {self.previous_solution_weight_ * 100.0:.1f}%)')

        except Exception as e:
            self.get_logger().error(f'Exception during robot description processing: {e}')
            import traceback
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')

    def extract_joint_names(self):
        """Extract joint names for left and right arms."""
        # Get actuated joint names from robot
        if hasattr(self.robot.joints, 'actuated_names'):
            actuated_joint_names = list(self.robot.joints.actuated_names)
        else:
            actuated_joint_names = list(self.robot.joints.names[:self.robot.joints.num_actuated_joints])

        # Extract right arm joints (arm_r_joint1-7)
        self.right_joint_names_ = []
        self.right_joint_indices_ = []
        for i in range(1, 8):  # joint1 to joint7
            joint_name = f'arm_r_joint{i}'
            if joint_name in actuated_joint_names:
                idx = actuated_joint_names.index(joint_name)
                self.right_joint_names_.append(joint_name)
                self.right_joint_indices_.append(idx)
                self.get_logger().info(f'📋 Right: {joint_name} at actuated index {idx}')

        # Extract left arm joints (arm_l_joint1-7)
        self.left_joint_names_ = []
        self.left_joint_indices_ = []
        for i in range(1, 8):  # joint1 to joint7
            joint_name = f'arm_l_joint{i}'
            if joint_name in actuated_joint_names:
                idx = actuated_joint_names.index(joint_name)
                self.left_joint_names_.append(joint_name)
                self.left_joint_indices_.append(idx)
                self.get_logger().info(f'📋 Left: {joint_name} at actuated index {idx}')

        self.get_logger().info(f'✅ Right arm joints ({len(self.right_joint_names_)}): {self.right_joint_names_}')
        self.get_logger().info(f'✅ Left arm joints ({len(self.left_joint_names_)}): {self.left_joint_names_}')

    def setup_joint_limits(self):
        """Setup joint limits using hardcoded values."""
        if self.use_hardcoded_joint_limits_:
            # Setup right arm joint limits
            num_right_joints = len(self.right_joint_names_)
            if num_right_joints != len(self.right_min_joint_positions_):
                self.get_logger().warn(
                    f'Right arm joint count mismatch: chain has {num_right_joints} joints, '
                    f'hardcoded limits for {len(self.right_min_joint_positions_)}')
            else:
                self.right_q_min_ = self.right_min_joint_positions_[:num_right_joints].copy()
                self.right_q_max_ = self.right_max_joint_positions_[:num_right_joints].copy()
                self.get_logger().info('🔒 Setting up right arm joint limits with hardcoded values:')
                for i in range(num_right_joints):
                    self.get_logger().info(f'  Joint {i}: [{self.right_q_min_[i]:.3f}, {self.right_q_max_[i]:.3f}] rad')

            # Setup left arm joint limits
            num_left_joints = len(self.left_joint_names_)
            if num_left_joints != len(self.left_min_joint_positions_):
                self.get_logger().warn(
                    f'Left arm joint count mismatch: chain has {num_left_joints} joints, '
                    f'hardcoded limits for {len(self.left_min_joint_positions_)}')
            else:
                self.left_q_min_ = self.left_min_joint_positions_[:num_left_joints].copy()
                self.left_q_max_ = self.left_max_joint_positions_[:num_left_joints].copy()
                self.get_logger().info('🔒 Setting up left arm joint limits with hardcoded values:')
                for i in range(num_left_joints):
                    self.get_logger().info(f'  Joint {i}: [{self.left_q_min_[i]:.3f}, {self.left_q_max_[i]:.3f}] rad')

            self.get_logger().info('✅ Joint limits configured for both arms using hardcoded values')

    def joint_state_callback(self, msg: JointState):
        """Callback for joint_states topic."""
        if not self.setup_complete_:
            return

        # Extract current lift_joint position for coordinate transformation
        self.lift_joint_position_ = 0.0
        for i, joint_name in enumerate(msg.name):
            if joint_name == 'lift_joint' and i < len(msg.position):
                self.lift_joint_position_ = msg.position[i]
                break

        # Extract right arm joint positions
        self.right_current_joint_positions_ = np.zeros(len(self.right_joint_names_))
        right_all_joints_found = True
        for i, joint_name in enumerate(self.right_joint_names_):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                if idx < len(msg.position):
                    self.right_current_joint_positions_[i] = msg.position[idx]
                else:
                    right_all_joints_found = False
            else:
                right_all_joints_found = False
                self.get_logger().warn(f'Joint {joint_name} not found in joint_states')

        # Extract left arm joint positions
        self.left_current_joint_positions_ = np.zeros(len(self.left_joint_names_))
        left_all_joints_found = True
        for i, joint_name in enumerate(self.left_joint_names_):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                if idx < len(msg.position):
                    self.left_current_joint_positions_[i] = msg.position[idx]
                else:
                    left_all_joints_found = False
            else:
                left_all_joints_found = False
                self.get_logger().warn(f'Joint {joint_name} not found in joint_states')

        if right_all_joints_found and left_all_joints_found and not self.has_joint_states_:
            self.has_joint_states_ = True
            self.get_logger().info('✅ All joint states received. IK solver ready!')
            self.check_current_joint_limits()

        # Publish current poses on each joint state update
        self.publish_current_poses()

    def check_current_joint_limits(self):
        """Check current joint positions against limits."""
        self.get_logger().info('🔍 Checking current joint positions against limits:')

        # Check right arm joints
        right_all_within_limits = True
        self.get_logger().info('Right arm joints:')
        for i in range(len(self.right_current_joint_positions_)):
            pos = self.right_current_joint_positions_[i]
            min_limit = self.right_q_min_[i]
            max_limit = self.right_q_max_[i]
            within_limits = (pos >= min_limit and pos <= max_limit)
            if not within_limits:
                right_all_within_limits = False
            self.get_logger().info(
                f'  {self.right_joint_names_[i]}: {pos:.3f} rad [{min_limit:.3f}, {max_limit:.3f}] '
                f'{"✅" if within_limits else "❌"}')

        # Check left arm joints
        left_all_within_limits = True
        self.get_logger().info('Left arm joints:')
        for i in range(len(self.left_current_joint_positions_)):
            pos = self.left_current_joint_positions_[i]
            min_limit = self.left_q_min_[i]
            max_limit = self.left_q_max_[i]
            within_limits = (pos >= min_limit and pos <= max_limit)
            if not within_limits:
                left_all_within_limits = False
            self.get_logger().info(
                f'  {self.left_joint_names_[i]}: {pos:.3f} rad [{min_limit:.3f}, {max_limit:.3f}] '
                f'{"✅" if within_limits else "❌"}')

        if right_all_within_limits and left_all_within_limits:
            self.get_logger().info('✅ All current joint positions are within limits')
        else:
            self.get_logger().warn('⚠️ Some joints are outside limits - this is OK for initialization')

    def right_target_pose_callback(self, msg: PoseStamped):
        """Callback for right arm target pose."""
        if not self.setup_complete_ or not self.has_joint_states_:
            self.get_logger().warn('IK solver not ready. Ignoring right target pose.')
            return

        # Validate pose input
        if not (math.isfinite(msg.pose.position.x) and math.isfinite(msg.pose.position.y) and
                math.isfinite(msg.pose.position.z) and math.isfinite(msg.pose.orientation.w) and
                math.isfinite(msg.pose.orientation.x) and math.isfinite(msg.pose.orientation.y) and
                math.isfinite(msg.pose.orientation.z)):
            self.get_logger().error('Received invalid (non-finite) right target pose. Ignoring.')
            return

        # Transform pose from base_link to arm_base_link frame
        arm_base_pose = PoseStamped()
        arm_base_pose.header = msg.header
        arm_base_pose.pose = msg.pose

        # Transform: base_link -> arm_base_link using configured offsets
        # Note: lift_joint_position_ is not used to make IK independent of lift height
        arm_base_pose.pose.position.x -= self.lift_joint_x_offset_
        arm_base_pose.pose.position.y -= self.lift_joint_y_offset_
        arm_base_pose.pose.position.z -= self.lift_joint_z_offset_

        # Solve IK for the transformed target
        self.solve_ik(arm_base_pose, 'right')

    def left_target_pose_callback(self, msg: PoseStamped):
        """Callback for left arm target pose."""
        if not self.setup_complete_ or not self.has_joint_states_:
            self.get_logger().warn('IK solver not ready. Ignoring left target pose.')
            return

        # Validate pose input
        if not (math.isfinite(msg.pose.position.x) and math.isfinite(msg.pose.position.y) and
                math.isfinite(msg.pose.position.z) and math.isfinite(msg.pose.orientation.w) and
                math.isfinite(msg.pose.orientation.x) and math.isfinite(msg.pose.orientation.y) and
                math.isfinite(msg.pose.orientation.z)):
            self.get_logger().error('Received invalid (non-finite) left target pose. Ignoring.')
            return

        # Transform pose from base_link to arm_base_link frame
        arm_base_pose = PoseStamped()
        arm_base_pose.header = msg.header
        arm_base_pose.pose = msg.pose

        # Transform: base_link -> arm_base_link using configured offsets
        # Note: lift_joint_position_ is not used to make IK independent of lift height
        arm_base_pose.pose.position.x -= self.lift_joint_x_offset_
        arm_base_pose.pose.position.y -= self.lift_joint_y_offset_
        arm_base_pose.pose.position.z -= self.lift_joint_z_offset_

        # Solve IK for the transformed target
        self.solve_ik(arm_base_pose, 'left')

    def solve_ik(self, target_pose: PoseStamped, arm: str):
        """Solve inverse kinematics for specified arm."""
        if self.robot is None:
            return

        # Select arm-specific variables
        if arm == 'right':
            joint_names = self.right_joint_names_
            joint_indices = self.right_joint_indices_
            current_positions = self.right_current_joint_positions_
            q_min = self.right_q_min_
            q_max = self.right_q_max_
            previous_solution = self.right_previous_solution_
            publisher = self.right_joint_solution_pub_
            target_link_index = self.right_target_link_index_
        else:
            joint_names = self.left_joint_names_
            joint_indices = self.left_joint_indices_
            current_positions = self.left_current_joint_positions_
            q_min = self.left_q_min_
            q_max = self.left_q_max_
            previous_solution = self.left_previous_solution_
            publisher = self.left_joint_solution_pub_
            target_link_index = self.left_target_link_index_

        # Get initial guess using hybrid approach
        if self.use_hybrid_ik_ and self.has_previous_solution_:
            # Hybrid: weighted combination of current position and previous solution
            q_init = (self.current_position_weight_ * current_positions +
                     self.previous_solution_weight_ * previous_solution)
            self.get_logger().debug(
                f'Using hybrid initial guess for {arm} arm '
                f'({self.current_position_weight_ * 100.0:.1f}% current + '
                f'{self.previous_solution_weight_ * 100.0:.1f}% previous)')
        else:
            # Fallback: use only current positions
            q_init = current_positions.copy()
            self.get_logger().debug(f'Using current position as initial guess for {arm} arm')

        # Clamp initial guess to joint limits with margin
        clamp_margin = 0.1  # radians
        for i in range(len(q_init)):
            min_limit = q_min[i]
            max_limit = q_max[i]
            if q_init[i] < min_limit:
                target = min_limit + clamp_margin
                if target > max_limit:
                    target = max_limit
                q_init[i] = target
                self.get_logger().debug(
                    f'Clamped {arm} arm initial guess for joint {i} to min+margin ({q_init[i]:.3f})')
            if q_init[i] > max_limit:
                target = max_limit - clamp_margin
                if target < min_limit:
                    target = min_limit
                q_init[i] = target
                self.get_logger().debug(
                    f'Clamped {arm} arm initial guess for joint {i} to max-margin ({q_init[i]:.3f})')

        # Convert target pose to PyROKI format
        # Convert quaternion from xyzw to rotation matrix, then to SO3
        qx = target_pose.pose.orientation.x
        qy = target_pose.pose.orientation.y
        qz = target_pose.pose.orientation.z
        qw = target_pose.pose.orientation.w

        # Convert xyzw quaternion to rotation matrix
        # Quaternion to rotation matrix conversion (xyzw format)
        rotation_matrix = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])

        # Create SO3 from rotation matrix (same as PAPRLE usage)
        target_wxyz = jaxlie.SO3.from_matrix(rotation_matrix)
        target_position = np.array([
            target_pose.pose.position.x,
            target_pose.pose.position.y,
            target_pose.pose.position.z
        ]).reshape(1, 3)  # Shape: (1, 3) for single target

        # Solve IK using PyROKI
        initial_start_time = self.get_clock().now()
        solution = None

        try:
            # Call PyROKI solver
            # Note: PAPRLE's solve_ik_with_multiple_targets doesn't support initial configuration
            # but it has rest_cost which should help find solutions close to current position
            # target_wxyzs should be jaxlie.SO3 object (single target)
            solution = solve_ik_with_multiple_targets(
                robot=self.robot,
                target_link_indices=[target_link_index],
                target_wxyzs=target_wxyz,
                target_positions=target_position
            )

            if solution is None or len(solution) != self.robot.joints.num_actuated_joints:
                self.get_logger().error(f'❌ {arm} arm IK failed: Invalid solution')
                return

            # Extract arm-specific joints from solution
            q_result = np.zeros(len(joint_names))
            for i, joint_idx in enumerate(joint_indices):
                if joint_idx < len(solution):
                    q_result[i] = solution[joint_idx]
                else:
                    self.get_logger().error(
                        f'{arm} arm joint {joint_names[i]} index {joint_idx} >= solution length {len(solution)}')
                    q_result[i] = current_positions[i]

            end_time = self.get_clock().now()
            duration = (end_time - initial_start_time).nanoseconds / 1e9
            self.get_logger().debug(
                f'{arm} arm IK solver time: {duration * 1000.0:.3f} ms, {1.0 / duration:.1f} Hz')

        except Exception as e:
            self.get_logger().error(f'❌ {arm} arm IK failed with exception: {e}')
            import traceback
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
            return

        # Verify all joints are within limits
        all_within_limits = True
        for i in range(len(q_result)):
            if q_result[i] < q_min[i] or q_result[i] > q_max[i]:
                all_within_limits = False
                self.get_logger().warn(
                    f'{arm} arm joint {i} solution {q_result[i]:.3f} is outside limits '
                    f'[{q_min[i]:.3f}, {q_max[i]:.3f}]')

        if not all_within_limits:
            self.get_logger().error(f'❌ {arm} arm IK solution violates joint limits! Skipping.')
            return

        # Apply low-pass filter: blend current position toward IK target
        q_filtered = np.zeros(len(q_result))
        for i in range(len(q_result)):
            current_pos = current_positions[i]
            target_pos = q_result[i]
            q_filtered[i] = (1.0 - self.lpf_alpha_) * current_pos + self.lpf_alpha_ * target_pos
            # Clamp to joint hard limits
            if q_filtered[i] < q_min[i]:
                q_filtered[i] = q_min[i]
            if q_filtered[i] > q_max[i]:
                q_filtered[i] = q_max[i]

        # Clamp joint movement to max step for safety (applied to filtered values)
        max_joint_step = self.max_joint_step_degrees_ * math.pi / 180.0
        clamped = False
        for i in range(len(q_filtered)):
            delta = q_filtered[i] - current_positions[i]
            if abs(delta) > max_joint_step:
                clamped = True
                if delta > 0:
                    q_filtered[i] = current_positions[i] + max_joint_step
                else:
                    q_filtered[i] = current_positions[i] - max_joint_step

        if clamped:
            self.get_logger().debug(
                f'⚠️ {arm} arm joint movement clamped to max {self.max_joint_step_degrees_:.1f} deg per cycle for safety.')

        # Store solution for next iteration (hybrid approach)
        if arm == 'right':
            self.right_previous_solution_ = q_filtered.copy()
        else:
            self.left_previous_solution_ = q_filtered.copy()
        self.has_previous_solution_ = True

        # Create and publish JointTrajectory message with the solution
        joint_trajectory = JointTrajectory()
        joint_trajectory.header.frame_id = 'base_link'
        joint_trajectory.joint_names = joint_names

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = q_filtered.tolist()
        point.velocities = [0.0] * len(joint_names)
        point.accelerations = [0.0] * len(joint_names)
        point.time_from_start = rclpy.duration.Duration(seconds=0.0).to_msg()
        joint_trajectory.points = [point]

        publisher.publish(joint_trajectory)

    def publish_current_poses(self):
        """Publish current end effector poses."""
        if not self.setup_complete_ or not self.has_joint_states_:
            return

        # For now, we skip this as it requires forward kinematics
        # This can be implemented later if needed
        pass


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    node = FfwArmIKSolver()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
