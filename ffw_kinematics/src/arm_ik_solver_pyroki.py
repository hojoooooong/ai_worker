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

import math
import io
import os
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
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
from scipy.spatial.transform import Rotation
from ament_index_python.packages import get_package_share_directory
import yourdfpy


def solve_ik_pyroki(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: np.ndarray,
    target_position: np.ndarray,
    initial_cfg: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solve inverse kinematics using PyROKI.

    Args:
        robot: PyROKI Robot object
        target_link_name: Name of the target link
        target_wxyz: Target orientation in wxyz format (numpy array, shape (4,))
        target_position: Target position (numpy array, shape (3,))
        initial_cfg: Initial joint configuration (optional)

    Returns:
        Joint configuration solution (numpy array, shape (num_actuated_joints,))
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_index = robot.links.names.index(target_link_name)

    # Convert to JAX arrays
    target_link_idx = jnp.array(target_link_index)
    target_wxyz_jax = jnp.array(target_wxyz)
    target_pos_jax = jnp.array(target_position)

    # Use initial configuration if provided, otherwise use default
    if initial_cfg is not None:
        initial_cfg_jax = jnp.array(initial_cfg)
    else:
        initial_cfg_jax = None

    # Solve IK
    cfg = _solve_ik_jax(robot, target_link_idx, target_wxyz_jax, target_pos_jax, initial_cfg_jax)

    assert cfg.shape == (robot.joints.num_actuated_joints,)
    return np.array(cfg)


@jax.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    initial_cfg: Optional[jax.Array] = None,
) -> jax.Array:
    """JAX-compiled IK solver."""
    # Create joint variable with initial configuration if provided
    # Use initial_cfg if provided, otherwise use default (0)
    if initial_cfg is not None:
        # Create joint variable with initial configuration
        # joint_var_cls expects a single value, but we need to pass the full array
        # We'll create it with 0 and then manually set the initial value in the solve
        joint_var = robot.joint_var_cls(0)
        # Store initial_cfg for use in solve
        initial_cfg_value = initial_cfg
    else:
        joint_var = robot.joint_var_cls(0)
        initial_cfg_value = None

    # Create target SE(3) transform
    target_se3 = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )

    # Define cost factors with higher limit_cost weight to enforce joint limits
    factors = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            target_se3,
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=10000.0,  # Increased to 10000.0 to strictly enforce joint limits
        ),
    ]

    # Create problem
    problem = jaxls.LeastSquaresProblem(factors, [joint_var])
    analyzed = problem.analyze()

    # Solve least squares problem
    # jaxls uses default_factory for initial guess, which is always 0
    # We can't directly override this, but the solver should still converge
    sol = analyzed.solve(
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
    )

    result = sol[joint_var]

    # If initial_cfg was provided, blend the result with initial_cfg
    # This helps guide the solution toward the initial guess
    # This is a workaround since jaxls doesn't directly support initial values
    if initial_cfg_value is not None:
        # Blend: 30% initial_cfg + 70% solver result
        # This helps the solver start closer to the initial guess
        blend_weight = 0.3
        result = blend_weight * initial_cfg_value + (1.0 - blend_weight) * result

    return result


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

        # Coordinate transformation parameters
        self.declare_parameter('lift_joint_x_offset', 0.0055)
        self.declare_parameter('lift_joint_y_offset', 0.0)
        self.declare_parameter('lift_joint_z_offset', 1.6316)

        # IK solver parameters
        self.declare_parameter('max_joint_step_degrees', 30.0)
        self.declare_parameter('ik_max_iterations', 800)
        self.declare_parameter('ik_tolerance', 1e-2)

        # Hybrid IK parameters
        self.declare_parameter('use_hybrid_ik', True)
        self.declare_parameter('current_position_weight', 0.5)
        self.declare_parameter('previous_solution_weight', 0.5)

        # Low-pass filter parameter
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

        self.lpf_alpha_ = max(0.0, min(1.0, self.get_parameter('lpf_alpha').value))

        self.use_hardcoded_joint_limits_ = self.get_parameter('use_hardcoded_joint_limits').value

        # Hardcoded joint limits (7 joints per arm)
        # Right arm limits
        self.right_min_joint_positions_ = np.array([
            -3.14, -3.14, -1.57, -2.9361,
            -1.57, -1.57, -1.5804
        ])
        self.right_max_joint_positions_ = np.array([
            1.57, 0.0, 1.57, 0.5,
            1.57, 1.57, 1.8201
        ])

        # Left arm limits
        self.left_min_joint_positions_ = np.array([
            -3.14, 0.0, -1.57, -2.9361,
            -1.57, -1.57, -1.8201
        ])
        self.left_max_joint_positions_ = np.array([
            1.57, 3.14, 1.57, 0.5,
            1.57, 1.57, 1.5804
        ])

        # State variables
        self.setup_complete_ = False
        self.has_joint_states_ = False
        self.has_previous_solution_ = False
        self.lift_joint_position_ = 0.0

        # PyROKI robot and joint information
        self.robot: Optional[pk.Robot] = None
        self.right_joint_names_ = []
        self.left_joint_names_ = []
        self.right_joint_indices_ = []
        self.left_joint_indices_ = []
        self.right_q_min_ = None
        self.right_q_max_ = None
        self.left_q_min_ = None
        self.left_q_max_ = None

        # Current joint positions
        self.right_current_joint_positions_ = []
        self.left_current_joint_positions_ = []

        # Previous IK solutions
        self.right_previous_solution_ = None
        self.left_previous_solution_ = None

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
        self.get_logger().info('Publishing IK solutions on:')
        self.get_logger().info(f'Right arm: {right_ik_solution_topic}')
        self.get_logger().info(f'Left arm: {left_ik_solution_topic}')
        self.get_logger().info('Publishing current poses on:')
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
            # Filename handler that completely ignores mesh files
            # IK calculation only needs kinematic structure (joints, links, transforms)
            # Mesh files are for visualization/simulation only, not needed for IK
            def filename_handler(fname: str) -> str:
                """Handle file paths, but skip mesh files completely."""
                # Skip all mesh/visual files - not needed for IK
                mesh_extensions = ('.stl', '.dae', '.obj', '.ply', '.3ds', '.blend')
                if fname.lower().endswith(mesh_extensions):
                    return ''  # Return empty to skip
                # For URDF includes (if any), try to resolve package:// paths
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

            # Load URDF - mesh files are automatically skipped
            # Only kinematic structure is parsed, which is all we need for IK
            urdf_io = io.StringIO(robot_description)
            urdf = yourdfpy.URDF.load(urdf_io, filename_handler=filename_handler)

    # Validate URDF
            if not urdf.validate():
                self.get_logger().warn('URDF validation failed. Proceeding anyway...')
            else:
                self.get_logger().info('URDF validation passed.')

            # Log all joints from URDF (before PyROKI processing)
            # yourdfpy.URDF uses joint_map (dict) and actuated_joints (list)
            all_joints_in_urdf = [joint.name for joint in urdf.joint_map.values()]
            actuated_joints_in_urdf = [joint.name for joint in urdf.actuated_joints]
            self.get_logger().info(f'📋 URDF: All joints ({len(all_joints_in_urdf)}): {all_joints_in_urdf}')
            self.get_logger().info(f'📋 URDF: Actuated joints ({len(actuated_joints_in_urdf)}): {actuated_joints_in_urdf}')

            # Check for arm_r_joint4~7 in all joints
            arm_r_joints_all = [j for j in all_joints_in_urdf if j.startswith('arm_r_joint')]
            arm_r_joints_actuated = [j for j in actuated_joints_in_urdf if j.startswith('arm_r_joint')]
            self.get_logger().info(f'📋 URDF: All arm_r_joint* joints: {arm_r_joints_all}')
            self.get_logger().info(f'📋 URDF: Actuated arm_r_joint* joints: {arm_r_joints_actuated}')

            # Check joint types for arm_r_joint4~7 if they exist
            for joint_name in ['arm_r_joint4', 'arm_r_joint5', 'arm_r_joint6', 'arm_r_joint7']:
                if joint_name in urdf.joint_map:
                    joint = urdf.joint_map[joint_name]
                    is_actuated = joint in urdf.actuated_joints
                    self.get_logger().info(
                        f'📋 URDF: {joint_name} - type: {joint.type}, actuated: {is_actuated}')

            # Create PyROKI robot
            self.robot = pk.Robot.from_urdf(urdf)

            self.get_logger().info(f'✅ PyROKI robot loaded successfully!')
            self.get_logger().info(f'   Robot name: {urdf.robot.name}')
            self.get_logger().info(f'   Number of actuated joints: {self.robot.joints.num_actuated_joints}')

            # Store URDF actuated joints for joint extraction
            # PyROKI may reorder joints internally, so we use URDF's actuated_joints directly
            self.urdf_actuated_joint_names_ = [joint.name for joint in urdf.actuated_joints]

            # Extract joint names for both arms
            self.extract_joint_names()

            # Setup joint limits
            self.setup_joint_limits(urdf)

            # Initialize previous solution arrays
            num_right_joints = len(self.right_joint_names_)
            num_left_joints = len(self.left_joint_names_)
            self.right_previous_solution_ = np.zeros(num_right_joints)
            self.left_previous_solution_ = np.zeros(num_left_joints)

            self.setup_complete_ = True
            self.get_logger().info('🎉 IK solver setup complete!')
            self.get_logger().info(
                f'   Hybrid IK: {"enabled" if self.use_hybrid_ik_ else "disabled"} '
                f'(current: {self.current_position_weight_ * 100.0:.1f}%, '
                f'previous: {self.previous_solution_weight_ * 100.0:.1f}%)')

        except Exception as e:
            self.get_logger().error(f'Exception during robot description processing: {e}')

    def extract_joint_names(self):
        """Extract joint names for right and left arms from the robot."""
        # Use URDF actuated joints directly (PyROKI may reorder joints internally)
        actuated_joint_names = self.urdf_actuated_joint_names_

        # Get PyROKI robot's joint names to map indices
        robot_joint_names = list(self.robot.joints.names)
        num_actuated = self.robot.joints.num_actuated_joints
        num_total = len(robot_joint_names)

        # Log all robot joints for debugging
        self.get_logger().info(
            f'📋 PyROKI robot: Total joints ({num_total}), Actuated ({num_actuated})')
        self.get_logger().info(
            f'📋 PyROKI robot ALL joints: {robot_joint_names}')

        # Check if arm_r_joint4~7 exist in full robot.joints.names
        for joint_name in ['arm_r_joint4', 'arm_r_joint5', 'arm_r_joint6', 'arm_r_joint7']:
            if joint_name in robot_joint_names:
                idx = robot_joint_names.index(joint_name)
                is_actuated = idx < num_actuated
                self.get_logger().info(
                    f'📋 PyROKI: {joint_name} found at index {idx}, actuated: {is_actuated}')
            else:
                self.get_logger().warn(
                    f'📋 PyROKI: {joint_name} NOT FOUND in robot.joints.names!')

        # Create mapping from URDF actuated joint name to PyROKI robot index
        # PyROKI uses first num_actuated_joints elements for actuated joints
        robot_actuated_names = robot_joint_names[:num_actuated]
        name_to_robot_idx = {name: i for i, name in enumerate(robot_actuated_names)}

        # Also check in full list (beyond actuated) for missing joints
        name_to_robot_idx_full = {name: i for i, name in enumerate(robot_joint_names)}

        # For arm joints that are not in actuated list, use full list indices
        # This allows us to use arm_r_joint4~7 even if they're beyond num_actuated
        # We'll need to handle this in IK solving by using full joint configuration
        self.use_full_joint_list_for_arms_ = False
        missing_arm_joints = []
        for joint_name in actuated_joint_names:
            if (joint_name.startswith('arm_r_joint') or joint_name.startswith('arm_l_joint')):
                if joint_name not in name_to_robot_idx and joint_name in name_to_robot_idx_full:
                    missing_arm_joints.append(joint_name)

        if missing_arm_joints:
            self.get_logger().warn(
                f'⚠️ Found {len(missing_arm_joints)} arm joints outside actuated range: {missing_arm_joints}')
            self.get_logger().warn(
                f'⚠️ Will use full joint list indices for these joints. '
                f'Note: PyROKI IK solver may not work correctly with joints beyond num_actuated.')
            self.use_full_joint_list_for_arms_ = True

        # Find joints that belong to right arm (arm_r_*)
        self.right_joint_names_ = []
        self.right_joint_indices_ = []
        for joint_name in actuated_joint_names:
            # Match arm_r_joint1, arm_r_joint2, ..., arm_r_joint7
            if joint_name.startswith('arm_r_joint'):
                self.right_joint_names_.append(joint_name)
                # Map to PyROKI robot index
                if joint_name in name_to_robot_idx:
                    self.right_joint_indices_.append(name_to_robot_idx[joint_name])
                elif joint_name in name_to_robot_idx_full:
                    # Joint exists but is not in actuated list
                    # Use full list index - we'll handle this in IK solving
                    idx = name_to_robot_idx_full[joint_name]
                    self.right_joint_indices_.append(idx)
                    self.get_logger().warn(
                        f'⚠️ Right arm joint {joint_name} found at index {idx} but NOT in actuated joints! '
                        f'Will use full joint list index.')
                else:
                    self.get_logger().warn(
                        f'⚠️ Right arm joint {joint_name} not found in PyROKI robot at all!')

        # Log all actuated joints for debugging
        self.get_logger().info(f'📋 URDF actuated joints ({len(actuated_joint_names)}): {actuated_joint_names}')
        self.get_logger().info(f'📋 PyROKI robot actuated joints ({len(robot_actuated_names)}): {robot_actuated_names}')
        self.get_logger().info(f'📋 Right arm joints found ({len(self.right_joint_names_)}): {self.right_joint_names_}')

        # Find joints that belong to left arm (arm_l_*)
        self.left_joint_names_ = []
        self.left_joint_indices_ = []
        for joint_name in actuated_joint_names:
            # Match arm_l_joint1, arm_l_joint2, ..., arm_l_joint7
            if joint_name.startswith('arm_l_joint'):
                self.left_joint_names_.append(joint_name)
                # Map to PyROKI robot index
                if joint_name in name_to_robot_idx:
                    self.left_joint_indices_.append(name_to_robot_idx[joint_name])
                elif joint_name in name_to_robot_idx_full:
                    # Joint exists but is not in actuated list
                    # Use full list index - we'll handle this in IK solving
                    idx = name_to_robot_idx_full[joint_name]
                    self.left_joint_indices_.append(idx)
                    self.get_logger().warn(
                        f'⚠️ Left arm joint {joint_name} found at index {idx} but NOT in actuated joints! '
                        f'Will use full joint list index.')
                else:
                    self.get_logger().warn(
                        f'⚠️ Left arm joint {joint_name} not found in PyROKI robot at all!')

        self.get_logger().info(f'📋 Left arm joints found ({len(self.left_joint_names_)}): {self.left_joint_names_}')

        # Sort by joint number to ensure correct order
        def sort_key(name):
            # Extract number from joint name like "arm_r_joint1" -> 1
            import re
            match = re.search(r'joint(\d+)', name)
            return int(match.group(1)) if match else 0

        # Sort names and rebuild indices
        self.right_joint_names_.sort(key=sort_key)
        self.left_joint_names_.sort(key=sort_key)

        # Rebuild indices after sorting
        # Use full list indices for all joints (including those beyond num_actuated)
        self.right_joint_indices_ = [
            name_to_robot_idx_full[name] if name in name_to_robot_idx_full else name_to_robot_idx.get(name, -1)
            for name in self.right_joint_names_
        ]
        self.left_joint_indices_ = [
            name_to_robot_idx_full[name] if name in name_to_robot_idx_full else name_to_robot_idx.get(name, -1)
            for name in self.left_joint_names_
        ]

        # Filter out invalid indices and corresponding names
        # Use zip to keep names and indices in sync
        right_pairs = [(name, idx) for name, idx in zip(self.right_joint_names_, self.right_joint_indices_) if idx >= 0]
        left_pairs = [(name, idx) for name, idx in zip(self.left_joint_names_, self.left_joint_indices_) if idx >= 0]

        self.right_joint_names_ = [name for name, idx in right_pairs]
        self.right_joint_indices_ = [idx for name, idx in right_pairs]
        self.left_joint_names_ = [name for name, idx in left_pairs]
        self.left_joint_indices_ = [idx for name, idx in left_pairs]

        self.get_logger().info('Right arm joint names extracted:')
        for i, name in enumerate(self.right_joint_names_):
            self.get_logger().info(f'  [{i}] {name}')

        self.get_logger().info('Left arm joint names extracted:')
        for i, name in enumerate(self.left_joint_names_):
            self.get_logger().info(f'  [{i}] {name}')

    def setup_joint_limits(self, urdf):
        """Setup joint limits from URDF or hardcoded values."""
        if self.use_hardcoded_joint_limits_:
            self.setup_hardcoded_joint_limits()
        else:
            self.setup_urdf_joint_limits(urdf)

    def setup_hardcoded_joint_limits(self):
        """Setup joint limits using hardcoded values."""
        num_right_joints = len(self.right_joint_names_)
        num_left_joints = len(self.left_joint_names_)

        if num_right_joints != len(self.right_min_joint_positions_):
            self.get_logger().warn(
                f'Right arm joint count mismatch: found {num_right_joints} joints, '
                f'hardcoded limits for {len(self.right_min_joint_positions_)}')
            # Pad or truncate as needed
            if num_right_joints < len(self.right_min_joint_positions_):
                self.right_min_joint_positions_ = self.right_min_joint_positions_[:num_right_joints]
                self.right_max_joint_positions_ = self.right_max_joint_positions_[:num_right_joints]
            else:
                # Extend with default limits
                default_min = -math.pi
                default_max = math.pi
                while len(self.right_min_joint_positions_) < num_right_joints:
                    self.right_min_joint_positions_ = np.append(
                        self.right_min_joint_positions_, default_min)
                    self.right_max_joint_positions_ = np.append(
                        self.right_max_joint_positions_, default_max)

        if num_left_joints != len(self.left_min_joint_positions_):
            self.get_logger().warn(
                f'Left arm joint count mismatch: found {num_left_joints} joints, '
                f'hardcoded limits for {len(self.left_min_joint_positions_)}')
            # Pad or truncate as needed
            if num_left_joints < len(self.left_min_joint_positions_):
                self.left_min_joint_positions_ = self.left_min_joint_positions_[:num_left_joints]
                self.left_max_joint_positions_ = self.left_max_joint_positions_[:num_left_joints]
            else:
                # Extend with default limits
                default_min = -math.pi
                default_max = math.pi
                while len(self.left_min_joint_positions_) < num_left_joints:
                    self.left_min_joint_positions_ = np.append(
                        self.left_min_joint_positions_, default_min)
                    self.left_max_joint_positions_ = np.append(
                        self.left_max_joint_positions_, default_max)

        self.right_q_min_ = self.right_min_joint_positions_
        self.right_q_max_ = self.right_max_joint_positions_
        self.left_q_min_ = self.left_min_joint_positions_
        self.left_q_max_ = self.left_max_joint_positions_

        self.get_logger().info('🔒 Setting up right arm joint limits with hardcoded values:')
        for i in range(num_right_joints):
            self.get_logger().info(
                f'  Joint {i}: [{self.right_q_min_[i]:.3f}, {self.right_q_max_[i]:.3f}] rad')

        self.get_logger().info('🔒 Setting up left arm joint limits with hardcoded values:')
        for i in range(num_left_joints):
            self.get_logger().info(
                f'  Joint {i}: [{self.left_q_min_[i]:.3f}, {self.left_q_max_[i]:.3f}] rad')

        self.get_logger().info('✅ Joint limits configured for both arms using hardcoded values')

    def setup_urdf_joint_limits(self, urdf):
        """Setup joint limits from URDF."""
        self.get_logger().info('🔒 Setting up joint limits from URDF...')

        num_right_joints = len(self.right_joint_names_)
        num_left_joints = len(self.left_joint_names_)

        self.right_q_min_ = np.zeros(num_right_joints)
        self.right_q_max_ = np.zeros(num_right_joints)

        for i, joint_name in enumerate(self.right_joint_names_):
            joint = urdf.joint_map.get(joint_name)
            if joint and joint.limit:
                self.right_q_min_[i] = joint.limit.lower
                self.right_q_max_[i] = joint.limit.upper
                self.get_logger().info(
                    f'  Right {joint_name}: [{self.right_q_min_[i]:.3f}, {self.right_q_max_[i]:.3f}] rad')
            else:
                self.get_logger().warn(f'No limits found for right joint: {joint_name}')
                self.right_q_min_[i] = -math.pi
                self.right_q_max_[i] = math.pi

        self.left_q_min_ = np.zeros(num_left_joints)
        self.left_q_max_ = np.zeros(num_left_joints)

        for i, joint_name in enumerate(self.left_joint_names_):
            joint = urdf.joint_map.get(joint_name)
            if joint and joint.limit:
                self.left_q_min_[i] = joint.limit.lower
                self.left_q_max_[i] = joint.limit.upper
                self.get_logger().info(
                    f'  Left {joint_name}: [{self.left_q_min_[i]:.3f}, {self.left_q_max_[i]:.3f}] rad')
            else:
                self.get_logger().warn(f'No limits found for left joint: {joint_name}')
                self.left_q_min_[i] = -math.pi
                self.left_q_max_[i] = math.pi

        self.get_logger().info('✅ Joint limits configured for both arms from URDF')

    def joint_state_callback(self, msg: JointState):
        """Callback for joint_states topic."""
        if not self.setup_complete_:
            return

        # Extract current lift_joint position
        self.lift_joint_position_ = 0.0
        if 'lift_joint' in msg.name:
            idx = msg.name.index('lift_joint')
            self.lift_joint_position_ = msg.position[idx]

        # Extract right arm joint positions
        self.right_current_joint_positions_ = [0.0] * len(self.right_joint_names_)
        right_all_joints_found = True
        for i, joint_name in enumerate(self.right_joint_names_):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                self.right_current_joint_positions_[i] = msg.position[idx]
            else:
                right_all_joints_found = False
                self.get_logger().warn(f'Joint {joint_name} not found in joint_states')

        # Extract left arm joint positions
        self.left_current_joint_positions_ = [0.0] * len(self.left_joint_names_)
        left_all_joints_found = True
        for i, joint_name in enumerate(self.left_joint_names_):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                self.left_current_joint_positions_[i] = msg.position[idx]
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
        epsilon = 1e-6  # Small epsilon for floating point comparison
        for i, joint_name in enumerate(self.right_joint_names_):
            pos = self.right_current_joint_positions_[i]
            min_limit = self.right_q_min_[i]
            max_limit = self.right_q_max_[i]
            # Use epsilon for floating point comparison to handle -0.0 vs 0.0
            within_limits = (min_limit - epsilon <= pos <= max_limit + epsilon)
            if not within_limits:
                right_all_within_limits = False
            status = '✅' if within_limits else '❌'
            self.get_logger().info(
                f'  {joint_name}: {pos:.3f} rad [{min_limit:.3f}, {max_limit:.3f}] {status}')

        # Check left arm joints
        left_all_within_limits = True
        self.get_logger().info('Left arm joints:')
        epsilon = 1e-6  # Small epsilon for floating point comparison
        for i, joint_name in enumerate(self.left_joint_names_):
            pos = self.left_current_joint_positions_[i]
            min_limit = self.left_q_min_[i]
            max_limit = self.left_q_max_[i]
            # Use epsilon for floating point comparison to handle -0.0 vs 0.0
            within_limits = (min_limit - epsilon <= pos <= max_limit + epsilon)
            if not within_limits:
                left_all_within_limits = False
            status = '✅' if within_limits else '❌'
            self.get_logger().info(
                f'  {joint_name}: {pos:.3f} rad [{min_limit:.3f}, {max_limit:.3f}] {status}')

        if right_all_within_limits and left_all_within_limits:
            self.get_logger().info('✅ All current joint positions are within limits')
        else:
            self.get_logger().warn(
                '⚠️ Some joints are outside limits - this is OK for initialization')

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

        # Log received target pose
        self.get_logger().info(
            f'📥 Right arm target pose received: '
            f'pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}), '
            f'quat=({msg.pose.orientation.w:.3f}, {msg.pose.orientation.x:.3f}, '
            f'{msg.pose.orientation.y:.3f}, {msg.pose.orientation.z:.3f})')

        # Transform pose from base_link to arm_base_link frame
        arm_base_pose = PoseStamped()
        arm_base_pose.header = msg.header
        arm_base_pose.pose = msg.pose

        # Transform: base_link -> arm_base_link using configured offsets
        arm_base_pose.pose.position.x -= self.lift_joint_x_offset_
        arm_base_pose.pose.position.y -= self.lift_joint_y_offset_
        arm_base_pose.pose.position.z -= (self.lift_joint_z_offset_ + self.lift_joint_position_)

        # Log transformed pose
        self.get_logger().info(
            f'🔄 Right arm target pose transformed to arm_base_link: '
            f'pos=({arm_base_pose.pose.position.x:.3f}, {arm_base_pose.pose.position.y:.3f}, '
            f'{arm_base_pose.pose.position.z:.3f})')

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

        # Log received target pose
        self.get_logger().info(
            f'📥 Left arm target pose received: '
            f'pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}), '
            f'quat=({msg.pose.orientation.w:.3f}, {msg.pose.orientation.x:.3f}, '
            f'{msg.pose.orientation.y:.3f}, {msg.pose.orientation.z:.3f})')

        # Transform pose from base_link to arm_base_link frame
        arm_base_pose = PoseStamped()
        arm_base_pose.header = msg.header
        arm_base_pose.pose = msg.pose

        # Transform: base_link -> arm_base_link using configured offsets
        arm_base_pose.pose.position.x -= self.lift_joint_x_offset_
        arm_base_pose.pose.position.y -= self.lift_joint_y_offset_
        arm_base_pose.pose.position.z -= (self.lift_joint_z_offset_ + self.lift_joint_position_)

        # Log transformed pose
        self.get_logger().info(
            f'🔄 Left arm target pose transformed to arm_base_link: '
            f'pos=({arm_base_pose.pose.position.x:.3f}, {arm_base_pose.pose.position.y:.3f}, '
            f'{arm_base_pose.pose.position.z:.3f})')

        # Solve IK for the transformed target
        self.solve_ik(arm_base_pose, 'left')

    def solve_ik(self, target_pose: PoseStamped, arm: str):
        """Solve inverse kinematics for the given target pose."""
        if self.robot is None:
            return

        # Select arm-specific variables
        # Store original lists for later use (to include all joints in JointTrajectory)
        if arm == 'right':
            original_joint_names = self.right_joint_names_
            original_joint_indices = self.right_joint_indices_
            original_current_positions = self.right_current_joint_positions_
            joint_names = self.right_joint_names_
            joint_indices = self.right_joint_indices_
            current_positions = self.right_current_joint_positions_
            q_min = self.right_q_min_
            q_max = self.right_q_max_
            previous_solution = self.right_previous_solution_
            publisher = self.right_joint_solution_pub_
            end_effector_link = self.right_end_effector_link_
        else:
            original_joint_names = self.left_joint_names_
            original_joint_indices = self.left_joint_indices_
            original_current_positions = self.left_current_joint_positions_
            joint_names = self.left_joint_names_
            joint_indices = self.left_joint_indices_
            current_positions = self.left_current_joint_positions_
            q_min = self.left_q_min_
            q_max = self.left_q_max_
            previous_solution = self.left_previous_solution_
            publisher = self.left_joint_solution_pub_
            end_effector_link = self.left_end_effector_link_

        # Convert target pose to PyROKI format
        # PyROKI expects wxyz quaternion format (not xyzw)
        qx = target_pose.pose.orientation.x
        qy = target_pose.pose.orientation.y
        qz = target_pose.pose.orientation.z
        qw = target_pose.pose.orientation.w

        # Convert xyzw to wxyz format
        target_wxyz = np.array([qw, qx, qy, qz])
        target_position = np.array([
            target_pose.pose.position.x,
            target_pose.pose.position.y,
            target_pose.pose.position.z
        ])

        # Filter out joints that are beyond num_actuated_joints
        # PyROKI IK solver only uses num_actuated_joints, so we need to exclude joints beyond that range
        num_actuated = self.robot.joints.num_actuated_joints
        valid_indices = []
        valid_names = []
        valid_positions = []
        valid_q_min = []
        valid_q_max = []
        valid_previous = []
        excluded_joints = []

        for i, joint_idx in enumerate(joint_indices):
            if joint_idx < num_actuated:
                valid_indices.append(joint_idx)
                valid_names.append(joint_names[i])
                valid_positions.append(current_positions[i])
                valid_q_min.append(q_min[i])
                valid_q_max.append(q_max[i])
                if self.has_previous_solution_ and i < len(previous_solution):
                    valid_previous.append(previous_solution[i])
            else:
                excluded_joints.append(joint_names[i])

        if excluded_joints:
            self.get_logger().warn(
                f'⚠️ {arm} arm: Excluding {len(excluded_joints)} joints from IK solving '
                f'(beyond num_actuated={num_actuated}): {excluded_joints}')

        # Update variables to use only valid joints
        joint_indices = valid_indices
        joint_names = valid_names
        current_positions = np.array(valid_positions)
        q_min = np.array(valid_q_min)
        q_max = np.array(valid_q_max)
        if self.has_previous_solution_:
            previous_solution = np.array(valid_previous)

        # Get initial guess using hybrid approach
        num_joints = len(joint_names)
        q_init = np.zeros(num_actuated)

        # Set initial guess for arm joints
        if self.use_hybrid_ik_ and self.has_previous_solution_:
            # Hybrid: weighted combination of current position and previous solution
            for i, joint_idx in enumerate(joint_indices):
                q_init[joint_idx] = (
                    self.current_position_weight_ * current_positions[i] +
                    self.previous_solution_weight_ * previous_solution[i]
                )
            self.get_logger().info(
                f'🔧 {arm} arm IK: Using hybrid initial guess '
                f'(current: {self.current_position_weight_*100:.0f}%, '
                f'previous: {self.previous_solution_weight_*100:.0f}%)')
        else:
            # Fallback: use only current positions
            for i, joint_idx in enumerate(joint_indices):
                q_init[joint_idx] = current_positions[i]
            self.get_logger().info(f'🔧 {arm} arm IK: Using current position as initial guess')

        # Clamp initial guess to joint limits with margin
        clamp_margin = 0.1  # radians
        for i, joint_idx in enumerate(joint_indices):
            min_limit = q_min[i]
            max_limit = q_max[i]
            if q_init[joint_idx] < min_limit:
                target = min_limit + clamp_margin
                if target > max_limit:
                    target = max_limit
                q_init[joint_idx] = target
            if q_init[joint_idx] > max_limit:
                target = max_limit - clamp_margin
                if target < min_limit:
                    target = min_limit
                q_init[joint_idx] = target

        # Solve IK using PyROKI
        try:
            self.get_logger().info(
                f'🔍 {arm} arm IK solving: target_link={end_effector_link}, '
                f'target_pos=({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})')

            q_result = solve_ik_pyroki(
                robot=self.robot,
                target_link_name=end_effector_link,
                target_wxyz=target_wxyz,
                target_position=target_position,
                initial_cfg=q_init
            )

            if q_result is None or len(q_result) != self.robot.joints.num_actuated_joints:
                self.get_logger().error(f'❌ {arm} arm IK failed to converge')
                return

            self.get_logger().info(f'✅ {arm} arm IK solution computed successfully')

            # Extract solution for arm joints only
            q_arm_result = np.array([q_result[idx] for idx in joint_indices])

            # Log raw IK solution before any processing
            self.get_logger().info(
                f'🔬 {arm} arm raw IK solution: '
                f'{[f"{name}:{val:.3f}" for name, val in zip(joint_names, q_arm_result)]}')

            # Clamp solution to joint limits before verification
            # This ensures we always have a valid solution even if IK solver slightly violates limits
            q_arm_result_clamped = np.array(q_arm_result)
            clamping_needed = False
            clamped_joints = []
            for i in range(len(joint_names)):
                original_val = q_arm_result[i]
                clamped_val = max(q_min[i], min(q_max[i], original_val))
                q_arm_result_clamped[i] = clamped_val
                if abs(original_val - clamped_val) > 1e-6:
                    clamping_needed = True
                    clamped_joints.append((i, joint_names[i], original_val, clamped_val, q_min[i], q_max[i]))

            # Log clamping if significant (more than 0.01 rad difference)
            if clamping_needed:
                significant_clamping = any(abs(orig - clamped) > 0.01 for _, _, orig, clamped, _, _ in clamped_joints)
                if significant_clamping:
                    # Only log joints with significant clamping
                    significant_joints = [(i, name, orig, clamped, qmin, qmax)
                                         for i, name, orig, clamped, qmin, qmax in clamped_joints
                                         if abs(orig - clamped) > 0.01]
                    self.get_logger().warn(
                        f'⚠️ {arm} arm IK solution clamped (significant): '
                        f'{[(name, f"{orig:.3f}->{clamped:.3f}") for _, name, orig, clamped, _, _ in significant_joints]}')
                else:
                    # Minor clamping - use debug level
                    self.get_logger().debug(
                        f'{arm} arm IK solution clamped (minor): '
                        f'{[(name, f"{orig:.3f}->{clamped:.3f}") for _, name, orig, clamped, _, _ in clamped_joints]}')

            # Use clamped result
            q_arm_result = q_arm_result_clamped

            # Apply low-pass filter: blend current position toward IK target
            q_filtered = np.zeros(len(joint_names))
            for i in range(len(joint_names)):
                current_pos = current_positions[i]
                target_pos = q_arm_result[i]
                q_filtered[i] = (1.0 - self.lpf_alpha_) * current_pos + self.lpf_alpha_ * target_pos
                # Clamp to joint hard limits
                q_filtered[i] = max(q_min[i], min(q_max[i], q_filtered[i]))

            # Log after LPF
            self.get_logger().info(
                f'🔬 {arm} arm after LPF (alpha={self.lpf_alpha_}): '
                f'{[f"{name}:{val:.3f}" for name, val in zip(joint_names, q_filtered)]}')

            # Clamp joint movement to max step for safety
            max_joint_step = math.radians(self.max_joint_step_degrees_)
            clamped = False
            clamped_joints_step = []
            for i in range(len(joint_names)):
                delta = q_filtered[i] - current_positions[i]
                if abs(delta) > max_joint_step:
                    clamped = True
                    old_val = q_filtered[i]
                    if delta > 0:
                        q_filtered[i] = current_positions[i] + max_joint_step
                    else:
                        q_filtered[i] = current_positions[i] - max_joint_step
                    clamped_joints_step.append((joint_names[i], old_val, q_filtered[i], delta))

            if clamped:
                self.get_logger().info(
                    f'⏸️ {arm} arm joint movement clamped to max step ({self.max_joint_step_degrees_}°): '
                    f'{[(name, f"{old:.3f}->{new:.3f}, delta={d:.3f}") for name, old, new, d in clamped_joints_step]}')

            # Store solution for next iteration (hybrid approach)
            # Store full solution including excluded joints (they keep current position)
            full_previous_solution = np.zeros(len(original_joint_names))
            valid_joint_name_to_result = {name: val for name, val in zip(joint_names, q_filtered)}

            for i, joint_name in enumerate(original_joint_names):
                if joint_name in valid_joint_name_to_result:
                    full_previous_solution[i] = valid_joint_name_to_result[joint_name]
                else:
                    # Keep current position for excluded joints
                    full_previous_solution[i] = original_current_positions[i]

            if arm == 'right':
                self.right_previous_solution_ = full_previous_solution.copy()
            else:
                self.left_previous_solution_ = full_previous_solution.copy()
            self.has_previous_solution_ = True

            # Create and publish JointTrajectory message
            # Include all original joints (filtered joints keep current position)
            joint_trajectory = JointTrajectory()
            joint_trajectory.header.frame_id = 'base_link'
            joint_trajectory.joint_names = original_joint_names

            # Build full position array: IK solution for valid joints, current position for excluded joints
            full_positions = np.zeros(len(original_joint_names))
            valid_joint_name_to_result = {name: val for name, val in zip(joint_names, q_filtered)}

            for i, joint_name in enumerate(original_joint_names):
                if joint_name in valid_joint_name_to_result:
                    full_positions[i] = valid_joint_name_to_result[joint_name]
                else:
                    # Keep current position for excluded joints
                    full_positions[i] = original_current_positions[i]

            # Create trajectory point
            point = JointTrajectoryPoint()
            point.positions = full_positions.tolist()
            point.velocities = [0.0] * len(original_joint_names)
            point.accelerations = [0.0] * len(original_joint_names)
            point.time_from_start = rclpy.duration.Duration(seconds=0.0).to_msg()

            joint_trajectory.points = [point]
            publisher.publish(joint_trajectory)

            # Log published solution
            self.get_logger().info(
                f'📤 {arm} arm IK solution published: '
                f'joints={[f"{name}:{pos:.3f}" for name, pos in zip(original_joint_names, full_positions)]}')

        except Exception as e:
            import traceback
            self.get_logger().error(f'❌ Exception during IK solving for {arm} arm: {e}')
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')

    def publish_current_poses(self):
        """Publish current end-effector poses using forward kinematics."""
        if not self.setup_complete_ or not self.has_joint_states_ or self.robot is None:
            return

        # Get full joint configuration
        # PyROKI uses num_actuated_joints for forward_kinematics, but we need to handle
        # joints beyond that range. We'll use the full joint list size.
        num_actuated = self.robot.joints.num_actuated_joints
        num_total = len(self.robot.joints.names)

        # Create full joint configuration (use total joints size)
        # For joints beyond num_actuated, we'll pad with zeros
        q_full = np.zeros(num_total)

        # Set right arm joint positions (using full list indices)
        for i, joint_idx in enumerate(self.right_joint_indices_):
            if joint_idx < num_total:
                q_full[joint_idx] = self.right_current_joint_positions_[i]
            else:
                self.get_logger().warn(
                    f'Right arm joint index {joint_idx} is out of bounds (max: {num_total-1})')

        # Set left arm joint positions (using full list indices)
        for i, joint_idx in enumerate(self.left_joint_indices_):
            if joint_idx < num_total:
                q_full[joint_idx] = self.left_current_joint_positions_[i]
            else:
                self.get_logger().warn(
                    f'Left arm joint index {joint_idx} is out of bounds (max: {num_total-1})')

        # PyROKI's forward_kinematics expects num_actuated_joints, so we need to extract
        # only the actuated portion. However, if we have joints beyond num_actuated,
        # we need to handle this differently.
        # For now, use only actuated joints for forward_kinematics
        q_actuated = q_full[:num_actuated]

        # Compute forward kinematics for all links
        try:
            # forward_kinematics returns poses for all links in order of self.links.names
            # Format: (link_count, 7) where 7 is [w, x, y, z, px, py, pz] (wxyz_xyz)
            # Note: PyROKI's forward_kinematics only uses num_actuated_joints
            # If we have joints beyond num_actuated, they won't be used in FK
            all_link_poses = self.robot.forward_kinematics(q_actuated)

            # Get link indices
            right_link_idx = self.robot.links.names.index(self.right_end_effector_link_)
            left_link_idx = self.robot.links.names.index(self.left_end_effector_link_)

            # Extract right arm end effector pose
            right_ee_pose = all_link_poses[right_link_idx]  # Shape: (7,)
            right_pose = PoseStamped()
            right_pose.header.frame_id = self.arm_base_link_
            right_pose.header.stamp = self.get_clock().now().to_msg()

            # Extract position (last 3 elements)
            right_pose.pose.position.x = float(right_ee_pose[4])
            right_pose.pose.position.y = float(right_ee_pose[5])
            right_pose.pose.position.z = float(right_ee_pose[6])

            # Extract orientation (first 4 elements are wxyz)
            right_pose.pose.orientation.w = float(right_ee_pose[0])
            right_pose.pose.orientation.x = float(right_ee_pose[1])
            right_pose.pose.orientation.y = float(right_ee_pose[2])
            right_pose.pose.orientation.z = float(right_ee_pose[3])

            self.right_current_pose_pub_.publish(right_pose)

            self.get_logger().info(
                f'📤 Right arm current EE pose published: '
                f'pos=({right_pose.pose.position.x:.3f}, {right_pose.pose.position.y:.3f}, '
                f'{right_pose.pose.position.z:.3f})')

            # Extract left arm end effector pose
            left_ee_pose = all_link_poses[left_link_idx]  # Shape: (7,)
            left_pose = PoseStamped()
            left_pose.header.frame_id = self.arm_base_link_
            left_pose.header.stamp = self.get_clock().now().to_msg()

            # Extract position (last 3 elements)
            left_pose.pose.position.x = float(left_ee_pose[4])
            left_pose.pose.position.y = float(left_ee_pose[5])
            left_pose.pose.position.z = float(left_ee_pose[6])

            # Extract orientation (first 4 elements are wxyz)
            left_pose.pose.orientation.w = float(left_ee_pose[0])
            left_pose.pose.orientation.x = float(left_ee_pose[1])
            left_pose.pose.orientation.y = float(left_ee_pose[2])
            left_pose.pose.orientation.z = float(left_ee_pose[3])

            self.left_current_pose_pub_.publish(left_pose)

            self.get_logger().info(
                f'📤 Left arm current EE pose published: '
                f'pos=({left_pose.pose.position.x:.3f}, {left_pose.pose.position.y:.3f}, '
                f'{left_pose.pose.position.z:.3f})')
        except Exception as e:
            self.get_logger().debug(f'❌ Failed to compute FK: {e}', exc_info=True)


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    node = FfwArmIKSolver()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
