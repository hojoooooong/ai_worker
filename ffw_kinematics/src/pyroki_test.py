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

"""Bimanual Robot IK ROS 2 Node

Inverse Kinematics for both arms of the robot using PyROKI.
This node receives robot description from ROS 2 topic and solves IK for both arms simultaneously.
"""

import io
import math
import sys
import contextlib
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
from ament_index_python.packages import get_package_share_directory
import yourdfpy
import os


def solve_ik_with_multiple_targets(
    robot: pk.Robot,
    target_link_names: list[str],
    target_wxyzs: np.ndarray,
    target_positions: np.ndarray,
    initial_cfg: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solve inverse kinematics for multiple target links simultaneously.

    Args:
        robot: PyROKI Robot object
        target_link_names: List of target link names
        target_wxyzs: Target orientations in wxyz format (numpy array, shape (N, 4))
        target_positions: Target positions (numpy array, shape (N, 3))
        initial_cfg: Initial joint configuration (optional)

    Returns:
        Joint configuration solution (numpy array, shape (num_actuated_joints,))
    """
    assert len(target_link_names) == len(target_wxyzs) == len(target_positions)
    assert target_wxyzs.shape[1] == 4 and target_positions.shape[1] == 3

    # Get target link indices
    target_link_indices = [robot.links.names.index(name) for name in target_link_names]
    target_link_indices_jax = jnp.array(target_link_indices)

    # Convert to JAX arrays
    target_wxyzs_jax = jnp.array(target_wxyzs)
    target_positions_jax = jnp.array(target_positions)

    # Use initial configuration if provided, otherwise use zeros
    # This ensures we always pass a valid array to the JAX function
    if initial_cfg is not None:
        initial_cfg_jax = jnp.array(initial_cfg)
    else:
        initial_cfg_jax = jnp.zeros(robot.joints.num_actuated_joints)

    # Solve IK
    cfg = _solve_ik_multiple_targets_jax(
        robot, target_link_indices_jax, target_wxyzs_jax, target_positions_jax, initial_cfg_jax
    )

    assert cfg.shape == (robot.joints.num_actuated_joints,)
    return np.array(cfg)


@jax.jit
def _solve_ik_multiple_targets_jax(
    robot: pk.Robot,
    target_link_indices: jax.Array,
    target_wxyzs: jax.Array,
    target_positions: jax.Array,
    initial_cfg: jax.Array,  # Now always provided (zeros if None in outer function)
) -> jax.Array:
    """JAX-compiled IK solver for multiple targets."""
    # Create joint variable - joint_var_cls takes an index, not initial value
    joint_var = robot.joint_var_cls(0)

    # Create cost factors for each target
    factors = []

    # Add pose cost for each target link
    for i in range(len(target_link_indices)):
        target_se3 = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(target_wxyzs[i]), target_positions[i]
        )
        factors.append(
            pk.costs.pose_cost_analytic_jac(
                robot,
                joint_var,
                target_se3,
                target_link_indices[i],
                pos_weight=50.0,
                ori_weight=10.0,
            )
        )

    # Add joint limit cost
    factors.append(
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=10000.0,
        )
    )

    # Create problem
    problem = jaxls.LeastSquaresProblem(factors, [joint_var])
    analyzed = problem.analyze()

    # Solve least squares problem with initial values
    # initial_cfg is always provided (zeros if None in outer function)
    sol = analyzed.solve(
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        initial_vals=jaxls.VarValues.make(
            [joint_var.with_value(initial_cfg)]
        ),
    )

    result = sol[joint_var]
    return result


class BimanualIKSolver(Node):
    """ROS 2 node for bimanual inverse kinematics solver using PyROKI."""

    def __init__(self):
        super().__init__('bimanual_ik_solver')

        # Declare parameters
        self.declare_parameter('left_end_effector_link', 'arm_l_link7')
        self.declare_parameter('right_end_effector_link', 'arm_r_link7')
        self.declare_parameter('right_target_pose_topic', '/vr_hand/right_wrist')
        self.declare_parameter('left_target_pose_topic', '/vr_hand/left_wrist')
        self.declare_parameter('right_ik_solution_topic', '/leader/joint_trajectory_command_broadcaster_right/joint_trajectory')
        self.declare_parameter('left_ik_solution_topic', '/leader/joint_trajectory_command_broadcaster_left/joint_trajectory')

        # Get parameters
        self.left_end_effector_link_ = self.get_parameter('left_end_effector_link').value
        self.right_end_effector_link_ = self.get_parameter('right_end_effector_link').value
        right_target_pose_topic = self.get_parameter('right_target_pose_topic').value
        left_target_pose_topic = self.get_parameter('left_target_pose_topic').value
        right_ik_solution_topic = self.get_parameter('right_ik_solution_topic').value
        left_ik_solution_topic = self.get_parameter('left_ik_solution_topic').value

        # State variables
        self.setup_complete_ = False
        self.robot: Optional[pk.Robot] = None

        # Store target poses
        self.right_target_pose_: Optional[PoseStamped] = None
        self.left_target_pose_: Optional[PoseStamped] = None

        # Current joint positions (all joints, not just actuated)
        self.current_joint_positions_: Optional[np.ndarray] = None
        self.current_joint_positions_dict_: dict[str, float] = {}  # Store all joint positions by name
        self.joint_names_: list[str] = []

        # Arm-specific joint names and indices
        self.left_joint_names_: list[str] = []
        self.left_joint_indices_: list[int] = []
        self.right_joint_names_: list[str] = []
        self.right_joint_indices_: list[int] = []

        # Track which joints we've warned about (to avoid spam)
        self.warned_missing_joints_: set[str] = set()

        self.get_logger().info('🚀 Bimanual IK Solver starting...')
        self.get_logger().info(f'Left end effector link: {self.left_end_effector_link_}')
        self.get_logger().info(f'Right end effector link: {self.right_end_effector_link_}')

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

        # Try to get robot_description from parameter server
        try:
            robot_desc = self.get_parameter('robot_description').value
            if robot_desc:
                self.get_logger().info('Retrieved robot_description from parameter server')
                self.process_robot_description(robot_desc)
        except Exception:
            # Parameter not available, will wait for topic
            self.get_logger().info('Waiting for robot_description topic...')

        self.get_logger().info('✅ Bimanual IK solver initialized. Waiting for target poses on:')
        self.get_logger().info(f'Right arm: {right_target_pose_topic}')
        self.get_logger().info(f'Left arm: {left_target_pose_topic}')

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

            # Load URDF (suppress "Can't find" messages from yourdfpy for missing mesh files)
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

            self.get_logger().info(f'✅ PyROKI robot loaded successfully!')
            self.get_logger().info(f'   Robot name: {urdf.robot.name}')
            self.get_logger().info(f'   Number of actuated joints: {self.robot.joints.num_actuated_joints}')

            # Verify target links exist
            if self.left_end_effector_link_ not in self.robot.links.names:
                self.get_logger().error(f"Error: Target link '{self.left_end_effector_link_}' not found!")
                return
            if self.right_end_effector_link_ not in self.robot.links.names:
                self.get_logger().error(f"Error: Target link '{self.right_end_effector_link_}' not found!")
                return

            # Store joint names from robot
            self.joint_names_ = list(self.robot.joints.names[:self.robot.joints.num_actuated_joints])

            # Extract arm-specific joint names and indices
            self.extract_arm_joint_info()

            self.setup_complete_ = True
            self.get_logger().info('🎉 IK solver setup complete!')

        except Exception as e:
            self.get_logger().error(f'Exception during robot description processing: {e}')
            import traceback
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')

    def extract_arm_joint_info(self):
        """Extract joint names and indices for left and right arms."""
        # Get all joint names from robot
        all_joint_names = list(self.robot.joints.names)
        num_actuated = self.robot.joints.num_actuated_joints

        self.get_logger().info(f'📋 Total joints: {len(all_joint_names)}, Actuated: {num_actuated}')

        # Extract left arm joints (arm_l_joint1-7) with their indices
        left_joint_pairs = []
        for i in range(1, 8):  # joint1 to joint7
            joint_name = f'arm_l_joint{i}'
            if joint_name in all_joint_names:
                idx = all_joint_names.index(joint_name)
                is_actuated = idx < num_actuated
                self.get_logger().info(f'📋 Left: {joint_name} at index {idx}, actuated: {is_actuated}')
                # Include all joints, even if beyond num_actuated
                # We'll handle this when extracting from solution
                left_joint_pairs.append((joint_name, idx))
            else:
                self.get_logger().warn(f'⚠️ Left: {joint_name} not found in robot joints!')

        # Extract right arm joints (arm_r_joint1-7) with their indices
        right_joint_pairs = []
        for i in range(1, 8):  # joint1 to joint7
            joint_name = f'arm_r_joint{i}'
            if joint_name in all_joint_names:
                idx = all_joint_names.index(joint_name)
                is_actuated = idx < num_actuated
                self.get_logger().info(f'📋 Right: {joint_name} at index {idx}, actuated: {is_actuated}')
                # Include all joints, even if beyond num_actuated
                # We'll handle this when extracting from solution
                right_joint_pairs.append((joint_name, idx))
            else:
                self.get_logger().warn(f'⚠️ Right: {joint_name} not found in robot joints!')

        # Sort by joint number to ensure correct order
        def sort_key(pair):
            import re
            name = pair[0]
            match = re.search(r'joint(\d+)', name)
            return int(match.group(1)) if match else 0

        # Sort pairs and extract names and indices
        left_joint_pairs.sort(key=sort_key)
        right_joint_pairs.sort(key=sort_key)

        self.left_joint_names_ = [name for name, _ in left_joint_pairs]
        self.left_joint_indices_ = [idx for _, idx in left_joint_pairs]
        self.right_joint_names_ = [name for name, _ in right_joint_pairs]
        self.right_joint_indices_ = [idx for _, idx in right_joint_pairs]

        self.get_logger().info(f'✅ Left arm joints ({len(self.left_joint_names_)}): {self.left_joint_names_}')
        self.get_logger().info(f'✅ Right arm joints ({len(self.right_joint_names_)}): {self.right_joint_names_}')

    def joint_state_callback(self, msg: JointState):
        """Callback for joint_states topic."""
        if not self.setup_complete_:
            return

        # Store all joint positions in a dictionary for easy lookup
        for i, joint_name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions_dict_[joint_name] = msg.position[i]

        # Debug: Log once to verify all arm joints are in joint_states
        if hasattr(self, 'right_joint_names_') and len(self.right_joint_names_) > 0:
            if not hasattr(self, '_checked_missing_joints'):
                self._checked_missing_joints = True
                missing_right = [name for name in self.right_joint_names_ if name not in self.current_joint_positions_dict_]
                missing_left = [name for name in self.left_joint_names_ if name not in self.current_joint_positions_dict_]
                if missing_right:
                    available_arm_r = [name for name in self.current_joint_positions_dict_.keys() if name.startswith('arm_r_')]
                    self.get_logger().warn(
                        f'⚠️ Right arm joints NOT in joint_states: {missing_right}. '
                        f'Available arm_r_* joints: {available_arm_r}')
                if missing_left:
                    available_arm_l = [name for name in self.current_joint_positions_dict_.keys() if name.startswith('arm_l_')]
                    self.get_logger().warn(
                        f'⚠️ Left arm joints NOT in joint_states: {missing_left}. '
                        f'Available arm_l_* joints: {available_arm_l}')
                if not missing_right and not missing_left:
                    self.get_logger().info('✅ All arm joints found in joint_states')

        # Debug: Log once which right arm joints are missing from joint_states
        if hasattr(self, 'right_joint_names_') and len(self.right_joint_names_) > 0:
            if not hasattr(self, '_checked_missing_joints'):
                self._checked_missing_joints = True
                missing_right = [name for name in self.right_joint_names_ if name not in self.current_joint_positions_dict_]
                if missing_right:
                    available_arm_r = [name for name in self.current_joint_positions_dict_.keys() if name.startswith('arm_r_')]
                    self.get_logger().warn(
                        f'⚠️ Right arm joints NOT in joint_states: {missing_right}. '
                        f'Available arm_r_* joints: {available_arm_r}')

        # Extract current joint positions for actuated joints (for IK initial guess)
        num_actuated = self.robot.joints.num_actuated_joints
        self.current_joint_positions_ = np.zeros(num_actuated)

        for i, joint_name in enumerate(self.joint_names_):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                self.current_joint_positions_[i] = msg.position[idx]
            else:
                # Only warn once per missing joint (many joints like cameras/lidars are fixed)
                if joint_name not in self.warned_missing_joints_:
                    self.warned_missing_joints_.add(joint_name)
                    self.get_logger().debug(
                        f'Joint {joint_name} not found in joint_states (using 0.0 as default)')
                # Use 0.0 as default for missing joints (fine for fixed joints)

    def right_target_pose_callback(self, msg: PoseStamped):
        """Callback for right arm target pose."""
        if not self.setup_complete_:
            self.get_logger().warn('IK solver not ready. Ignoring right target pose.')
            return

        # Validate pose input
        if not (math.isfinite(msg.pose.position.x) and math.isfinite(msg.pose.position.y) and
                math.isfinite(msg.pose.position.z) and math.isfinite(msg.pose.orientation.w) and
                math.isfinite(msg.pose.orientation.x) and math.isfinite(msg.pose.orientation.y) and
                math.isfinite(msg.pose.orientation.z)):
            self.get_logger().error('Received invalid (non-finite) right target pose. Ignoring.')
            return

        self.right_target_pose_ = msg
        self.get_logger().info(
            f'📥 Right arm target pose received: '
            f'pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})')

        # Try to solve IK if both poses are available
        self.solve_bimanual_ik()

    def left_target_pose_callback(self, msg: PoseStamped):
        """Callback for left arm target pose."""
        if not self.setup_complete_:
            self.get_logger().warn('IK solver not ready. Ignoring left target pose.')
            return

        # Validate pose input
        if not (math.isfinite(msg.pose.position.x) and math.isfinite(msg.pose.position.y) and
                math.isfinite(msg.pose.position.z) and math.isfinite(msg.pose.orientation.w) and
                math.isfinite(msg.pose.orientation.x) and math.isfinite(msg.pose.orientation.y) and
                math.isfinite(msg.pose.orientation.z)):
            self.get_logger().error('Received invalid (non-finite) left target pose. Ignoring.')
            return

        self.left_target_pose_ = msg
        self.get_logger().info(
            f'📥 Left arm target pose received: '
            f'pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})')

        # Try to solve IK if both poses are available
        self.solve_bimanual_ik()

    def solve_bimanual_ik(self):
        """Solve inverse kinematics for both arms simultaneously."""
        if self.robot is None:
            return

        # Check if both target poses are available
        if self.right_target_pose_ is None or self.left_target_pose_ is None:
            return

        if self.current_joint_positions_ is None:
            self.get_logger().warn('No current joint positions available. Using zeros.')
            initial_cfg = np.zeros(self.robot.joints.num_actuated_joints)
        else:
            initial_cfg = self.current_joint_positions_

        try:
            # Convert target poses to PyROKI format (wxyz quaternion)
            # Right arm
            qx_r = self.right_target_pose_.pose.orientation.x
            qy_r = self.right_target_pose_.pose.orientation.y
            qz_r = self.right_target_pose_.pose.orientation.z
            qw_r = self.right_target_pose_.pose.orientation.w
            target_wxyz_right = np.array([qw_r, qx_r, qy_r, qz_r])
            target_position_right = np.array([
                self.right_target_pose_.pose.position.x,
                self.right_target_pose_.pose.position.y,
                self.right_target_pose_.pose.position.z
            ])

            # Left arm
            qx_l = self.left_target_pose_.pose.orientation.x
            qy_l = self.left_target_pose_.pose.orientation.y
            qz_l = self.left_target_pose_.pose.orientation.z
            qw_l = self.left_target_pose_.pose.orientation.w
            target_wxyz_left = np.array([qw_l, qx_l, qy_l, qz_l])
            target_position_left = np.array([
                self.left_target_pose_.pose.position.x,
                self.left_target_pose_.pose.position.y,
                self.left_target_pose_.pose.position.z
            ])

            # Solve IK for both arms simultaneously
            self.get_logger().info('🔍 Solving bimanual IK...')

            # Debug: Log target information
            left_link_idx = self.robot.links.names.index(self.left_end_effector_link_)
            right_link_idx = self.robot.links.names.index(self.right_end_effector_link_)
            self.get_logger().info(
                f'🔍 Target links: left={self.left_end_effector_link_} (idx {left_link_idx}), '
                f'right={self.right_end_effector_link_} (idx {right_link_idx})'
            )
            self.get_logger().info(
                f'🔍 Left target: pos={target_position_left}, wxyz={target_wxyz_left}'
            )
            self.get_logger().info(
                f'🔍 Right target: pos={target_position_right}, wxyz={target_wxyz_right}'
            )
            self.get_logger().info(
                f'🔍 Initial cfg shape: {initial_cfg.shape}, range: [{initial_cfg.min():.4f}, {initial_cfg.max():.4f}]'
            )

            solution = solve_ik_with_multiple_targets(
                robot=self.robot,
                target_link_names=[self.left_end_effector_link_, self.right_end_effector_link_],
                target_wxyzs=np.array([target_wxyz_left, target_wxyz_right]),
                target_positions=np.array([target_position_left, target_position_right]),
                initial_cfg=initial_cfg
            )

            if solution is None or len(solution) != self.robot.joints.num_actuated_joints:
                self.get_logger().error('❌ Bimanual IK failed to converge')
                return

            self.get_logger().info('✅ Bimanual IK solution computed successfully')

            # Debug: Log joint indices and solution info
            num_actuated = self.robot.joints.num_actuated_joints
            self.get_logger().info(f'🔍 Right arm joints: {list(zip(self.right_joint_names_, self.right_joint_indices_))}')
            self.get_logger().info(f'🔍 Left arm joints: {list(zip(self.left_joint_names_, self.left_joint_indices_))}')
            self.get_logger().info(f'🔍 Solution shape: {solution.shape}, num_actuated: {num_actuated}')
            self.get_logger().info(f'🔍 Solution range: [{solution.min():.4f}, {solution.max():.4f}]')

            # Debug: Check actual solution values at right arm indices
            self.get_logger().info(f'🔍 Solution values at right arm indices:')
            for name, idx in zip(self.right_joint_names_, self.right_joint_indices_):
                if idx < num_actuated:
                    self.get_logger().info(f'  solution[{idx}] = {solution[idx]:.4f} for {name}')

            # Check which right arm joints are actuated
            right_actuated = [idx < num_actuated for idx in self.right_joint_indices_]
            left_actuated = [idx < num_actuated for idx in self.left_joint_indices_]
            self.get_logger().info(f'🔍 Right arm actuated flags: {right_actuated}')
            self.get_logger().info(f'🔍 Left arm actuated flags: {left_actuated}')

            # Debug: Check PyROKI's joint ordering
            actuated_joint_names = list(self.robot.joints.names[:num_actuated])
            self.get_logger().info(f'🔍 PyROKI actuated joint names (first 25): {actuated_joint_names}')
            # Check where right arm joints are in the actuated list
            for name in self.right_joint_names_:
                if name in actuated_joint_names:
                    actual_idx = actuated_joint_names.index(name)
                    self.get_logger().info(f'  {name} is at index {actual_idx} in actuated list (we think it\'s at {self.right_joint_indices_[self.right_joint_names_.index(name)]})')

            # Extract arm-specific joints from solution
            # Note: solution only contains num_actuated_joints, so we need to handle joints beyond that
            num_actuated = self.robot.joints.num_actuated_joints

            # Extract right arm solution
            right_arm_solution = []
            for i, idx in enumerate(self.right_joint_indices_):
                joint_name = self.right_joint_names_[i]
                if idx < num_actuated:
                    # Joint is in actuated range - get value from IK solution
                    val = solution[idx]
                    right_arm_solution.append(val)
                    self.get_logger().debug(f'  Right {joint_name} (idx {idx}): {val:.4f} from IK solution')
                else:
                    # Joint beyond actuated range - MUST use current position from joint_states
                    # PyROKI doesn't solve these joints, so we keep their current positions
                    if joint_name in self.current_joint_positions_dict_:
                        val = self.current_joint_positions_dict_[joint_name]
                        right_arm_solution.append(val)
                        self.get_logger().debug(f'  Right {joint_name} (idx {idx}): {val:.4f} from joint_states')
                    else:
                        # This should not happen - joint_states should have all arm joints
                        available_arm_r = [name for name in self.current_joint_positions_dict_.keys() if name.startswith('arm_r_')]
                        self.get_logger().error(
                            f'❌ Right arm joint {joint_name} (idx {idx}) NOT in joint_states! '
                            f'Available arm_r_*: {available_arm_r}. Using 0.0')
                        right_arm_solution.append(0.0)

            # Extract left arm solution
            left_arm_solution = []
            for i, idx in enumerate(self.left_joint_indices_):
                joint_name = self.left_joint_names_[i]
                if idx < num_actuated:
                    val = solution[idx]
                    left_arm_solution.append(val)
                else:
                    # Joint beyond actuated range - use current position from joint_states if available
                    if joint_name in self.current_joint_positions_dict_:
                        val = self.current_joint_positions_dict_[joint_name]
                        left_arm_solution.append(val)
                    else:
                        # This should not happen - joint_states should have all arm joints
                        available_arm_l = [name for name in self.current_joint_positions_dict_.keys() if name.startswith('arm_l_')]
                        self.get_logger().error(
                            f'❌ Left arm joint {joint_name} (idx {idx}) NOT in joint_states! '
                            f'Available arm_l_*: {available_arm_l}. Using 0.0')
                        left_arm_solution.append(0.0)

            right_arm_solution = np.array(right_arm_solution)
            left_arm_solution = np.array(left_arm_solution)

            # Debug: Log what values we got for right arm
            self.get_logger().info(f'🔍 Right arm solution breakdown:')
            for i, (name, idx) in enumerate(zip(self.right_joint_names_, self.right_joint_indices_)):
                source = 'IK solution' if idx < num_actuated else 'joint_states'
                val_in_dict = self.current_joint_positions_dict_.get(name, 'N/A')
                self.get_logger().info(
                    f'  {name} (idx {idx}): {right_arm_solution[i]:.4f} from {source} '
                    f'(dict has: {val_in_dict})')

            # Publish solutions for both arms
            self.publish_joint_trajectory(right_arm_solution, self.right_joint_names_, self.right_joint_solution_pub_, 'right')
            print(f'Right arm solution: {right_arm_solution}')
            self.publish_joint_trajectory(left_arm_solution, self.left_joint_names_, self.left_joint_solution_pub_, 'left')
            print(f'Left arm solution: {left_arm_solution}')
            print('--------------------------------')

        except Exception as e:
            import traceback
            self.get_logger().error(f'❌ Exception during bimanual IK solving: {e}')
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')

    def publish_joint_trajectory(self, solution: np.ndarray, joint_names: list[str], publisher, arm: str):
        """Publish joint trajectory message with arm-specific joints."""
        joint_trajectory = JointTrajectory()
        joint_trajectory.header.frame_id = 'base_link'
        # Don't set header.stamp - let it default to zero to avoid timing issues
        # The controller will interpret time_from_start relative to when it receives the message
        joint_trajectory.joint_names = joint_names

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = solution.tolist()
        point.velocities = [0.0] * len(joint_names)
        point.accelerations = [0.0] * len(joint_names)
        point.time_from_start = rclpy.duration.Duration(seconds=0.0).to_msg()

        joint_trajectory.points = [point]
        publisher.publish(joint_trajectory)

        self.get_logger().info(
            f'📤 {arm} arm IK solution published: '
            f'{len(joint_names)} joints ({", ".join(joint_names)})')


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    node = BimanualIKSolver()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

