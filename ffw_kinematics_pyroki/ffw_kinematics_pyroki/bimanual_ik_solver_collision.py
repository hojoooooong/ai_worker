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
import contextlib
import os
from typing import Optional, Sequence

import numpy as np
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
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls

from ffw_kinematics_pyroki.examples.pyroki_snippets import solve_ik_with_multiple_targets


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
        self.robot_collision: Optional[pk.collision.RobotCollision] = None
        self.world_collision_objects: list[pk.collision.CollGeom] = []

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

            # Initialize collision information
            self.robot_collision = pk.collision.RobotCollision.from_robot(self.robot)
            # Add some basic world collision objects (can be extended)
            self.world_collision_objects = []

            self.get_logger().info(f'✅ PyROKI robot loaded successfully!')
            self.get_logger().info(f'   Robot name: {urdf.robot.name}')
            self.get_logger().info(f'   Number of actuated joints: {self.robot.joints.num_actuated_joints}')
            self.get_logger().info(f'   Collision information initialized')

            # Verify target links exist
            if self.left_end_effector_link_ not in self.robot.links.names:
                self.get_logger().error(f"Error: Target link '{self.left_end_effector_link_}' not found!")
                return
            if self.right_end_effector_link_ not in self.robot.links.names:
                self.get_logger().error(f"Error: Target link '{self.right_end_effector_link_}' not found!")
                return

            # Store actuated joint names from robot
            # Use robot.joints.actuated_names if available, otherwise use first num_actuated
            if hasattr(self.robot.joints, 'actuated_names'):
                self.joint_names_ = list(self.robot.joints.actuated_names)
            else:
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
                self.get_logger().info(f'📋 Left: {joint_name} at index {idx}')
                left_joint_pairs.append((joint_name, idx))
            else:
                self.get_logger().warn(f'⚠️ Left: {joint_name} not found in robot joints!')

        # Extract right arm joints (arm_r_joint1-7) with their indices
        right_joint_pairs = []
        for i in range(1, 8):  # joint1 to joint7
            joint_name = f'arm_r_joint{i}'
            if joint_name in all_joint_names:
                idx = all_joint_names.index(joint_name)
                self.get_logger().info(f'📋 Right: {joint_name} at index {idx}')
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

        # Extract current joint positions for actuated joints (for IK initial guess)
        # IMPORTANT: We need to map from robot.joints.names indices to actuated indices
        num_actuated = self.robot.joints.num_actuated_joints
        self.current_joint_positions_ = np.zeros(num_actuated)

        # Create mapping from robot.joints.names index to actuated index
        all_joint_names = list(self.robot.joints.names)
        actuated_indices = []  # Indices in robot.joints.names for actuated joints
        for i in range(num_actuated):
            actuated_indices.append(i)

        # Now fill initial_cfg using the actual indices in robot.joints.names
        for actuated_idx, joint_name in enumerate(self.joint_names_):
            # Find this joint in robot.joints.names to get its actual index
            if joint_name in all_joint_names:
                robot_idx = all_joint_names.index(joint_name)
                # Now get value from joint_states using the joint name
                if joint_name in msg.name:
                    msg_idx = msg.name.index(joint_name)
                    self.current_joint_positions_[actuated_idx] = msg.position[msg_idx]
                else:
                    # Only warn once per missing joint
                    if joint_name not in self.warned_missing_joints_:
                        self.warned_missing_joints_.add(joint_name)
                        self.get_logger().debug(
                            f'Joint {joint_name} not found in joint_states (using 0.0 as default)')
            else:
                self.get_logger().warn(f'Joint {joint_name} not found in robot.joints.names!')

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

            # Debug: Log initial_cfg for right arm joints
            self.get_logger().info('🔍 Initial cfg for right arm joints:')
            for name, robot_idx in zip(self.right_joint_names_, self.right_joint_indices_):
                # Find this joint in self.joint_names_ to get actuated index
                if name in self.joint_names_:
                    actuated_idx = self.joint_names_.index(name)
                    if actuated_idx < len(initial_cfg):
                        self.get_logger().info(f'  {name} (robot_idx {robot_idx}, actuated_idx {actuated_idx}): {initial_cfg[actuated_idx]:.4f}')
                    else:
                        self.get_logger().warn(f'  {name} actuated_idx {actuated_idx} >= len(initial_cfg)={len(initial_cfg)}')
                else:
                    self.get_logger().warn(f'  {name} not in actuated joints!')

            solution = solve_ik_with_multiple_targets_and_collision(
                robot=self.robot,
                robot_collision=self.robot_collision,
                world_collision_objects=self.world_collision_objects,
                target_link_names=[self.left_end_effector_link_, self.right_end_effector_link_],
                target_positions=np.array([target_position_left, target_position_right]),
                target_wxyzs=np.array([target_wxyz_left, target_wxyz_right])
            )

            if solution is None or len(solution) != self.robot.joints.num_actuated_joints:
                self.get_logger().error('❌ Bimanual IK failed to converge')
                return

            self.get_logger().info('✅ Bimanual IK solution computed successfully')

            # Debug: Log solution for right arm joints
            self.get_logger().info('🔍 Solution for right arm joints:')
            for name, robot_idx in zip(self.right_joint_names_, self.right_joint_indices_):
                # Find this joint in self.joint_names_ to get actuated index
                if name in self.joint_names_:
                    actuated_idx = self.joint_names_.index(name)
                    if actuated_idx < len(solution):
                        self.get_logger().info(f'  {name} (robot_idx {robot_idx}, actuated_idx {actuated_idx}): {solution[actuated_idx]:.4f}')
                    else:
                        val = self.current_joint_positions_dict_.get(name, 0.0)
                        self.get_logger().info(f'  {name} (robot_idx {robot_idx}, not in solution): {val:.4f} from joint_states')
                else:
                    val = self.current_joint_positions_dict_.get(name, 0.0)
                    self.get_logger().info(f'  {name} (robot_idx {robot_idx}, not actuated): {val:.4f} from joint_states')

            # Extract arm-specific joints from solution
            # Extract right arm solution
            right_arm_solution = []
            for i, robot_idx in enumerate(self.right_joint_indices_):
                joint_name = self.right_joint_names_[i]
                # Check if this joint is in actuated list
                if joint_name in self.joint_names_:
                    actuated_idx = self.joint_names_.index(joint_name)
                    # Joint is actuated - get value from IK solution
                    if actuated_idx < len(solution):
                        val = solution[actuated_idx]
                        right_arm_solution.append(val)
                    else:
                        self.get_logger().error(f'Right arm joint {joint_name} actuated_idx {actuated_idx} >= solution length {len(solution)}')
                        right_arm_solution.append(0.0)
                else:
                    # Joint beyond actuated range - use current position from joint_states
                    if joint_name in self.current_joint_positions_dict_:
                        val = self.current_joint_positions_dict_[joint_name]
                        right_arm_solution.append(val)
                    else:
                        self.get_logger().warn(
                            f'Right arm joint {joint_name} (robot_idx {robot_idx}) not in joint_states. Using 0.0')
                        right_arm_solution.append(0.0)

            # Extract left arm solution
            left_arm_solution = []
            for i, robot_idx in enumerate(self.left_joint_indices_):
                joint_name = self.left_joint_names_[i]
                # Check if this joint is in actuated list
                if joint_name in self.joint_names_:
                    actuated_idx = self.joint_names_.index(joint_name)
                    if actuated_idx < len(solution):
                        val = solution[actuated_idx]
                        left_arm_solution.append(val)
                    else:
                        self.get_logger().error(f'Left arm joint {joint_name} actuated_idx {actuated_idx} >= solution length {len(solution)}')
                        left_arm_solution.append(0.0)
                else:
                    # Joint beyond actuated range - use current position from joint_states
                    if joint_name in self.current_joint_positions_dict_:
                        val = self.current_joint_positions_dict_[joint_name]
                        left_arm_solution.append(val)
                    else:
                        self.get_logger().warn(
                            f'Left arm joint {joint_name} (robot_idx {robot_idx}) not in joint_states. Using 0.0')
                        left_arm_solution.append(0.0)

            right_arm_solution = np.array(right_arm_solution)
            left_arm_solution = np.array(left_arm_solution)

            # Debug: Log final right arm solution
            self.get_logger().info('🔍 Final right arm solution:')
            for name, val in zip(self.right_joint_names_, right_arm_solution):
                self.get_logger().info(f'  {name}: {val:.4f}')

            # Publish solutions for both arms
            self.publish_joint_trajectory(right_arm_solution, self.right_joint_names_, self.right_joint_solution_pub_, 'right')
            self.publish_joint_trajectory(left_arm_solution, self.left_joint_names_, self.left_joint_solution_pub_, 'left')

        except Exception as e:
            import traceback
            self.get_logger().error(f'❌ Exception during bimanual IK solving: {e}')
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')

    def publish_joint_trajectory(self, solution: np.ndarray, joint_names: list[str], publisher, arm: str):
        """Publish joint trajectory message with arm-specific joints."""
        joint_trajectory = JointTrajectory()
        joint_trajectory.header.frame_id = 'base_link'
        # Don't set header.stamp - let it default to zero to avoid timing issues
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


def solve_ik_with_multiple_targets_and_collision(
    robot: pk.Robot,
    robot_collision: pk.collision.RobotCollision,
    world_collision_objects: Sequence[pk.collision.CollGeom],
    target_link_names: Sequence[str],
    target_positions: np.ndarray,
    target_wxyzs: np.ndarray,
) -> np.ndarray:
    """
    Solves the basic IK problem for multiple targets with collision avoidance.

    Args:
        robot: PyRoKi Robot.
        robot_collision: Robot collision information.
        world_collision_objects: List of world collision objects.
        target_link_names: Sequence[str]. List of link names to be controlled.
        target_positions: np.ndarray. Shape: (num_targets, 3). Target positions.
        target_wxyzs: np.ndarray. Shape: (num_targets, 4). Target orientations.

    Returns:
        cfg: np.ndarray. Shape: (robot.joints.num_actuated_joints,).
    """
    num_targets = len(target_link_names)
    assert target_positions.shape == (num_targets, 3)
    assert target_wxyzs.shape == (num_targets, 4)
    target_link_indices = [robot.links.names.index(name) for name in target_link_names]

    cfg = _solve_ik_with_multiple_targets_and_collision_jax(
        robot,
        robot_collision,
        world_collision_objects,
        jnp.array(target_positions),
        jnp.array(target_wxyzs),
        jnp.array(target_link_indices),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    return np.array(cfg)


@jdc.jit
def _solve_ik_with_multiple_targets_and_collision_jax(
    robot: pk.Robot,
    robot_collision: pk.collision.RobotCollision,
    world_collision_objects: Sequence[pk.collision.CollGeom],
    target_positions: jax.Array,
    target_wxyzs: jax.Array,
    target_link_indices: jax.Array,
) -> jax.Array:
    """Solves the multi-target IK problem with collision avoidance. Returns joint configuration."""
    JointVar = robot.joint_var_cls
    joint_var = JointVar(0)
    vars = [joint_var]

    # Create target poses
    target_poses = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyzs), target_positions
    )
    batch_axes = target_poses.get_batch_axes()

    # Weights and margins defined directly in factors
    costs = [
        pk.costs.pose_cost_analytic_jac(
            jax.tree.map(lambda x: x[None], robot),
            JointVar(jnp.full(batch_axes, 0)),
            target_poses,
            target_link_indices,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var=joint_var,
            weight=100.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=jnp.array(joint_var.default_factory()),
            weight=0.01,
        ),
        pk.costs.self_collision_cost(
            robot,
            robot_coll=robot_collision,
            joint_var=joint_var,
            margin=0.02,
            weight=5.0,
        ),
    ]

    # Add world collision costs
    costs.extend(
        [
            pk.costs.world_collision_cost(
                robot, robot_collision, joint_var, world_coll, 0.05, 10.0
            )
            for world_coll in world_collision_objects
        ]
    )

    sol = (
        jaxls.LeastSquaresProblem(costs, vars)
        .analyze()
        .solve(verbose=False, linear_solver="dense_cholesky")
    )
    return sol[joint_var]


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    node = BimanualIKSolver()
    rclpy.spin(node)
    rclpy.shutdown()
