// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ffw_robot_state_publisher/ffw_robot_state_publisher.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "kdl/jntarray.hpp"
#include "kdl_parser/kdl_parser.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace ffw_robot_state_publisher
{

FFWRobotStatePublisher::FFWRobotStatePublisher(const rclcpp::NodeOptions & options)
: robot_state_publisher::RobotStatePublisher(options)
{
  // Declare parameters for left arm
  left_base_link_ = this->declare_parameter("left_arm.base_link", "base_link");
  left_end_effector_link_ = this->declare_parameter(
    "left_arm.end_effector_link", "end_effector_l_link");

  // Declare parameters for right arm
  right_base_link_ = this->declare_parameter("right_arm.base_link", "base_link");
  right_end_effector_link_ = this->declare_parameter(
    "right_arm.end_effector_link", "end_effector_r_link");

  // Create publishers for end effector poses
  left_eef_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
    "end_effector_pose/left", rclcpp::SensorDataQoS());

  right_eef_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
    "end_effector_pose/right", rclcpp::SensorDataQoS());

  // Setup KDL chains (reuses robot_description already loaded by parent)
  setupKDLChains();

  // Subscribe to joint_states for FK computation
  // Note: Parent also subscribes for TF publishing - this is intentional as
  // parent's callbackJointState is not virtual and cannot be overridden
  joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    "joint_states",
    rclcpp::SensorDataQoS(),
    std::bind(&FFWRobotStatePublisher::jointStateCallback, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "FFW Robot State Publisher initialized");
  RCLCPP_INFO(
    get_logger(), "Left arm: %s -> %s",
    left_base_link_.c_str(), left_end_effector_link_.c_str());
  RCLCPP_INFO(
    get_logger(), "Right arm: %s -> %s",
    right_base_link_.c_str(), right_end_effector_link_.c_str());
}

void FFWRobotStatePublisher::setupKDLChains()
{
  // Get robot_description parameter (already declared by parent)
  std::string urdf_xml = this->get_parameter("robot_description").as_string();

  if (urdf_xml.empty()) {
    RCLCPP_ERROR(get_logger(), "robot_description is empty");
    return;
  }

  // Parse URDF to KDL tree
  // Note: Parent parses URDF but doesn't expose KDL tree, so we parse again
  if (!kdl_parser::treeFromString(urdf_xml, kdl_tree_)) {
    RCLCPP_ERROR(get_logger(), "Failed to parse URDF to KDL tree");
    return;
  }

  // Extract left arm chain
  if (kdl_tree_.getChain(left_base_link_, left_end_effector_link_, left_arm_chain_)) {
    extractJointNames(left_arm_chain_, left_arm_joint_names_);
    left_arm_fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(left_arm_chain_);
    RCLCPP_INFO(
      get_logger(), "Left arm chain: %zu joints", left_arm_joint_names_.size());
  } else {
    RCLCPP_WARN(
      get_logger(), "Failed to get left arm chain: %s -> %s",
      left_base_link_.c_str(), left_end_effector_link_.c_str());
  }

  // Extract right arm chain
  if (kdl_tree_.getChain(right_base_link_, right_end_effector_link_, right_arm_chain_)) {
    extractJointNames(right_arm_chain_, right_arm_joint_names_);
    right_arm_fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(right_arm_chain_);
    RCLCPP_INFO(
      get_logger(), "Right arm chain: %zu joints", right_arm_joint_names_.size());
  } else {
    RCLCPP_WARN(
      get_logger(), "Failed to get right arm chain: %s -> %s",
      right_base_link_.c_str(), right_end_effector_link_.c_str());
  }

  chains_initialized_ = (left_arm_fk_solver_ != nullptr) || (right_arm_fk_solver_ != nullptr);
}

void FFWRobotStatePublisher::extractJointNames(
  const KDL::Chain & chain,
  std::vector<std::string> & joint_names)
{
  joint_names.clear();
  for (unsigned int i = 0; i < chain.getNrOfSegments(); ++i) {
    const KDL::Segment & segment = chain.getSegment(i);
    if (segment.getJoint().getType() != KDL::Joint::None) {
      joint_names.push_back(segment.getJoint().getName());
    }
  }
}

void FFWRobotStatePublisher::jointStateCallback(
  const sensor_msgs::msg::JointState::ConstSharedPtr state)
{
  if (!chains_initialized_ || state->name.size() != state->position.size()) {
    return;
  }

  // Build joint position map
  std::map<std::string, double> joint_positions;
  for (size_t i = 0; i < state->name.size(); ++i) {
    joint_positions[state->name[i]] = state->position[i];
  }

  // Publish end effector poses
  if (left_arm_fk_solver_) {
    publishEndEffectorPose(
      left_arm_chain_, left_arm_joint_names_, left_arm_fk_solver_,
      joint_positions, state->header, left_base_link_, left_eef_pose_pub_);
  }

  if (right_arm_fk_solver_) {
    publishEndEffectorPose(
      right_arm_chain_, right_arm_joint_names_, right_arm_fk_solver_,
      joint_positions, state->header, right_base_link_, right_eef_pose_pub_);
  }
}

void FFWRobotStatePublisher::publishEndEffectorPose(
  const KDL::Chain & chain,
  const std::vector<std::string> & joint_names,
  const std::unique_ptr<KDL::ChainFkSolverPos_recursive> & fk_solver,
  const std::map<std::string, double> & joint_positions,
  const std_msgs::msg::Header & header,
  const std::string & frame_id,
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr & publisher)
{
  KDL::JntArray jnt_pos(chain.getNrOfJoints());

  unsigned int jnt_idx = 0;
  for (const auto & joint_name : joint_names) {
    auto it = joint_positions.find(joint_name);
    jnt_pos(jnt_idx) = (it != joint_positions.end()) ? it->second : 0.0;
    ++jnt_idx;
  }

  KDL::Frame end_effector_frame;
  if (fk_solver->JntToCart(jnt_pos, end_effector_frame) < 0) {
    return;
  }

  geometry_msgs::msg::PoseStamped pose_msg;
  pose_msg.header = header;
  pose_msg.header.frame_id = frame_id;

  pose_msg.pose.position.x = end_effector_frame.p.x();
  pose_msg.pose.position.y = end_effector_frame.p.y();
  pose_msg.pose.position.z = end_effector_frame.p.z();

  double x, y, z, w;
  end_effector_frame.M.GetQuaternion(x, y, z, w);
  pose_msg.pose.orientation.x = x;
  pose_msg.pose.orientation.y = y;
  pose_msg.pose.orientation.z = z;
  pose_msg.pose.orientation.w = w;

  publisher->publish(pose_msg);
}

}  // namespace ffw_robot_state_publisher

RCLCPP_COMPONENTS_REGISTER_NODE(ffw_robot_state_publisher::FFWRobotStatePublisher)
