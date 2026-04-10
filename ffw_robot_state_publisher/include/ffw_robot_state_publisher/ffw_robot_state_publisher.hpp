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

#ifndef FFW_ROBOT_STATE_PUBLISHER__FFW_ROBOT_STATE_PUBLISHER_HPP_
#define FFW_ROBOT_STATE_PUBLISHER__FFW_ROBOT_STATE_PUBLISHER_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "kdl/chain.hpp"
#include "kdl/chainfksolverpos_recursive.hpp"
#include "kdl/tree.hpp"
#include "rclcpp/rclcpp.hpp"
#include "robot_state_publisher/robot_state_publisher.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace ffw_robot_state_publisher
{

/// FFW Robot State Publisher - extends RobotStatePublisher with end-effector pose publishing
/**
 * Inherits from robot_state_publisher::RobotStatePublisher and adds
 * forward kinematics computation using KDL to publish end-effector poses
 * for a dual-arm humanoid robot.
 *
 * Note: Parent's callbackJointState is not virtual, so we use a separate
 * subscription for FK computation. Both subscriptions (parent's TF publishing
 * and ours for FK) will process joint_states messages.
 */
class FFWRobotStatePublisher : public robot_state_publisher::RobotStatePublisher
{
public:
  explicit FFWRobotStatePublisher(const rclcpp::NodeOptions & options);
  ~FFWRobotStatePublisher() override = default;

private:
  /// Setup KDL chains for forward kinematics
  void setupKDLChains();

  /// Extract joint names from a KDL chain
  void extractJointNames(
    const KDL::Chain & chain,
    std::vector<std::string> & joint_names);

  /// Callback for joint state messages (FK computation only)
  void jointStateCallback(const sensor_msgs::msg::JointState::ConstSharedPtr state);

  /// Compute FK and publish end-effector pose
  void publishEndEffectorPose(
    const KDL::Chain & chain,
    const std::vector<std::string> & joint_names,
    const std::unique_ptr<KDL::ChainFkSolverPos_recursive> & fk_solver,
    const std::map<std::string, double> & joint_positions,
    const std_msgs::msg::Header & header,
    const std::string & frame_id,
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr & publisher);

  // KDL data
  KDL::Tree kdl_tree_;
  KDL::Chain left_arm_chain_;
  KDL::Chain right_arm_chain_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> left_arm_fk_solver_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> right_arm_fk_solver_;
  std::vector<std::string> left_arm_joint_names_;
  std::vector<std::string> right_arm_joint_names_;

  // Parameters
  std::string left_base_link_;
  std::string left_end_effector_link_;
  std::string right_base_link_;
  std::string right_end_effector_link_;

  // ROS interfaces
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr left_eef_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr right_eef_pose_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

  bool chains_initialized_{false};
};

}  // namespace ffw_robot_state_publisher

#endif  // FFW_ROBOT_STATE_PUBLISHER__FFW_ROBOT_STATE_PUBLISHER_HPP_
