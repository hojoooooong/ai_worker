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

#ifndef FFW_JOINT_STATE_BROADCASTER__FFW_JOINT_STATE_BROADCASTER_HPP_
#define FFW_JOINT_STATE_BROADCASTER__FFW_JOINT_STATE_BROADCASTER_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "ffw_joint_state_broadcaster/visibility_control.h"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace ffw_joint_state_broadcaster
{

/**
 * \brief FFW Joint State Broadcaster publishes state interfaces from ros2_control
 * as separate ROS messages for configurable joint groups.
 *
 * This broadcaster publishes joint_states topics configured via parameters.
 * Each group can specify:
 * - topic_name: The ROS topic to publish to (e.g., "/leader/arm_left/joint_states")
 * - joints: List of joint names to include in this topic
 *
 * Example parameter configuration:
 * joint_groups:
 *   arm_left:
 *     topic_name: "/leader/arm_left/joint_states"
 *     joints: ["arm_l_joint1", "arm_l_joint2", ...]
 *   arm_right:
 *     topic_name: "/leader/arm_right/joint_states"
 *     joints: ["arm_r_joint1", "arm_r_joint2", ...]
 *
 * The timestamp in the header is set to the actual time when the sensor data
 * was read from the hardware interface.
 */
class FFWJointStateBroadcaster : public controller_interface::ControllerInterface
{
public:
  FFW_JOINT_STATE_BROADCASTER_PUBLIC
  FFWJointStateBroadcaster();

  FFW_JOINT_STATE_BROADCASTER_PUBLIC
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;

  FFW_JOINT_STATE_BROADCASTER_PUBLIC
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  FFW_JOINT_STATE_BROADCASTER_PUBLIC
  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  FFW_JOINT_STATE_BROADCASTER_PUBLIC
  controller_interface::CallbackReturn on_init() override;

  FFW_JOINT_STATE_BROADCASTER_PUBLIC
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  FFW_JOINT_STATE_BROADCASTER_PUBLIC
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  FFW_JOINT_STATE_BROADCASTER_PUBLIC
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

protected:
  bool init_joint_data();
  void publish_joint_states(const rclcpp::Time & time);

protected:
  // Joint group configuration structure
  struct JointGroupConfig
  {
    std::string topic_name;
    std::vector<std::string> joints;
  };

  // Joint groups configuration (key: group name, value: config)
  std::unordered_map<std::string, JointGroupConfig> joint_groups_;

  // Joint name to group mapping
  std::unordered_map<std::string, std::string> joint_to_group_;

  // Publishers for each group (key: group name)
  std::unordered_map<std::string,
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::JointState>>> publishers_;

  // Realtime publishers (key: group name)
  std::unordered_map<std::string,
    std::shared_ptr<realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>>>
    rt_publishers_;

  // Joint data storage
  std::unordered_map<std::string, std::unordered_map<std::string, double>> joint_data_;

  // Interface type mapping
  std::unordered_map<std::string, std::string> map_interface_to_joint_state_;
};

}  // namespace ffw_joint_state_broadcaster

#endif  // FFW_JOINT_STATE_BROADCASTER__FFW_JOINT_STATE_BROADCASTER_HPP_
