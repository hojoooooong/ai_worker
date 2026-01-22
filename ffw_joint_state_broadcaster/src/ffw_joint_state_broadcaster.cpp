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

#include "ffw_joint_state_broadcaster/ffw_joint_state_broadcaster.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/time.hpp"

namespace ffw_joint_state_broadcaster
{

using hardware_interface::HW_IF_EFFORT;
using hardware_interface::HW_IF_POSITION;
using hardware_interface::HW_IF_VELOCITY;

FFWJointStateBroadcaster::FFWJointStateBroadcaster() {}

controller_interface::InterfaceConfiguration
FFWJointStateBroadcaster::command_interface_configuration() const
{
  return controller_interface::InterfaceConfiguration{
    controller_interface::interface_configuration_type::NONE};
}

controller_interface::InterfaceConfiguration
FFWJointStateBroadcaster::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration state_interfaces_config;
  state_interfaces_config.type = controller_interface::interface_configuration_type::ALL;
  return state_interfaces_config;
}

controller_interface::CallbackReturn FFWJointStateBroadcaster::on_init()
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FFWJointStateBroadcaster::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  joint_groups_.clear();
  joint_to_group_.clear();
  publishers_.clear();
  rt_publishers_.clear();

  // Get joint groups from parameters
  // Expected parameter structure in YAML:
  // joint_group_names: ["arm_left", "arm_right", ...]  # List of group names
  // joint_groups:
  //   arm_left:
  //     topic_name: "/leader/arm_left/joint_states"
  //     joints: ["arm_l_joint1", "arm_l_joint2", ...]
  //   arm_right:
  //     topic_name: "/leader/arm_right/joint_states"
  //     joints: ["arm_r_joint1", "arm_r_joint2", ...]

  // First, get the list of group names
  std::vector<std::string> group_names;

  // Try to get joint_group_names parameter first
  // Declare it first if it doesn't exist
  if (!get_node()->has_parameter("joint_group_names")) {
    get_node()->declare_parameter("joint_group_names", std::vector<std::string>());
  }

  try {
    group_names = get_node()->get_parameter("joint_group_names").as_string_array();
    RCLCPP_INFO(
      get_node()->get_logger(),
      "Found 'joint_group_names' parameter with %zu groups", group_names.size());
  } catch (const std::exception & e) {
    RCLCPP_WARN(
      get_node()->get_logger(),
      "Failed to get 'joint_group_names' parameter: %s. "
      "Trying to discover groups from nested parameters...", e.what());
  }

  // If joint_group_names not found or empty, try to discover from nested parameters
  if (group_names.empty()) {
    RCLCPP_INFO(
      get_node()->get_logger(),
      "Attempting to discover joint groups from nested parameters...");

    // Try with different depth values (0 = current level, 1 = one level deep, etc.)
    for (int depth = 1; depth <= 2; ++depth) {
      try {
        rcl_interfaces::msg::ListParametersResult param_list =
          get_node()->list_parameters({"joint_groups"}, depth);

        RCLCPP_DEBUG(
          get_node()->get_logger(),
          "list_parameters with depth %d returned %zu parameters", depth,
          param_list.names.size());

        if (!param_list.names.empty()) {
          // Log all found parameters for debugging
          for (const auto & param_name : param_list.names) {
            RCLCPP_DEBUG(get_node()->get_logger(), "Found parameter: %s", param_name.c_str());
          }
        }

        std::set<std::string> discovered_groups;
        for (const auto & param_name : param_list.names) {
          // Format: joint_groups.<group_name>.topic_name or joint_groups.<group_name>.joints
          size_t first_dot = param_name.find('.');
          size_t second_dot = param_name.find('.', first_dot + 1);
          if (first_dot == std::string::npos || second_dot == std::string::npos) {
            continue;
          }
          std::string group_name = param_name.substr(first_dot + 1, second_dot - first_dot - 1);
          discovered_groups.insert(group_name);
        }

        if (!discovered_groups.empty()) {
          group_names.assign(discovered_groups.begin(), discovered_groups.end());
          RCLCPP_INFO(
            get_node()->get_logger(),
            "Discovered %zu joint groups from nested parameters", group_names.size());
          break;
        }
      } catch (const std::exception & e) {
        RCLCPP_DEBUG(
          get_node()->get_logger(),
          "list_parameters with depth %d failed: %s", depth, e.what());
      }
    }
  }

  if (group_names.empty()) {
    RCLCPP_WARN(
      get_node()->get_logger(),
      "No joint groups found. No joint state topics will be published.");
    return controller_interface::CallbackReturn::SUCCESS;
  }

  // Read configuration for each group
  for (const auto & group_name : group_names) {
    std::string topic_param = "joint_groups." + group_name + ".topic_name";
    std::string joints_param = "joint_groups." + group_name + ".joints";

    JointGroupConfig config;

    // Declare and get topic_name parameter
    if (!get_node()->has_parameter(topic_param)) {
      get_node()->declare_parameter(topic_param, "");
    }
    try {
      config.topic_name = get_node()->get_parameter(topic_param).as_string();
    } catch (const std::exception & e) {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "Failed to get 'topic_name' for group '%s': %s. Skipping.",
        group_name.c_str(), e.what());
      continue;
    }

    // Declare and get joints parameter
    if (!get_node()->has_parameter(joints_param)) {
      get_node()->declare_parameter(joints_param, std::vector<std::string>());
    }
    try {
      config.joints = get_node()->get_parameter(joints_param).as_string_array();
    } catch (const std::exception & e) {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "Failed to get 'joints' for group '%s': %s. Skipping.",
        group_name.c_str(), e.what());
      continue;
    }

    if (config.topic_name.empty() || config.joints.empty()) {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "Group '%s' has empty topic_name or joints. Skipping.", group_name.c_str());
      continue;
    }

    joint_groups_[group_name] = config;
  }

  // Build joint to group mapping and create publishers
  for (const auto & [group_name, config] : joint_groups_) {
    if (config.topic_name.empty()) {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "Group '%s' has no topic_name specified. Skipping.", group_name.c_str());
      continue;
    }

    if (config.joints.empty()) {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "Group '%s' has no joints specified. Skipping.", group_name.c_str());
      continue;
    }

    // Build joint to group mapping
    for (const auto & joint : config.joints) {
      joint_to_group_[joint] = group_name;
    }

    // Create publisher
    publishers_[group_name] = get_node()->create_publisher<sensor_msgs::msg::JointState>(
      config.topic_name, rclcpp::SystemDefaultsQoS());

    // Create realtime publisher
    rt_publishers_[group_name] =
      std::make_shared<realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>>(
      publishers_[group_name]);

    RCLCPP_INFO(
      get_node()->get_logger(),
      "Created joint state publisher for group '%s' on topic '%s' with %zu joints",
      group_name.c_str(), config.topic_name.c_str(), config.joints.size());
  }

  // Initialize interface mapping
  map_interface_to_joint_state_[HW_IF_POSITION] = "position";
  map_interface_to_joint_state_[HW_IF_VELOCITY] = "velocity";
  map_interface_to_joint_state_[HW_IF_EFFORT] = "effort";

  RCLCPP_INFO(
    get_node()->get_logger(),
    "FFW Joint State Broadcaster configured with %zu joint groups",
    joint_groups_.size());

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FFWJointStateBroadcaster::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  if (!init_joint_data()) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Failed to initialize joint data. Controller will not run.");
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FFWJointStateBroadcaster::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  joint_data_.clear();
  return controller_interface::CallbackReturn::SUCCESS;
}

bool FFWJointStateBroadcaster::init_joint_data()
{
  joint_data_.clear();

  if (state_interfaces_.empty()) {
    return false;
  }

  // Initialize joint data structure
  for (const auto & state_interface : state_interfaces_) {
    const std::string & joint_name = state_interface.get_prefix_name();
    const std::string & interface_name = state_interface.get_interface_name();

    // Map interface name if needed
    std::string mapped_name = interface_name;
    if (map_interface_to_joint_state_.count(interface_name) > 0) {
      mapped_name = map_interface_to_joint_state_[interface_name];
    }

    // For joystick sensor interfaces (e.g., "sensorxel_l_joy/JOYSTICK X VALUE"),
    // we need to create a synthetic joint name from the sensor name and interface
    // Format: sensorxel_l_joy -> sensorxel_l_joy_x, sensorxel_l_joy_y, etc.
    std::string data_key = joint_name;
    if (interface_name.find("JOYSTICK") != std::string::npos) {
      // This is a joystick sensor interface
      // Create a synthetic joint name: sensorxel_l_joy + "_" + simplified_interface_name
      std::string simplified_interface = interface_name;
      // Replace spaces and convert to lowercase for consistency
      std::replace(simplified_interface.begin(), simplified_interface.end(), ' ', '_');
      // Extract meaningful part: "JOYSTICK X VALUE" -> "x", "JOYSTICK Y VALUE" -> "y", "JOYSTICK TACT SWITCH" -> "tact"
      if (interface_name.find("X VALUE") != std::string::npos) {
        simplified_interface = "x";
      } else if (interface_name.find("Y VALUE") != std::string::npos) {
        simplified_interface = "y";
      } else if (interface_name.find("TACT SWITCH") != std::string::npos) {
        simplified_interface = "tact";
      } else {
        simplified_interface = interface_name;
        std::replace(simplified_interface.begin(), simplified_interface.end(), ' ', '_');
        std::transform(simplified_interface.begin(), simplified_interface.end(),
                      simplified_interface.begin(), ::tolower);
      }
      data_key = joint_name + "_" + simplified_interface;
      // For joystick interfaces, we'll store them as "position" in joint state
      mapped_name = "position";
    }

    // Initialize if not exists
    if (joint_data_.count(data_key) == 0) {
      joint_data_[data_key] = {};
    }

    joint_data_[data_key][mapped_name] = std::numeric_limits<double>::quiet_NaN();
  }

  return true;
}

void FFWJointStateBroadcaster::publish_joint_states(const rclcpp::Time & time)
{
  // Update joint data from state interfaces
  for (const auto & state_interface : state_interfaces_) {
    const std::string & joint_name = state_interface.get_prefix_name();
    const std::string & interface_name = state_interface.get_interface_name();

    // Map interface name if needed
    std::string mapped_name = interface_name;
    if (map_interface_to_joint_state_.count(interface_name) > 0) {
      mapped_name = map_interface_to_joint_state_[interface_name];
    }

    // For joystick sensor interfaces, create synthetic joint name
    std::string data_key = joint_name;
    if (interface_name.find("JOYSTICK") != std::string::npos) {
      // This is a joystick sensor interface
      std::string simplified_interface;
      if (interface_name.find("X VALUE") != std::string::npos) {
        simplified_interface = "x";
      } else if (interface_name.find("Y VALUE") != std::string::npos) {
        simplified_interface = "y";
      } else if (interface_name.find("TACT SWITCH") != std::string::npos) {
        simplified_interface = "tact";
      } else {
        simplified_interface = interface_name;
        std::replace(simplified_interface.begin(), simplified_interface.end(), ' ', '_');
        std::transform(simplified_interface.begin(), simplified_interface.end(),
                      simplified_interface.begin(), ::tolower);
      }
      data_key = joint_name + "_" + simplified_interface;
      mapped_name = "position";  // Store joystick values as position
    }

    // Get value
    auto value = state_interface.get_optional();
    if (value && joint_data_.count(data_key) > 0) {
      joint_data_[data_key][mapped_name] = *value;
    }
  }

  // Publish for each configured group
  for (const auto & [group_name, config] : joint_groups_) {
    if (rt_publishers_.count(group_name) == 0 || config.joints.empty()) {
      continue;
    }

    auto & rt_pub = rt_publishers_[group_name];
    if (!rt_pub) {
      continue;
    }

    if (rt_pub->trylock()) {
      auto & msg = rt_pub->msg_;

      // Set header with actual timestamp when sensor data was read
      msg.header.stamp = time;
      msg.header.frame_id = "";

      // Clear and resize message
      msg.name.clear();
      msg.position.clear();
      msg.velocity.clear();
      msg.effort.clear();

      // Fill message with joint data
      for (const auto & joint_name : config.joints) {
        if (joint_data_.count(joint_name) == 0) {
          continue;
        }

        const auto & joint_interfaces = joint_data_[joint_name];
        msg.name.push_back(joint_name);

        // Add position if available
        if (joint_interfaces.count("position") > 0 &&
          !std::isnan(joint_interfaces.at("position")))
        {
          msg.position.push_back(joint_interfaces.at("position"));
        } else {
          msg.position.push_back(std::numeric_limits<double>::quiet_NaN());
        }

        // Add velocity if available
        if (joint_interfaces.count("velocity") > 0 &&
          !std::isnan(joint_interfaces.at("velocity")))
        {
          msg.velocity.push_back(joint_interfaces.at("velocity"));
        } else {
          msg.velocity.push_back(std::numeric_limits<double>::quiet_NaN());
        }

        // Add effort if available
        if (joint_interfaces.count("effort") > 0 &&
          !std::isnan(joint_interfaces.at("effort")))
        {
          msg.effort.push_back(joint_interfaces.at("effort"));
        } else {
          msg.effort.push_back(std::numeric_limits<double>::quiet_NaN());
        }
      }

      rt_pub->unlockAndPublish();
    }
  }
}

controller_interface::return_type FFWJointStateBroadcaster::update(
  const rclcpp::Time & time, const rclcpp::Duration & /*period*/)
{
  publish_joint_states(time);
  return controller_interface::return_type::OK;
}

}  // namespace ffw_joint_state_broadcaster

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(
  ffw_joint_state_broadcaster::FFWJointStateBroadcaster,
  controller_interface::ControllerInterface)
