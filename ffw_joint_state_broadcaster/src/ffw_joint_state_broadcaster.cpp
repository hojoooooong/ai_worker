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
#include <cstddef>
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

    // Read optional offsets parameter (must match joints size if provided)
    std::string offsets_param = "joint_groups." + group_name + ".offsets";
    if (!get_node()->has_parameter(offsets_param)) {
      get_node()->declare_parameter(offsets_param, std::vector<double>());
    }
    try {
      config.offsets = get_node()->get_parameter(offsets_param).as_double_array();
    } catch (const std::exception & e) {
      RCLCPP_DEBUG(
        get_node()->get_logger(),
        "Failed to get 'offsets' for group '%s': %s. Using zeros.",
        group_name.c_str(), e.what());
      config.offsets.clear();
    }

    if (!config.offsets.empty() && config.offsets.size() != config.joints.size()) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "The number of offsets (%zu) for group '%s' does not match the number of joints (%zu).",
        config.offsets.size(), group_name.c_str(), config.joints.size());
      return controller_interface::CallbackReturn::ERROR;
    }

    if (config.offsets.empty()) {
      config.offsets.assign(config.joints.size(), 0.0);
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
  if (!init_publishing_handles()) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Failed to initialize publishing handles. Controller will not run.");
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FFWJointStateBroadcaster::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  handles_per_group_.clear();
  group_infos_.clear();
  group_names_.clear();
  return controller_interface::CallbackReturn::SUCCESS;
}

bool FFWJointStateBroadcaster::init_publishing_handles()
{
  handles_per_group_.clear();
  group_infos_.clear();
  group_names_.clear();

  if (state_interfaces_.empty() || joint_groups_.empty()) {
    return false;
  }

  // Build group_names_ vector for indexing
  group_names_.reserve(joint_groups_.size());
  for (const auto & [group_name, config] : joint_groups_) {
    group_names_.push_back(group_name);
  }

  // Initialize group_infos_ and handles_per_group_ with correct size
  const size_t num_groups = group_names_.size();
  group_infos_.resize(num_groups);
  handles_per_group_.resize(num_groups);

  // Build joint name to group index and joint index mapping (done once, not in update loop)
  std::unordered_map<std::string, std::pair<size_t, size_t>> joint_to_group_and_index;
  for (size_t group_idx = 0; group_idx < group_names_.size(); ++group_idx) {
    const std::string & group_name = group_names_[group_idx];
    const auto & config = joint_groups_[group_name];

    group_infos_[group_idx].group_index = group_idx;

    for (size_t joint_idx = 0; joint_idx < config.joints.size(); ++joint_idx) {
      const std::string & joint_name = config.joints[joint_idx];
      joint_to_group_and_index[joint_name] = {group_idx, joint_idx};
    }
  }

  // Pre-allocate message sizes and joint names for each group
  for (size_t group_idx = 0; group_idx < group_names_.size(); ++group_idx) {
    const std::string & group_name = group_names_[group_idx];
    const auto & config = joint_groups_[group_name];

    if (rt_publishers_.count(group_name) == 0) {
      continue;
    }

    auto & rt_pub = rt_publishers_[group_name];
    if (!rt_pub) {
      continue;
    }

    // Pre-allocate message size and set joint names (done once, not in update loop)
    auto & msg = rt_pub->msg_;
    const size_t num_joints = config.joints.size();
    msg.name.resize(num_joints);
    msg.position.resize(num_joints, std::numeric_limits<double>::quiet_NaN());
    msg.velocity.resize(num_joints, std::numeric_limits<double>::quiet_NaN());
    msg.effort.resize(num_joints, std::numeric_limits<double>::quiet_NaN());

    // Set joint names once
    for (size_t i = 0; i < num_joints; ++i) {
      msg.name[i] = config.joints[i];
    }
  }

  // Process all state interfaces and create handles (all string parsing done here)
  for (size_t interface_idx = 0; interface_idx < state_interfaces_.size(); ++interface_idx) {
    const auto & state_interface = state_interfaces_[interface_idx];
    const std::string & joint_name = state_interface.get_prefix_name();
    const std::string & interface_name = state_interface.get_interface_name();

    // Map interface name to joint state field (position, velocity, effort)
    std::string mapped_name = interface_name;
    if (map_interface_to_joint_state_.count(interface_name) > 0) {
      mapped_name = map_interface_to_joint_state_[interface_name];
    }

    // For joystick sensor interfaces, create synthetic joint name
    // All string operations done here, not in update loop
    std::string data_key = joint_name;
    if (interface_name.find("JOYSTICK") != std::string::npos) {
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
      mapped_name = "position";  // Joystick values stored as position
    }

    // Find which group(s) this joint belongs to
    auto it = joint_to_group_and_index.find(data_key);
    if (it == joint_to_group_and_index.end()) {
      continue;  // This joint is not in any configured group
    }

    const auto & [group_idx, joint_idx] = it->second;

    // Create handle for this interface
    const auto & config = joint_groups_[group_names_[group_idx]];
    InterfaceHandle handle;
    handle.state_interface_index = interface_idx;
    handle.group_index = group_idx;
    handle.has_position = (mapped_name == "position");
    handle.has_velocity = (mapped_name == "velocity");
    handle.has_effort = (mapped_name == "effort");

    // Set message indices
    handle.msg_position_index = handle.has_position ? joint_idx : SIZE_MAX;
    handle.msg_velocity_index = handle.has_velocity ? joint_idx : SIZE_MAX;
    handle.msg_effort_index = handle.has_effort ? joint_idx : SIZE_MAX;

    // Set position offset (applied only for position interfaces)
    handle.position_offset =
      (handle.has_position && joint_idx < config.offsets.size()) ? config.offsets[joint_idx] : 0.0;

    // Add handle to the corresponding group's vector (pre-grouped for efficient iteration)
    handles_per_group_[group_idx].push_back(handle);

    // Update group info indices (for potential future use)
    const size_t handle_index = handles_per_group_[group_idx].size() - 1;
    if (handle.has_position) {
      group_infos_[group_idx].position_indices.push_back(handle_index);
    }
    if (handle.has_velocity) {
      group_infos_[group_idx].velocity_indices.push_back(handle_index);
    }
    if (handle.has_effort) {
      group_infos_[group_idx].effort_indices.push_back(handle_index);
    }
  }

  // Count total handles for logging
  size_t total_handles = 0;
  for (const auto & handles : handles_per_group_) {
    total_handles += handles.size();
  }

  RCLCPP_INFO(
    get_node()->get_logger(),
    "Initialized %zu interface handles across %zu groups",
    total_handles, group_names_.size());

  return true;
}

void FFWJointStateBroadcaster::publish_joint_states(const rclcpp::Time & time)
{
  // Real-time safe: Only copy values using pre-computed handles
  // No string operations, no map lookups, no dynamic memory allocation

  // Process each group separately (lock once per group, not per interface)
  for (size_t group_idx = 0; group_idx < group_names_.size(); ++group_idx) {
    const std::string & group_name = group_names_[group_idx];

    auto rt_pub_it = rt_publishers_.find(group_name);
    if (rt_pub_it == rt_publishers_.end() || !rt_pub_it->second) {
      continue;
    }

    auto & rt_pub = rt_pub_it->second;

    // Try to lock once per group (non-blocking)
    if (!rt_pub->trylock()) {
      continue;  // Skip this group if lock fails
    }

    auto & msg = rt_pub->msg_;

    // Set timestamp once per group
    msg.header.stamp = time;

    // Copy all interface values for this group
    // No if checks needed - handles_per_group_[group_idx] contains only handles for this group
    for (const auto & handle : handles_per_group_[group_idx]) {
      // Get value from state interface (fast, no string operations)
      const auto & state_interface = state_interfaces_[handle.state_interface_index];
      auto value = state_interface.get_optional();

      if (!value.has_value()) {
        continue;
      }

      const double val = *value;

      // Copy value directly to pre-allocated message position
      // Apply position_offset for alignment with joint_trajectory_command_broadcaster
      if (handle.has_position && handle.msg_position_index != SIZE_MAX) {
        msg.position[handle.msg_position_index] = val + handle.position_offset;
      }
      if (handle.has_velocity && handle.msg_velocity_index != SIZE_MAX) {
        msg.velocity[handle.msg_velocity_index] = val;
      }
      if (handle.has_effort && handle.msg_effort_index != SIZE_MAX) {
        msg.effort[handle.msg_effort_index] = val;
      }
    }

    rt_pub->unlockAndPublish();
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
