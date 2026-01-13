
// Copyright (c) 2024
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

#include "velocity_limit_controller/velocity_limit_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "controller_interface/helpers.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/qos.hpp"

namespace velocity_limit_controller
{

VelocityLimitController::VelocityLimitController()
: num_joints_(0)
{
}

controller_interface::CallbackReturn VelocityLimitController::on_init()
{
  try
  {
    // Declare parameters
    auto_declare<std::vector<std::string>>("joints", std::vector<std::string>());
    auto_declare<std::string>("trajectory_topic", "");
    auto_declare<std::vector<double>>("a_brake", std::vector<double>());
    auto_declare<std::vector<double>>("v_max", std::vector<double>());
    auto_declare<std::vector<double>>("v_min", std::vector<double>());
    auto_declare<std::vector<double>>("ramp_up", std::vector<double>());
    auto_declare<std::vector<double>>("ramp_down", std::vector<double>());
    auto_declare<std::vector<double>>("e_dead_in", std::vector<double>());
    auto_declare<std::vector<double>>("e_dead_out", std::vector<double>());
    auto_declare<bool>("use_lock", false);
    auto_declare<double>("lpf_cutoff_hz", 10.0);
    auto_declare<double>("timeout_ms", 100.0);
  }
  catch (const std::exception & e)
  {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }

  return CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
VelocityLimitController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration conf;
  conf.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  conf.names.reserve(num_joints_);

  for (const auto & joint_name : joint_names_)
  {
    conf.names.push_back(joint_name + "/" + hardware_interface::HW_IF_VELOCITY);
  }

  return conf;
}

controller_interface::InterfaceConfiguration
VelocityLimitController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration conf;
  conf.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  conf.names.reserve(num_joints_);

  for (const auto & joint_name : joint_names_)
  {
    conf.names.push_back(joint_name + "/" + hardware_interface::HW_IF_POSITION);
  }

  return conf;
}

controller_interface::CallbackReturn VelocityLimitController::on_configure(
  const rclcpp_lifecycle::State &)
{
  auto logger = get_node()->get_logger();

  // Get parameters
  params_.joints = get_node()->get_parameter("joints").as_string_array();
  params_.trajectory_topic = get_node()->get_parameter("trajectory_topic").as_string();
  params_.a_brake = get_node()->get_parameter("a_brake").as_double_array();
  params_.v_max = get_node()->get_parameter("v_max").as_double_array();
  params_.v_min = get_node()->get_parameter("v_min").as_double_array();
  params_.ramp_up = get_node()->get_parameter("ramp_up").as_double_array();
  params_.ramp_down = get_node()->get_parameter("ramp_down").as_double_array();
  params_.e_dead_in = get_node()->get_parameter("e_dead_in").as_double_array();
  params_.e_dead_out = get_node()->get_parameter("e_dead_out").as_double_array();
  params_.use_lock = get_node()->get_parameter("use_lock").as_bool();
  params_.lpf_cutoff_hz = get_node()->get_parameter("lpf_cutoff_hz").as_double();
  params_.timeout_ms = get_node()->get_parameter("timeout_ms").as_double();

  // Validate parameters
  num_joints_ = params_.joints.size();
  if (num_joints_ == 0)
  {
    RCLCPP_ERROR(logger, "No joints specified");
    return CallbackReturn::ERROR;
  }

  if (params_.trajectory_topic.empty())
  {
    RCLCPP_ERROR(logger, "trajectory_topic parameter is empty");
    return CallbackReturn::ERROR;
  }

  // Validate array sizes
  auto validate_array_size = [&](const std::vector<double> & arr, const std::string & name,
                                  bool allow_empty = false) -> bool
  {
    if (allow_empty && arr.empty())
    {
      return true;
    }
    if (arr.size() != num_joints_)
    {
      RCLCPP_ERROR(
        logger, "Parameter '%s' has size %zu, expected %zu", name.c_str(), arr.size(),
        num_joints_);
      return false;
    }
    return true;
  };

  if (!validate_array_size(params_.a_brake, "a_brake") ||
      !validate_array_size(params_.v_max, "v_max") ||
      !validate_array_size(params_.v_min, "v_min") ||
      !validate_array_size(params_.ramp_up, "ramp_up") ||
      !validate_array_size(params_.ramp_down, "ramp_down") ||
      !validate_array_size(params_.e_dead_in, "e_dead_in", true) ||
      !validate_array_size(params_.e_dead_out, "e_dead_out", true))
  {
    return CallbackReturn::ERROR;
  }

  // Initialize deadband arrays if empty
  if (params_.e_dead_in.empty())
  {
    params_.e_dead_in.resize(num_joints_, 0.0);
  }
  if (params_.e_dead_out.empty())
  {
    params_.e_dead_out.resize(num_joints_, 0.0);
  }

  // Compute low-pass filter coefficient
  const double update_rate = get_update_rate();
  if (update_rate <= 0.0)
  {
    RCLCPP_ERROR(logger, "Controller update rate is invalid: %f", update_rate);
    return CallbackReturn::ERROR;
  }

  const double dt = 1.0 / update_rate;
  const double cutoff_rad = 2.0 * M_PI * params_.lpf_cutoff_hz;
  lpf_alpha_ = cutoff_rad * dt / (1.0 + cutoff_rad * dt);

  // Initialize state vectors
  joint_names_ = params_.joints;
  target_positions_.resize(num_joints_, 0.0);
  output_velocities_.resize(num_joints_, 0.0);
  prev_output_velocities_.resize(num_joints_, 0.0);
  is_locked_.resize(num_joints_, false);

  // Create trajectory subscriber
  trajectory_subscriber_ = get_node()->create_subscription<trajectory_msgs::msg::JointTrajectory>(
    params_.trajectory_topic, rclcpp::SystemDefaultsQoS(),
    std::bind(&VelocityLimitController::trajectory_callback, this, std::placeholders::_1));

  RCLCPP_INFO(
    logger, "Configured VelocityLimitController with %zu joints, subscribing to '%s'", num_joints_,
    params_.trajectory_topic.c_str());

  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn VelocityLimitController::on_activate(
  const rclcpp_lifecycle::State &)
{
  auto logger = get_node()->get_logger();

  // Claim command interfaces
  goal_velocity_command_interfaces_.reserve(num_joints_);
  if (!controller_interface::get_ordered_interfaces(
        command_interfaces_, joint_names_, hardware_interface::HW_IF_VELOCITY,
        goal_velocity_command_interfaces_))
  {
      RCLCPP_ERROR(
        logger, "Expected %zu '%s' command interfaces, got %zu.", num_joints_,
        hardware_interface::HW_IF_VELOCITY, goal_velocity_command_interfaces_.size());
    return CallbackReturn::ERROR;
  }

  // Claim state interfaces
  position_state_interfaces_.reserve(num_joints_);
  if (!controller_interface::get_ordered_interfaces(
        state_interfaces_, joint_names_, hardware_interface::HW_IF_POSITION,
        position_state_interfaces_))
  {
      RCLCPP_ERROR(
        logger, "Expected %zu '%s' state interfaces, got %zu.", num_joints_,
        hardware_interface::HW_IF_POSITION, position_state_interfaces_.size());
    return CallbackReturn::ERROR;
  }

  // Initialize output velocities to zero
  for (size_t i = 0; i < num_joints_; ++i)
  {
    output_velocities_[i] = 0.0;
    prev_output_velocities_[i] = 0.0;
    is_locked_[i] = false;
    if (!goal_velocity_command_interfaces_[i].get().set_value(0.0))
    {
      RCLCPP_WARN(
        logger, "Failed to set initial velocity to 0.0 for joint '%s'", joint_names_[i].c_str());
    }
  }

  // Initialize target positions from current state
  for (size_t i = 0; i < num_joints_; ++i)
  {
    const auto position_value_op = position_state_interfaces_[i].get().get_optional();
    if (position_value_op.has_value())
    {
      target_positions_[i] = position_value_op.value();
    }
  }

  last_trajectory_time_ = rclcpp::Time(0, 0, get_node()->get_clock()->get_clock_type());
  subscriber_is_active_ = true;

  RCLCPP_INFO(logger, "VelocityLimitController activated");
  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn VelocityLimitController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  auto logger = get_node()->get_logger();

  // Set all velocities to zero
  for (size_t i = 0; i < num_joints_; ++i)
  {
    if (i < goal_velocity_command_interfaces_.size())
    {
      if (!goal_velocity_command_interfaces_[i].get().set_value(0.0))
      {
        RCLCPP_WARN(
          logger, "Failed to set velocity to 0.0 for joint '%s' during deactivate",
          joint_names_[i].c_str());
      }
    }
  }

  subscriber_is_active_ = false;

  // Clear interfaces
  goal_velocity_command_interfaces_.clear();
  position_state_interfaces_.clear();

  RCLCPP_INFO(logger, "VelocityLimitController deactivated");
  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn VelocityLimitController::on_error(
  const rclcpp_lifecycle::State &)
{
  subscriber_is_active_ = false;
  goal_velocity_command_interfaces_.clear();
  position_state_interfaces_.clear();
  return CallbackReturn::SUCCESS;
}

controller_interface::return_type VelocityLimitController::update(
  const rclcpp::Time & time, const rclcpp::Duration & period)
{
  // Check for new trajectory message
  auto new_trajectory_msg = trajectory_msg_buffer_.readFromRT();
  if (new_trajectory_msg && *new_trajectory_msg)
  {
    const auto & msg = **new_trajectory_msg;
    if (validate_trajectory_msg(msg) && !msg.points.empty())
    {
      // Map trajectory joints to controller joints
      std::vector<size_t> joint_map = map_trajectory_joints(msg.joint_names);

      // Extract target positions from points[0]
      const auto & point = msg.points[0];
      if (point.positions.size() == msg.joint_names.size())
      {
        for (size_t i = 0; i < joint_map.size(); ++i)
        {
          if (joint_map[i] < msg.joint_names.size())
          {
            target_positions_[i] = point.positions[joint_map[i]];
          }
        }
        last_trajectory_time_ = time;
      }
    }
    // Clear the buffer after reading
    trajectory_msg_buffer_.reset();
  }

  // Check for timeout
  bool timed_out = is_trajectory_timed_out(time);

  // Read current positions
  std::vector<double> current_positions(num_joints_);
  for (size_t i = 0; i < num_joints_; ++i)
  {
    const auto position_value_op = position_state_interfaces_[i].get().get_optional();
    if (position_value_op.has_value())
    {
      current_positions[i] = position_value_op.value();
    }
    else
    {
      current_positions[i] = target_positions_[i];  // Fallback to target if no reading
    }
  }

  // Compute and apply velocity limits for each joint
  const double dt = period.seconds();
  for (size_t i = 0; i < num_joints_; ++i)
  {
    double v_target = 0.0;

    if (!timed_out)
    {
      // Compute velocity limit based on braking distance
      v_target = compute_velocity_limit(i, target_positions_[i], current_positions[i], dt);
    }
    // If timed out, v_target remains 0.0 (will ramp down)

    // Apply low-pass filter
    v_target = apply_low_pass_filter(i, v_target);

    // Apply rate limiting
    const double v_delta = v_target - output_velocities_[i];
    const double max_delta_up = params_.ramp_up[i] * dt;
    const double max_delta_down = params_.ramp_down[i] * dt;

    output_velocities_[i] += std::clamp(v_delta, -max_delta_down, max_delta_up);

    // Clamp to limits (using absolute value for clamping since goal_velocity is always positive)
    const double v_abs = std::abs(output_velocities_[i]);
    const double v_clamped = std::clamp(v_abs, 0.0, params_.v_max[i]);

    // Write absolute value to command interface
    // Note: goal_velocity for Dynamixel Y is always positive (velocity limit, not direction)
    if (!goal_velocity_command_interfaces_[i].get().set_value(v_clamped))
    {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "Failed to set velocity value for joint '%s' in command interface",
        joint_names_[i].c_str());
    }
  }

  // Store for next iteration (for low-pass filter)
  prev_output_velocities_ = output_velocities_;

  return controller_interface::return_type::OK;
}

void VelocityLimitController::trajectory_callback(
  const std::shared_ptr<trajectory_msgs::msg::JointTrajectory> msg)
{
  if (!subscriber_is_active_)
  {
    return;
  }

  if (!validate_trajectory_msg(*msg))
  {
    return;
  }

  // Write to realtime buffer
  trajectory_msg_buffer_.writeFromNonRT(msg);
}

bool VelocityLimitController::validate_trajectory_msg(
  const trajectory_msgs::msg::JointTrajectory & msg) const
{
  auto logger = get_node()->get_logger();

  if (msg.joint_names.empty())
  {
    RCLCPP_WARN(logger, "Received trajectory with empty joint names");
    return false;
  }

  if (msg.points.empty())
  {
    RCLCPP_WARN(logger, "Received trajectory with no points");
    return false;
  }

  // Check that points[0] has positions
  if (msg.points[0].positions.empty())
  {
    RCLCPP_WARN(logger, "Received trajectory with no positions in first point");
    return false;
  }

  if (msg.points[0].positions.size() != msg.joint_names.size())
  {
    RCLCPP_WARN(
      logger, "Mismatch between joint_names size (%zu) and positions size (%zu) in first point",
      msg.joint_names.size(), msg.points[0].positions.size());
    return false;
  }

  return true;
}

std::vector<size_t> VelocityLimitController::map_trajectory_joints(
  const std::vector<std::string> & trajectory_joint_names) const
{
  std::vector<size_t> joint_map(num_joints_, SIZE_MAX);

  for (size_t i = 0; i < num_joints_; ++i)
  {
    const std::string & controller_joint = joint_names_[i];
    auto it = std::find(trajectory_joint_names.begin(), trajectory_joint_names.end(), controller_joint);
    if (it != trajectory_joint_names.end())
    {
      joint_map[i] = std::distance(trajectory_joint_names.begin(), it);
    }
  }

  return joint_map;
}

double VelocityLimitController::compute_velocity_limit(
  size_t joint_idx, double target_pos, double current_pos, double /* dt */)
{
  // Position error
  double e = target_pos - current_pos;
  double e_abs = std::abs(e);

  // Check for locking near target (optional)
  if (params_.use_lock)
  {
    const double e_dead_in = params_.e_dead_in[joint_idx];
    const double e_dead_out = params_.e_dead_out[joint_idx];

    // Hysteresis: if inside deadband, lock (return 0)
    // If outside outer deadband, unlock
    if (e_abs < e_dead_in)
    {
      is_locked_[joint_idx] = true;
      return 0.0;
    }
    else if (e_abs > e_dead_out)
    {
      is_locked_[joint_idx] = false;
    }
    else if (is_locked_[joint_idx])
    {
      // Still in hysteresis zone and locked
      return 0.0;
    }
  }

  // Compute braking-distance-based velocity limit
  // v_stop = sqrt(2 * a_brake * |e|)
  const double a_brake = params_.a_brake[joint_idx];
  double v_stop = std::sqrt(2.0 * a_brake * e_abs);

  // Apply minimum velocity threshold to overcome friction/deadband
  // If computed velocity is very small but non-zero, use v_min to ensure motor moves
  if (v_stop > 0.0 && v_stop < params_.v_min[joint_idx])
  {
    v_stop = params_.v_min[joint_idx];
  }

  // Clamp to maximum limit (absolute value, since goal_velocity is always positive)
  v_stop = std::clamp(v_stop, 0.0, params_.v_max[joint_idx]);

  // Sign based on error direction (kept for internal computation, but abs() applied when writing)
  if (e < 0.0)
  {
    v_stop = -v_stop;
  }

  return v_stop;
}

double VelocityLimitController::apply_low_pass_filter(size_t joint_idx, double input) const
{
  // Simple first-order low-pass filter: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
  double output = lpf_alpha_ * input + (1.0 - lpf_alpha_) * prev_output_velocities_[joint_idx];
  return output;
}

bool VelocityLimitController::is_trajectory_timed_out(const rclcpp::Time & current_time) const
{
  if (params_.timeout_ms <= 0.0)
  {
    return false;  // Timeout disabled
  }

  if (last_trajectory_time_.seconds() == 0.0 && last_trajectory_time_.nanoseconds() == 0)
  {
    return false;  // No trajectory received yet
  }

  const rclcpp::Duration timeout_duration =
    rclcpp::Duration::from_seconds(params_.timeout_ms / 1000.0);
  const rclcpp::Duration elapsed = current_time - last_trajectory_time_;

  return elapsed > timeout_duration;
}

}  // namespace velocity_limit_controller

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(
  velocity_limit_controller::VelocityLimitController, controller_interface::ControllerInterface)
