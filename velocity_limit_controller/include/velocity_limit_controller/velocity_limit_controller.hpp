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

#ifndef VELOCITY_LIMIT_CONTROLLER__VELOCITY_LIMIT_CONTROLLER_HPP_
#define VELOCITY_LIMIT_CONTROLLER__VELOCITY_LIMIT_CONTROLLER_HPP_

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/duration.hpp"
#include "rclcpp/subscription.hpp"
#include "rclcpp/time.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace velocity_limit_controller
{

class VelocityLimitController : public controller_interface::ControllerInterface
{
public:
  VelocityLimitController();

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;

  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  controller_interface::CallbackReturn on_init() override;

  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn on_error(
    const rclcpp_lifecycle::State & previous_state) override;

protected:
  // Number of joints
  size_t num_joints_;

  // Joint names
  std::vector<std::string> joint_names_;

  // Command interfaces: goal_velocity for each joint
  std::vector<std::reference_wrapper<hardware_interface::LoanedCommandInterface>>
    goal_velocity_command_interfaces_;

  // State interfaces: position for each joint
  std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>>
    position_state_interfaces_;

  // Trajectory subscription
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_subscriber_;

  // Realtime-safe buffer for trajectory messages
  realtime_tools::RealtimeBuffer<std::shared_ptr<trajectory_msgs::msg::JointTrajectory>>
    trajectory_msg_buffer_;

  // Flag to gate message processing
  std::atomic<bool> subscriber_is_active_{false};

  // Current target positions (from trajectory points[0])
  std::vector<double> target_positions_;

  // Current output velocities
  std::vector<double> output_velocities_;

  // Previous output velocities (for low-pass filter)
  std::vector<double> prev_output_velocities_;

  // Lock state per joint (for use_lock feature)
  std::vector<bool> is_locked_;

  // Last trajectory message time (for timeout detection)
  rclcpp::Time last_trajectory_time_;

  // Parameters
  struct Params
  {
    std::vector<std::string> joints;
    std::string trajectory_topic;
    std::vector<double> a_brake;  // Braking acceleration per joint
    std::vector<double> v_max;    // Maximum velocity per joint
    std::vector<double> v_min;     // Minimum velocity per joint
    std::vector<double> ramp_up;  // Velocity ramp-up rate per joint (rad/s²)
    std::vector<double> ramp_down;  // Velocity ramp-down rate per joint (rad/s²)
    std::vector<double> e_dead_in;   // Deadband inner threshold per joint (for lock)
    std::vector<double> e_dead_out;  // Deadband outer threshold per joint (for lock)
    bool use_lock;                    // Enable locking near target
    double lpf_cutoff_hz;             // Low-pass filter cutoff frequency
    double timeout_ms;                 // Trajectory timeout in milliseconds
  } params_;

  // Low-pass filter state (alpha coefficient)
  double lpf_alpha_;

  // Callback for trajectory subscription
  void trajectory_callback(const std::shared_ptr<trajectory_msgs::msg::JointTrajectory> msg);

  // Validate trajectory message
  bool validate_trajectory_msg(const trajectory_msgs::msg::JointTrajectory & msg) const;

  // Map joint names from trajectory to controller joints
  std::vector<size_t> map_trajectory_joints(
    const std::vector<std::string> & trajectory_joint_names) const;

  // Compute velocity limit for a single joint
  double compute_velocity_limit(
    size_t joint_idx, double target_pos, double current_pos, double dt);

  // Apply low-pass filter
  double apply_low_pass_filter(size_t joint_idx, double input) const;

  // Check if trajectory has timed out
  bool is_trajectory_timed_out(const rclcpp::Time & current_time) const;
};

}  // namespace velocity_limit_controller

#endif  // VELOCITY_LIMIT_CONTROLLER__VELOCITY_LIMIT_CONTROLLER_HPP_
