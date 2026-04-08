// Copyright 2023 ROBOTIS CO., LTD.
// Authors: Sungho Woo

#ifndef V_MARKER_ESTIMATION__HOLONOMIC_DRIVE_CONTROLLER_HPP_
#define V_MARKER_ESTIMATION__HOLONOMIC_DRIVE_CONTROLLER_HPP_

#include <memory>
#include <chrono>
#include <cmath>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"

namespace v_marker_estimation
{

class HolonomicDriveController : public rclcpp::Node
{
public:
  explicit HolonomicDriveController(
    const std::string & node_name = "holonomic_drive_controller");
  ~HolonomicDriveController();

private:
  // Service callbacks
  void drive_trigger_callback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response);

  // Subscription callbacks
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void goal_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);

  // Timer callback for PD control
  void control_timer_callback();

  // Helper functions
  double quaternion_to_yaw(const geometry_msgs::msg::Quaternion & quat);
  double normalize_angle(double angle);
  double clamp_velocity(double vel, double min_vel, double max_vel);
  void notify_drive_finish();
  void reset_state();
  void publish_zero_velocity();
  void publish_velocity(double vx, double vy, double angular_z);

  // Path visualization helpers
  geometry_msgs::msg::PoseStamped make_pose_stamped(
    double x, double y, double yaw, const rclcpp::Time & stamp);
  geometry_msgs::msg::Quaternion yaw_to_quaternion(double yaw);
  void publish_planned_path();
  void append_actual_path();
  bool is_point_in_collision_ring(double x, double y) const;
  void publish_collision_markers(const rclcpp::Time & stamp);

  // ROS 2 interfaces
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr drive_trigger_srv_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr drive_finish_client_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_stamped_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr planned_path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr actual_path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr collision_marker_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  // Control parameters
  double Kp_linear_;
  double Kd_linear_;
  double Kp_angular_;
  double Kd_angular_;
  double max_linear_velocity_;
  double min_linear_velocity_;
  double max_angular_velocity_;
  double min_angular_velocity_;
  double position_tolerance_;
  double orientation_tolerance_;
  bool use_twist_stamped_;
  bool enable_collision_stop_;
  double collision_outer_front_x_;
  double collision_outer_back_x_;
  double collision_outer_half_y_;
  double collision_inner_front_x_;
  double collision_inner_back_x_;
  double collision_inner_half_y_;
  int collision_min_points_;
  double collision_range_min_;
  double collision_range_max_;
  std::string collision_frame_;
  std::string collision_marker_topic_;

  // Current robot state (from odom)
  double current_x_;
  double current_y_;
  double current_yaw_;

  // Goal state
  double goal_x_;
  double goal_y_;
  double goal_yaw_;

  // Previous errors for derivative term
  double prev_error_x_;
  double prev_error_y_;
  double prev_error_yaw_;
  rclcpp::Time prev_time_;

  // Path visualization
  nav_msgs::msg::Path planned_path_;
  nav_msgs::msg::Path actual_path_;
  std::string path_frame_id_;
  double start_x_;
  double start_y_;
  double start_yaw_;

  // State flags
  bool drive_active_;
  bool odom_received_;
  bool controller_enabled_;
  bool final_yaw_tracking_active_;
  bool collision_blocked_;
  int collision_points_in_ring_;
};

}  // namespace v_marker_estimation

#endif  // V_MARKER_ESTIMATION__HOLONOMIC_DRIVE_CONTROLLER_HPP_
