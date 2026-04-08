// Copyright 2023 ROBOTIS CO., LTD.
// Authors: Sungho Woo

#include "v_marker_estimation/holonomic_drive_controller.hpp"

namespace v_marker_estimation
{

HolonomicDriveController::HolonomicDriveController(const std::string & node_name)
: Node(node_name),
  Kp_linear_(1.0),
  Kd_linear_(0.3),
  Kp_angular_(1.5),
  Kd_angular_(0.5),
  max_linear_velocity_(0.3),
  min_linear_velocity_(0.02),
  max_angular_velocity_(0.6),
  min_angular_velocity_(0.05),
  position_tolerance_(0.02),
  orientation_tolerance_(0.03),
  use_twist_stamped_(true),
  enable_collision_stop_(false),
  collision_outer_front_x_(0.45),
  collision_outer_back_x_(0.45),
  collision_outer_half_y_(0.35),
  collision_inner_front_x_(0.20),
  collision_inner_back_x_(0.20),
  collision_inner_half_y_(0.15),
  collision_min_points_(6),
  collision_range_min_(0.05),
  collision_range_max_(2.5),
  collision_frame_("base_link"),
  collision_marker_topic_("/holonomic_drive/collision_zone"),
  current_x_(0.0),
  current_y_(0.0),
  current_yaw_(0.0),
  goal_x_(0.0),
  goal_y_(0.0),
  goal_yaw_(0.0),
  prev_error_x_(0.0),
  prev_error_y_(0.0),
  prev_error_yaw_(0.0),
  path_frame_id_("odom"),
  start_x_(0.0),
  start_y_(0.0),
  start_yaw_(0.0),
  drive_active_(false),
  odom_received_(false),
  controller_enabled_(true),
  final_yaw_tracking_active_(false),
  collision_blocked_(false),
  collision_points_in_ring_(0)
{
  // Declare parameters
  this->declare_parameter("Kp_linear", 1.0);
  this->declare_parameter("Kd_linear", 0.3);
  this->declare_parameter("Kp_angular", 1.5);
  this->declare_parameter("Kd_angular", 0.5);
  this->declare_parameter("max_linear_velocity", 0.3);
  this->declare_parameter("min_linear_velocity", 0.02);
  this->declare_parameter("max_angular_velocity", 0.6);
  this->declare_parameter("min_angular_velocity", 0.05);
  this->declare_parameter("position_tolerance", 0.02);
  this->declare_parameter("orientation_tolerance", 0.03);
  this->declare_parameter("use_twist_stamped", true);
  this->declare_parameter("path_frame_id", "odom");
  this->declare_parameter("enable_collision_stop", false);
  this->declare_parameter("collision_outer_front_x", 0.45);
  this->declare_parameter("collision_outer_back_x", 0.45);
  this->declare_parameter("collision_outer_half_y", 0.35);
  this->declare_parameter("collision_inner_front_x", 0.20);
  this->declare_parameter("collision_inner_back_x", 0.20);
  this->declare_parameter("collision_inner_half_y", 0.15);
  this->declare_parameter("collision_min_points", 6);
  this->declare_parameter("collision_range_min", 0.05);
  this->declare_parameter("collision_range_max", 2.5);
  this->declare_parameter("collision_frame", "base_link");
  this->declare_parameter("collision_marker_topic", "/holonomic_drive/collision_zone");

  // Get parameters
  Kp_linear_ = this->get_parameter("Kp_linear").as_double();
  Kd_linear_ = this->get_parameter("Kd_linear").as_double();
  Kp_angular_ = this->get_parameter("Kp_angular").as_double();
  Kd_angular_ = this->get_parameter("Kd_angular").as_double();
  max_linear_velocity_ = this->get_parameter("max_linear_velocity").as_double();
  min_linear_velocity_ = this->get_parameter("min_linear_velocity").as_double();
  max_angular_velocity_ = this->get_parameter("max_angular_velocity").as_double();
  min_angular_velocity_ = this->get_parameter("min_angular_velocity").as_double();
  position_tolerance_ = this->get_parameter("position_tolerance").as_double();
  orientation_tolerance_ = this->get_parameter("orientation_tolerance").as_double();
  use_twist_stamped_ = this->get_parameter("use_twist_stamped").as_bool();
  path_frame_id_ = this->get_parameter("path_frame_id").as_string();
  enable_collision_stop_ = this->get_parameter("enable_collision_stop").as_bool();
  collision_outer_front_x_ = this->get_parameter("collision_outer_front_x").as_double();
  collision_outer_back_x_ = this->get_parameter("collision_outer_back_x").as_double();
  collision_outer_half_y_ = this->get_parameter("collision_outer_half_y").as_double();
  collision_inner_front_x_ = this->get_parameter("collision_inner_front_x").as_double();
  collision_inner_back_x_ = this->get_parameter("collision_inner_back_x").as_double();
  collision_inner_half_y_ = this->get_parameter("collision_inner_half_y").as_double();
  collision_min_points_ = this->get_parameter("collision_min_points").as_int();
  collision_range_min_ = this->get_parameter("collision_range_min").as_double();
  collision_range_max_ = this->get_parameter("collision_range_max").as_double();
  collision_frame_ = this->get_parameter("collision_frame").as_string();
  collision_marker_topic_ = this->get_parameter("collision_marker_topic").as_string();

  // Validate parameters
  if (max_linear_velocity_ < min_linear_velocity_) {
    RCLCPP_WARN(this->get_logger(),
      "max_linear_velocity (%.2f) < min_linear_velocity (%.2f), swapping values",
      max_linear_velocity_, min_linear_velocity_);
    std::swap(max_linear_velocity_, min_linear_velocity_);
  }
  if (max_angular_velocity_ < min_angular_velocity_) {
    RCLCPP_WARN(this->get_logger(),
      "max_angular_velocity (%.2f) < min_angular_velocity (%.2f), swapping values",
      max_angular_velocity_, min_angular_velocity_);
    std::swap(max_angular_velocity_, min_angular_velocity_);
  }
  if (collision_inner_front_x_ >= collision_outer_front_x_ ||
    collision_inner_back_x_ >= collision_outer_back_x_ ||
    collision_inner_half_y_ >= collision_outer_half_y_)
  {
    RCLCPP_WARN(this->get_logger(),
      "Invalid collision ring sizes (inner >= outer). Adjusting inner rectangle.");
    collision_inner_front_x_ = std::max(0.01, collision_outer_front_x_ * 0.5);
    collision_inner_back_x_ = std::max(0.01, collision_outer_back_x_ * 0.5);
    collision_inner_half_y_ = std::max(0.01, collision_outer_half_y_ * 0.5);
  }
  if (collision_range_max_ < collision_range_min_) {
    std::swap(collision_range_max_, collision_range_min_);
  }
  if (collision_min_points_ < 1) {
    collision_min_points_ = 1;
  }

  RCLCPP_INFO(this->get_logger(),
    "Holonomic Drive Controller initialized "
    "(Kp_lin: %.2f, Kd_lin: %.2f, Kp_ang: %.2f, Kd_ang: %.2f, "
    "lin_vel: [%.2f, %.2f], ang_vel: [%.2f, %.2f], "
    "pos_tol: %.3f, ori_tol: %.3f, msg_type: %s, collision_stop: %s)",
    Kp_linear_, Kd_linear_, Kp_angular_, Kd_angular_,
    min_linear_velocity_, max_linear_velocity_,
    min_angular_velocity_, max_angular_velocity_,
    position_tolerance_, orientation_tolerance_,
    use_twist_stamped_ ? "TwistStamped" : "Twist",
    enable_collision_stop_ ? "on" : "off");

  // Create service server for drive enable/disable
  drive_trigger_srv_ = this->create_service<std_srvs::srv::SetBool>(
    "/holonomic_drive_trigger",
    std::bind(&HolonomicDriveController::drive_trigger_callback, this,
      std::placeholders::_1, std::placeholders::_2));

  // Create client for drive finish notification
  drive_finish_client_ = this->create_client<std_srvs::srv::Trigger>(
    "/drive_finish");

  // Create publisher for cmd_vel (TwistStamped or Twist based on parameter)
  if (use_twist_stamped_) {
    cmd_vel_stamped_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
      "/cmd_vel", rclcpp::QoS(10).reliable());
    cmd_vel_pub_ = nullptr;
  } else {
    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
      "/cmd_vel", rclcpp::QoS(10).reliable());
    cmd_vel_stamped_pub_ = nullptr;
  }

  // Create path visualization publishers
  planned_path_pub_ = this->create_publisher<nav_msgs::msg::Path>(
    "/holonomic_drive/planned_path", rclcpp::QoS(10).reliable());
  actual_path_pub_ = this->create_publisher<nav_msgs::msg::Path>(
    "/holonomic_drive/actual_path", rclcpp::QoS(10).reliable());

  // Create subscribers
  const auto QOS_RKL10V = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();

  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "/odom", QOS_RKL10V,
    std::bind(&HolonomicDriveController::odom_callback, this, std::placeholders::_1));

  goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/goal_pose", QOS_RKL10V,
    std::bind(&HolonomicDriveController::goal_pose_callback, this, std::placeholders::_1));
  const auto QOS_SENSOR =
    rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();
  scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
    "/scan", QOS_SENSOR,
    std::bind(&HolonomicDriveController::scan_callback, this, std::placeholders::_1));
  collision_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    collision_marker_topic_, QOS_RKL10V);

  // Control loop at 50 Hz
  control_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(20),
    std::bind(&HolonomicDriveController::control_timer_callback, this));

  prev_time_ = this->now();
}

HolonomicDriveController::~HolonomicDriveController()
{
}

void HolonomicDriveController::drive_trigger_callback(
  const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
  std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
  controller_enabled_ = request->data;

  if (!controller_enabled_ && drive_active_) {
    RCLCPP_INFO(this->get_logger(), "Controller disabled, stopping active drive");
    publish_zero_velocity();
    reset_state();
  }

  RCLCPP_INFO(this->get_logger(), "Holonomic drive controller %s",
    controller_enabled_ ? "enabled" : "disabled");

  response->success = true;
  response->message = controller_enabled_ ? "Controller enabled" : "Controller disabled";
}

void HolonomicDriveController::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  current_x_ = msg->pose.pose.position.x;
  current_y_ = msg->pose.pose.position.y;
  current_yaw_ = quaternion_to_yaw(msg->pose.pose.orientation);
  odom_received_ = true;
}

void HolonomicDriveController::goal_pose_callback(
  const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if (!controller_enabled_) {
    RCLCPP_DEBUG(this->get_logger(), "Goal pose received but controller is disabled, ignoring");
    return;
  }

  goal_x_ = msg->pose.position.x;
  goal_y_ = msg->pose.position.y;
  goal_yaw_ = quaternion_to_yaw(msg->pose.orientation);

  // Record start position for planned path
  start_x_ = current_x_;
  start_y_ = current_y_;
  start_yaw_ = current_yaw_;

  // Reset derivative terms for new goal
  prev_error_x_ = 0.0;
  prev_error_y_ = 0.0;
  prev_error_yaw_ = 0.0;
  prev_time_ = this->now();
  final_yaw_tracking_active_ = false;

  drive_active_ = true;

  // Build and publish planned path (straight line from start to goal)
  publish_planned_path();

  // Reset actual path and add starting point
  actual_path_.poses.clear();
  actual_path_.header.frame_id = path_frame_id_;
  actual_path_.header.stamp = this->now();
  actual_path_.poses.push_back(
    make_pose_stamped(current_x_, current_y_, current_yaw_, this->now()));

  double dx = goal_x_ - current_x_;
  double dy = goal_y_ - current_y_;
  double distance = std::hypot(dx, dy);
  double yaw_error = normalize_angle(goal_yaw_ - current_yaw_);

  RCLCPP_INFO(this->get_logger(),
    "New goal received: (%.3f, %.3f, %.3f rad) | "
    "Current: (%.3f, %.3f, %.3f rad) | "
    "Distance: %.3f m, Yaw error: %.3f rad",
    goal_x_, goal_y_, goal_yaw_,
    current_x_, current_y_, current_yaw_,
    distance, yaw_error);
}

void HolonomicDriveController::scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
  if (!enable_collision_stop_) {
    collision_blocked_ = false;
    collision_points_in_ring_ = 0;
    publish_collision_markers(this->now());
    return;
  }

  if (!collision_frame_.empty() && msg->header.frame_id != collision_frame_) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Collision scan frame mismatch (scan=%s, expected=%s)",
      msg->header.frame_id.c_str(), collision_frame_.c_str());
  }

  int hit_count = 0;
  double angle = msg->angle_min;
  for (const float r : msg->ranges) {
    if (std::isfinite(r) && r >= collision_range_min_ && r <= collision_range_max_) {
      const double x = static_cast<double>(r) * std::cos(angle);
      const double y = static_cast<double>(r) * std::sin(angle);
      if (is_point_in_collision_ring(x, y)) {
        ++hit_count;
      }
    }
    angle += msg->angle_increment;
  }

  collision_points_in_ring_ = hit_count;
  collision_blocked_ = (collision_points_in_ring_ >= collision_min_points_);
  publish_collision_markers(msg->header.stamp);
}

bool HolonomicDriveController::is_point_in_collision_ring(double x, double y) const
{
  const bool in_outer =
    (x <= collision_outer_front_x_) && (x >= -collision_outer_back_x_) &&
    (std::abs(y) <= collision_outer_half_y_);
  const bool in_inner =
    (x <= collision_inner_front_x_) && (x >= -collision_inner_back_x_) &&
    (std::abs(y) <= collision_inner_half_y_);
  return in_outer && !in_inner;
}

void HolonomicDriveController::publish_collision_markers(const rclcpp::Time & stamp)
{
  if (!collision_marker_pub_) {
    return;
  }

  auto build_rect_marker = [&](int id, double front_x, double back_x, double hy, double z, bool outer)
    -> visualization_msgs::msg::Marker {
      visualization_msgs::msg::Marker m;
      m.header.stamp = stamp;
      m.header.frame_id = collision_frame_;
      m.ns = "collision_zone";
      m.id = id;
      m.type = visualization_msgs::msg::Marker::LINE_STRIP;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.scale.x = outer ? 0.02 : 0.015;
      m.pose.orientation.w = 1.0;
      m.color.a = 1.0;
      m.color.r = collision_blocked_ ? 1.0f : 0.0f;
      m.color.g = collision_blocked_ ? 0.0f : 1.0f;
      m.color.b = 0.0f;

      geometry_msgs::msg::Point p;
      p.z = z;
      p.x = front_x; p.y = hy; m.points.push_back(p);
      p.x = front_x; p.y = -hy; m.points.push_back(p);
      p.x = -back_x; p.y = -hy; m.points.push_back(p);
      p.x = -back_x; p.y = hy; m.points.push_back(p);
      p.x = front_x; p.y = hy; m.points.push_back(p);
      return m;
    };

  visualization_msgs::msg::Marker text;
  text.header.stamp = stamp;
  text.header.frame_id = collision_frame_;
  text.ns = "collision_zone";
  text.id = 2;
  text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  text.action = visualization_msgs::msg::Marker::ADD;
  text.pose.position.x = 0.0;
  text.pose.position.y = 0.0;
  text.pose.position.z = 0.08;
  text.pose.orientation.w = 1.0;
  text.scale.z = 0.08;
  text.color.a = 1.0;
  text.color.r = 1.0f;
  text.color.g = 1.0f;
  text.color.b = 1.0f;
  text.text = "collision_hits=" + std::to_string(collision_points_in_ring_) +
    " threshold=" + std::to_string(collision_min_points_) +
    (collision_blocked_ ? " BLOCKED" : " CLEAR") +
    " | outer(f=" + std::to_string(collision_outer_front_x_) +
    ",b=" + std::to_string(collision_outer_back_x_) + ")" +
    " inner(f=" + std::to_string(collision_inner_front_x_) +
    ",b=" + std::to_string(collision_inner_back_x_) + ")";

  visualization_msgs::msg::Marker front_axis;
  front_axis.header.stamp = stamp;
  front_axis.header.frame_id = collision_frame_;
  front_axis.ns = "collision_zone";
  front_axis.id = 3;
  front_axis.type = visualization_msgs::msg::Marker::LINE_STRIP;
  front_axis.action = visualization_msgs::msg::Marker::ADD;
  front_axis.scale.x = 0.02;
  front_axis.pose.orientation.w = 1.0;
  front_axis.color.a = 1.0;
  front_axis.color.r = 0.0f;
  front_axis.color.g = 0.6f;
  front_axis.color.b = 1.0f;
  geometry_msgs::msg::Point p;
  p.x = 0.0; p.y = 0.0; p.z = 0.03; front_axis.points.push_back(p);
  p.x = collision_outer_front_x_; p.y = 0.0; p.z = 0.03; front_axis.points.push_back(p);

  visualization_msgs::msg::Marker back_axis = front_axis;
  back_axis.id = 4;
  back_axis.color.r = 1.0f;
  back_axis.color.g = 0.5f;
  back_axis.color.b = 0.0f;
  back_axis.points.clear();
  p.x = 0.0; p.y = 0.0; p.z = 0.03; back_axis.points.push_back(p);
  p.x = -collision_outer_back_x_; p.y = 0.0; p.z = 0.03; back_axis.points.push_back(p);

  visualization_msgs::msg::MarkerArray arr;
  arr.markers.push_back(build_rect_marker(
    0, collision_outer_front_x_, collision_outer_back_x_, collision_outer_half_y_, 0.02, true));
  arr.markers.push_back(build_rect_marker(
    1, collision_inner_front_x_, collision_inner_back_x_, collision_inner_half_y_, 0.01, false));
  arr.markers.push_back(text);
  arr.markers.push_back(front_axis);
  arr.markers.push_back(back_axis);
  collision_marker_pub_->publish(arr);
}

double HolonomicDriveController::quaternion_to_yaw(
  const geometry_msgs::msg::Quaternion & quat)
{
  tf2::Quaternion tf_quat(quat.x, quat.y, quat.z, quat.w);
  tf2::Matrix3x3 matrix(tf_quat);
  double roll, pitch, yaw;
  matrix.getRPY(roll, pitch, yaw);
  return yaw;
}

double HolonomicDriveController::normalize_angle(double angle)
{
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

double HolonomicDriveController::clamp_velocity(
  double vel, double min_vel, double max_vel)
{
  double abs_vel = std::abs(vel);
  if (abs_vel > max_vel) {
    return (vel > 0) ? max_vel : -max_vel;
  } else if (abs_vel < min_vel && abs_vel > 1e-6) {
    return (vel > 0) ? min_vel : -min_vel;
  }
  return vel;
}

void HolonomicDriveController::control_timer_callback()
{
  if (!drive_active_ || !odom_received_) {
    return;
  }

  if (enable_collision_stop_ && collision_blocked_) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 500,
      "Collision stop active: points_in_ring=%d (threshold=%d)",
      collision_points_in_ring_, collision_min_points_);
    publish_zero_velocity();
    return;
  }

  // Compute time delta
  rclcpp::Time current_time = this->now();
  double dt = (current_time - prev_time_).seconds();
  if (dt < 1e-6) {
    dt = 0.02;
  }

  // --- Position error in world frame ---
  double error_x_world = goal_x_ - current_x_;
  double error_y_world = goal_y_ - current_y_;
  double distance = std::hypot(error_x_world, error_y_world);

  // --- Transform position error to robot body frame ---
  // Body frame: x = forward, y = left
  double cos_yaw = std::cos(current_yaw_);
  double sin_yaw = std::sin(current_yaw_);
  double error_x_body = cos_yaw * error_x_world + sin_yaw * error_y_world;
  double error_y_body = -sin_yaw * error_x_world + cos_yaw * error_y_world;

  // --- Yaw error ---
  double error_yaw = normalize_angle(goal_yaw_ - current_yaw_);

  // --- Check if goal is reached ---
  bool position_reached = (distance < position_tolerance_);
  bool orientation_reached = (std::abs(error_yaw) < orientation_tolerance_);

  if (position_reached && !final_yaw_tracking_active_) {
    final_yaw_tracking_active_ = true;
    RCLCPP_INFO(this->get_logger(),
      "Entered final yaw tracking mode (pos_err: %.4f m <= %.4f m)",
      distance, position_tolerance_);
  }

  const bool finish_by_orientation_after_latch = final_yaw_tracking_active_ && orientation_reached;

  if ((position_reached && orientation_reached) || finish_by_orientation_after_latch) {
    RCLCPP_INFO(this->get_logger(),
      "Goal reached! (pos_err: %.4f m, yaw_err: %.4f rad)",
      distance, std::abs(error_yaw));
    publish_zero_velocity();
    notify_drive_finish();
    reset_state();
    return;
  }

  // --- PD control in body frame ---
  double d_error_x = (error_x_body - prev_error_x_) / dt;
  double d_error_y = (error_y_body - prev_error_y_) / dt;
  double d_error_yaw = (error_yaw - prev_error_yaw_) / dt;

  double vx = Kp_linear_ * error_x_body + Kd_linear_ * d_error_x;
  double vy = Kp_linear_ * error_y_body + Kd_linear_ * d_error_y;
  double wz = Kp_angular_ * error_yaw + Kd_angular_ * d_error_yaw;

  // --- Scale linear velocities to respect max_linear_velocity as total ---
  double linear_speed = std::hypot(vx, vy);
  if (linear_speed > max_linear_velocity_) {
    double scale = max_linear_velocity_ / linear_speed;
    vx *= scale;
    vy *= scale;
  } else if (linear_speed < min_linear_velocity_ && !position_reached) {
    double scale = min_linear_velocity_ / linear_speed;
    vx *= scale;
    vy *= scale;
  }

  // --- In final yaw tracking mode, rotate-only regardless of small XY drift ---
  if (position_reached || final_yaw_tracking_active_) {
    vx = 0.0;
    vy = 0.0;
  }

  // --- Clamp angular velocity ---
  wz = clamp_velocity(wz, min_angular_velocity_, max_angular_velocity_);

  // --- If orientation is reached, only translate ---
  if (orientation_reached) {
    wz = 0.0;
  }

  // --- Publish velocity ---
  publish_velocity(vx, vy, wz);

  // --- Publish path visualization ---
  append_actual_path();

  // --- Update previous errors ---
  prev_error_x_ = error_x_body;
  prev_error_y_ = error_y_body;
  prev_error_yaw_ = error_yaw;
  prev_time_ = current_time;
}

void HolonomicDriveController::notify_drive_finish()
{
  if (!drive_finish_client_->service_is_ready()) {
    RCLCPP_DEBUG(this->get_logger(),
      "Service /drive_finish not available (continuing without notification)");
    return;
  }

  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  auto result = drive_finish_client_->async_send_request(request);

  if (result.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready) {
    auto response = result.get();
    if (response->success) {
      RCLCPP_INFO(this->get_logger(), "Drive finish notification sent successfully");
    } else {
      RCLCPP_DEBUG(this->get_logger(),
        "Drive finish notification returned false: %s", response->message.c_str());
    }
  } else {
    RCLCPP_DEBUG(this->get_logger(),
      "Drive finish service call timeout (service may be busy)");
  }
}

geometry_msgs::msg::Quaternion HolonomicDriveController::yaw_to_quaternion(double yaw)
{
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, yaw);
  geometry_msgs::msg::Quaternion quat;
  quat.x = q.x();
  quat.y = q.y();
  quat.z = q.z();
  quat.w = q.w();
  return quat;
}

geometry_msgs::msg::PoseStamped HolonomicDriveController::make_pose_stamped(
  double x, double y, double yaw, const rclcpp::Time & stamp)
{
  geometry_msgs::msg::PoseStamped pose;
  pose.header.stamp = stamp;
  pose.header.frame_id = path_frame_id_;
  pose.pose.position.x = x;
  pose.pose.position.y = y;
  pose.pose.position.z = 0.0;
  pose.pose.orientation = yaw_to_quaternion(yaw);
  return pose;
}

void HolonomicDriveController::publish_planned_path()
{
  planned_path_.header.stamp = this->now();
  planned_path_.header.frame_id = path_frame_id_;
  planned_path_.poses.clear();

  rclcpp::Time stamp = this->now();

  // Interpolate a straight line from start to goal
  constexpr int num_points = 20;
  for (int i = 0; i <= num_points; ++i) {
    double t = static_cast<double>(i) / static_cast<double>(num_points);
    double x = start_x_ + t * (goal_x_ - start_x_);
    double y = start_y_ + t * (goal_y_ - start_y_);

    // Interpolate yaw via shortest path
    double yaw_diff = normalize_angle(goal_yaw_ - start_yaw_);
    double yaw = normalize_angle(start_yaw_ + t * yaw_diff);

    planned_path_.poses.push_back(make_pose_stamped(x, y, yaw, stamp));
  }

  planned_path_pub_->publish(planned_path_);
}

void HolonomicDriveController::append_actual_path()
{
  rclcpp::Time stamp = this->now();
  actual_path_.header.stamp = stamp;
  actual_path_.poses.push_back(
    make_pose_stamped(current_x_, current_y_, current_yaw_, stamp));
  actual_path_pub_->publish(actual_path_);
}

void HolonomicDriveController::reset_state()
{
  drive_active_ = false;
  prev_error_x_ = 0.0;
  prev_error_y_ = 0.0;
  prev_error_yaw_ = 0.0;
  final_yaw_tracking_active_ = false;
}

void HolonomicDriveController::publish_zero_velocity()
{
  if (use_twist_stamped_ && cmd_vel_stamped_pub_) {
    auto twist_stamped = geometry_msgs::msg::TwistStamped();
    twist_stamped.header.stamp = this->now();
    twist_stamped.header.frame_id = "base_link";
    cmd_vel_stamped_pub_->publish(twist_stamped);
  } else if (!use_twist_stamped_ && cmd_vel_pub_) {
    auto twist = geometry_msgs::msg::Twist();
    cmd_vel_pub_->publish(twist);
  }
}

void HolonomicDriveController::publish_velocity(double vx, double vy, double angular_z)
{
  if (use_twist_stamped_ && cmd_vel_stamped_pub_) {
    auto twist_stamped = geometry_msgs::msg::TwistStamped();
    twist_stamped.header.stamp = this->now();
    twist_stamped.header.frame_id = "base_link";
    twist_stamped.twist.linear.x = vx;
    twist_stamped.twist.linear.y = vy;
    twist_stamped.twist.angular.z = angular_z;
    cmd_vel_stamped_pub_->publish(twist_stamped);
  } else if (!use_twist_stamped_ && cmd_vel_pub_) {
    auto twist = geometry_msgs::msg::Twist();
    twist.linear.x = vx;
    twist.linear.y = vy;
    twist.angular.z = angular_z;
    cmd_vel_pub_->publish(twist);
  }
}

}  // namespace v_marker_estimation

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<v_marker_estimation::HolonomicDriveController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
