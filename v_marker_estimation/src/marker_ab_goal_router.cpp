#include <cmath>
#include <chrono>
#include <future>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "example_interfaces/srv/add_two_ints.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"

namespace v_marker_estimation
{

class MarkerAbGoalRouter : public rclcpp::Node
{
public:
  explicit MarkerAbGoalRouter(const std::string & node_name = "marker_ab_goal_router")
  : Node(node_name),
    dock_pose_received_(false),
    odom_received_(false),
    has_target_selection_(false),
    selected_target_id_(1),
    goal_frozen_(false),
    yaw_republished_once_(false),
    enable_goal_freeze_(true),
    goal_freeze_distance_(0.12),
    yaw_republish_distance_(0.02),
    dock_pose_timeout_sec_(1.0),
    odom_timeout_sec_(1.0),
    goal_publish_rate_hz_(10.0),
    frame_id_("odom"),
    a_x_(0.0), a_y_(0.0), a_yaw_(0.0),
    b_x_(0.0), b_y_(0.0), b_yaw_(0.0),
    c_x_(0.0), c_y_(0.0), c_yaw_(0.0)
  {
    this->declare_parameter("dock_pose_topic", "/dock_pose");
    this->declare_parameter("odom_topic", "/odom");
    this->declare_parameter("goal_pose_topic", "/goal_pose_rviz");
    this->declare_parameter("goal_pose_viz_topic", "/goal_pose_rviz");
    this->declare_parameter("current_pose_marker_topic", "/current_pose_marker");
    this->declare_parameter("current_pose_publish_rate_hz", 10.0);
    this->declare_parameter("move_service_name", "/marker_ab_move");
    this->declare_parameter("move_add_two_ints_service_name", "/move_ab_move");
    this->declare_parameter("drive_finish_service_name", "/drive_finish");
    this->declare_parameter(
      "v_marker_roi_lock_reset_service_name", "/v_marker_roi_lock_reset");
    this->declare_parameter("frame_id", "odom");
    this->declare_parameter("enable_goal_freeze", true);
    this->declare_parameter("goal_freeze_distance", 0.12);
    this->declare_parameter("yaw_republish_distance", 0.02);
    this->declare_parameter("dock_pose_timeout_sec", 1.0);
    this->declare_parameter("odom_timeout_sec", 1.0);
    this->declare_parameter("goal_publish_rate_hz", 10.0);
    this->declare_parameter("a_x", 0.0);
    this->declare_parameter("a_y", 0.0);
    this->declare_parameter("a_yaw", 0.0);
    this->declare_parameter("b_x", 0.0);
    this->declare_parameter("b_y", 0.0);
    this->declare_parameter("b_yaw", 0.0);
    this->declare_parameter("c_x", 0.0);
    this->declare_parameter("c_y", 0.0);
    this->declare_parameter("c_yaw", 0.0);

    const std::string dock_pose_topic = this->get_parameter("dock_pose_topic").as_string();
    const std::string odom_topic = this->get_parameter("odom_topic").as_string();
    const std::string goal_pose_topic = this->get_parameter("goal_pose_topic").as_string();
    const std::string goal_pose_viz_topic =
      this->get_parameter("goal_pose_viz_topic").as_string();
    const std::string current_pose_marker_topic =
      this->get_parameter("current_pose_marker_topic").as_string();
    const double current_pose_publish_rate_hz =
      this->get_parameter("current_pose_publish_rate_hz").as_double();
    const std::string move_service_name = this->get_parameter("move_service_name").as_string();
    const std::string move_add_two_ints_service_name =
      this->get_parameter("move_add_two_ints_service_name").as_string();
    const std::string drive_finish_service_name =
      this->get_parameter("drive_finish_service_name").as_string();
    const std::string v_marker_roi_lock_reset_service_name =
      this->get_parameter("v_marker_roi_lock_reset_service_name").as_string();
    frame_id_ = this->get_parameter("frame_id").as_string();
    enable_goal_freeze_ = this->get_parameter("enable_goal_freeze").as_bool();
    goal_freeze_distance_ = this->get_parameter("goal_freeze_distance").as_double();
    yaw_republish_distance_ = this->get_parameter("yaw_republish_distance").as_double();
    dock_pose_timeout_sec_ = this->get_parameter("dock_pose_timeout_sec").as_double();
    odom_timeout_sec_ = this->get_parameter("odom_timeout_sec").as_double();
    goal_publish_rate_hz_ = this->get_parameter("goal_publish_rate_hz").as_double();
    a_x_ = this->get_parameter("a_x").as_double();
    a_y_ = this->get_parameter("a_y").as_double();
    a_yaw_ = this->get_parameter("a_yaw").as_double();
    b_x_ = this->get_parameter("b_x").as_double();
    b_y_ = this->get_parameter("b_y").as_double();
    b_yaw_ = this->get_parameter("b_yaw").as_double();
    c_x_ = this->get_parameter("c_x").as_double();
    c_y_ = this->get_parameter("c_y").as_double();
    c_yaw_ = this->get_parameter("c_yaw").as_double();

    const auto QOS_RKL10V = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();

    dock_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      dock_pose_topic, QOS_RKL10V,
      std::bind(&MarkerAbGoalRouter::dock_pose_callback, this, std::placeholders::_1));
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic, QOS_RKL10V,
      std::bind(&MarkerAbGoalRouter::odom_callback, this, std::placeholders::_1));
    goal_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      goal_pose_topic, QOS_RKL10V);
    goal_pose_viz_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      goal_pose_viz_topic, QOS_RKL10V);
    current_pose_marker_pub_ = this->create_publisher<geometry_msgs::msg::Pose2D>(
      current_pose_marker_topic, QOS_RKL10V);
    move_srv_ = this->create_service<std_srvs::srv::SetBool>(
      move_service_name,
      std::bind(&MarkerAbGoalRouter::move_callback, this,
        std::placeholders::_1, std::placeholders::_2));
    move_add_two_ints_srv_ = this->create_service<example_interfaces::srv::AddTwoInts>(
      move_add_two_ints_service_name,
      std::bind(&MarkerAbGoalRouter::move_add_two_ints_callback, this,
        std::placeholders::_1, std::placeholders::_2));
    drive_finish_srv_ = this->create_service<std_srvs::srv::Trigger>(
      drive_finish_service_name,
      std::bind(&MarkerAbGoalRouter::drive_finish_callback, this,
        std::placeholders::_1, std::placeholders::_2));
    v_marker_roi_lock_reset_client_ = this->create_client<std_srvs::srv::Trigger>(
      v_marker_roi_lock_reset_service_name);

    const double clamped_rate = std::max(1.0, current_pose_publish_rate_hz);
    const auto period = std::chrono::milliseconds(
      static_cast<int>(1000.0 / clamped_rate));
    current_pose_timer_ = this->create_wall_timer(
      period, std::bind(&MarkerAbGoalRouter::publish_current_pose_marker, this));
    const double clamped_goal_rate = std::max(1.0, goal_publish_rate_hz_);
    const auto goal_period = std::chrono::milliseconds(
      static_cast<int>(1000.0 / clamped_goal_rate));
    goal_publish_timer_ = this->create_wall_timer(
      goal_period, std::bind(&MarkerAbGoalRouter::publish_selected_goal, this));

    RCLCPP_INFO(this->get_logger(),
      "Marker AB Goal Router initialized (service: %s, finish_service: %s, frame: %s, goal_topic: %s, goal_viz_topic: %s, pose_topic: %s, goal_rate: %.1f Hz, freeze: %s(%.3f m), yaw_republish_dist: %.3f m)",
      move_service_name.c_str(), drive_finish_service_name.c_str(), frame_id_.c_str(),
      goal_pose_topic.c_str(), goal_pose_viz_topic.c_str(),
      current_pose_marker_topic.c_str(), clamped_goal_rate,
      enable_goal_freeze_ ? "on" : "off", goal_freeze_distance_, yaw_republish_distance_);
  }

private:
  static double normalize_angle(double angle)
  {
    while (angle > M_PI) {
      angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
      angle += 2.0 * M_PI;
    }
    return angle;
  }

  static double quaternion_to_yaw(const geometry_msgs::msg::Quaternion & quat)
  {
    tf2::Quaternion tf_quat(quat.x, quat.y, quat.z, quat.w);
    tf2::Matrix3x3 matrix(tf_quat);
    double roll, pitch, yaw;
    matrix.getRPY(roll, pitch, yaw);
    return yaw;
  }

  static geometry_msgs::msg::Quaternion yaw_to_quaternion(double yaw)
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

  void reset_v_marker_roi_lock()
  {
    if (!v_marker_roi_lock_reset_client_) {
      return;
    }
    if (!v_marker_roi_lock_reset_client_->wait_for_service(std::chrono::milliseconds(300))) {
      RCLCPP_WARN(this->get_logger(),
        "Reset skipped: /v_marker_roi_lock_reset service not available");
      return;
    }

    auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
    auto fut = v_marker_roi_lock_reset_client_->async_send_request(req);
    if (fut.wait_for(std::chrono::milliseconds(500)) != std::future_status::ready) {
      RCLCPP_WARN(this->get_logger(),
        "Reset skipped: roi_lock_reset request timeout");
      return;
    }

    RCLCPP_INFO(this->get_logger(),
      "Reset only ROI lock state by /v_marker_roi_lock_reset");
  }

  void dock_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    last_dock_pose_ = *msg;
    last_dock_pose_rx_time_ = this->now();
    dock_pose_received_ = true;
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    last_odom_ = *msg;
    last_odom_rx_time_ = this->now();
    odom_received_ = true;
  }

  bool set_target_and_publish_once(int target_id, std::string & reason_out)
  {
    if (target_id < 1 || target_id > 3) {
      reason_out = "invalid target id (use 1,2,3)";
      return false;
    }
    reset_v_marker_roi_lock();
    selected_target_id_ = target_id;
    has_target_selection_ = true;
    goal_frozen_ = false;
    yaw_republished_once_ = false;

    geometry_msgs::msg::PoseStamped goal;
    std::string reason;
    if (build_goal_pose(selected_target_id_, goal, reason)) {
      goal_pose_pub_->publish(goal);
      goal_pose_viz_pub_->publish(goal);
      reason_out = "ok";
      RCLCPP_INFO(this->get_logger(),
        "Selected target %d, started continuous goal publishing", selected_target_id_);
      return true;
    }
    reason_out = reason;
    return false;
  }

  void move_callback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response)
  {
    const int target_id = request->data ? 1 : 2;
    std::string reason;
    if (set_target_and_publish_once(target_id, reason)) {
      response->success = true;
      response->message = request->data ?
        "Target A selected and publishing continuously" :
        "Target B selected and publishing continuously";
      return;
    }

    // Selection is stored; timer will keep retrying and publish once data is valid.
    response->success = true;
    response->message = (request->data ? "Target A selected" : "Target B selected") +
      std::string(" (waiting: ") + reason + ")";
  }

  void move_add_two_ints_callback(
    const std::shared_ptr<example_interfaces::srv::AddTwoInts::Request> request,
    std::shared_ptr<example_interfaces::srv::AddTwoInts::Response> response)
  {
    const int target_id = static_cast<int>(request->a);
    std::string reason;
    if (set_target_and_publish_once(target_id, reason)) {
      response->sum = target_id;
      return;
    }
    response->sum = -1;
    RCLCPP_WARN(this->get_logger(),
      "move_add_two_ints failed for target %d: %s", target_id, reason.c_str());
  }

  void drive_finish_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    has_target_selection_ = false;
    goal_frozen_ = false;
    yaw_republished_once_ = false;
    response->success = true;
    response->message = "Stopped continuous goal publishing by drive_finish";
    RCLCPP_INFO(this->get_logger(), "Received drive_finish, stopped continuous goal publishing");
  }

  bool build_goal_pose(int target_id, geometry_msgs::msg::PoseStamped & goal, std::string & reason)
  {
    if (!dock_pose_received_) {
      reason = "dock_pose not received yet";
      return false;
    }
    if (!odom_received_) {
      reason = "odom not received yet";
      return false;
    }

    const double dock_age = (this->now() - last_dock_pose_rx_time_).seconds();
    if (dock_age > dock_pose_timeout_sec_) {
      reason = "dock_pose timeout";
      return false;
    }
    const double odom_age = (this->now() - last_odom_rx_time_).seconds();
    if (odom_age > odom_timeout_sec_) {
      reason = "odom timeout";
      return false;
    }

    double x_m = 0.0;
    double y_m = 0.0;
    double yaw_m = 0.0;
    if (target_id == 1) {
      x_m = a_x_;
      y_m = a_y_;
      yaw_m = a_yaw_;
    } else if (target_id == 2) {
      x_m = b_x_;
      y_m = b_y_;
      yaw_m = b_yaw_;
    } else if (target_id == 3) {
      x_m = c_x_;
      y_m = c_y_;
      yaw_m = c_yaw_;
    } else {
      reason = "invalid target id";
      return false;
    }

    const double x_d = last_dock_pose_.pose.position.x;
    const double y_d = last_dock_pose_.pose.position.y;
    const double yaw_d = quaternion_to_yaw(last_dock_pose_.pose.orientation);
    const double cos_d = std::cos(yaw_d);
    const double sin_d = std::sin(yaw_d);

    // marker frame -> odom frame
    const double x_o = x_d + cos_d * x_m - sin_d * y_m;
    const double y_o = y_d + sin_d * x_m + cos_d * y_m;
    const double yaw_o = normalize_angle(yaw_d + yaw_m);

    goal.header.stamp = this->now();
    goal.header.frame_id = frame_id_;
    goal.pose.position.x = x_o;
    goal.pose.position.y = y_o;
    goal.pose.position.z = 0.0;
    goal.pose.orientation = yaw_to_quaternion(yaw_o);
    reason.clear();
    return true;
  }

  void publish_selected_goal()
  {
    if (!has_target_selection_) {
      return;
    }

    geometry_msgs::msg::PoseStamped goal;
    std::string reason;
    if (!build_goal_pose(selected_target_id_, goal, reason)) {
      return;
    }

    const double cur_x = last_odom_.pose.pose.position.x;
    const double cur_y = last_odom_.pose.pose.position.y;

    // In frozen state, keep XY fixed and republish only once with updated yaw
    // when robot enters the near zone.
    if (goal_frozen_) {
      if (yaw_republished_once_) {
        return;
      }
      const double dist_to_frozen =
        std::hypot(frozen_goal_.pose.position.x - cur_x, frozen_goal_.pose.position.y - cur_y);
      if (dist_to_frozen <= yaw_republish_distance_) {
        frozen_goal_.header.stamp = this->now();
        frozen_goal_.pose.orientation = goal.pose.orientation;
        goal_pose_pub_->publish(frozen_goal_);
        goal_pose_viz_pub_->publish(frozen_goal_);
        yaw_republished_once_ = true;
        RCLCPP_INFO(this->get_logger(),
          "Republished frozen goal yaw once in near zone (dist=%.3f m, yaw_republish_dist=%.3f m)",
          dist_to_frozen, yaw_republish_distance_);
      }
      return;
    }

    if (enable_goal_freeze_) {
      const double dist_to_goal = std::hypot(goal.pose.position.x - cur_x, goal.pose.position.y - cur_y);
      if (dist_to_goal <= goal_freeze_distance_) {
        // Store and publish one final frozen goal, then stop further goal updates.
        frozen_goal_ = goal;
        goal_frozen_ = true;
        yaw_republished_once_ = false;
        goal_pose_pub_->publish(frozen_goal_);
        goal_pose_viz_pub_->publish(frozen_goal_);
        RCLCPP_INFO(this->get_logger(),
          "Goal frozen near end-phase; stop publishing updates (target %s, dist=%.3f m, freeze_dist=%.3f m)",
          selected_target_id_ == 1 ? "A" : (selected_target_id_ == 2 ? "B" : "C"),
          dist_to_goal, goal_freeze_distance_);
        return;
      }
    }

    goal_pose_pub_->publish(goal);
    goal_pose_viz_pub_->publish(goal);
  }

  void publish_current_pose_marker()
  {
    if (!dock_pose_received_ || !odom_received_) {
      return;
    }

    const double dock_age = (this->now() - last_dock_pose_rx_time_).seconds();
    const double odom_age = (this->now() - last_odom_rx_time_).seconds();
    if (dock_age > dock_pose_timeout_sec_ || odom_age > odom_timeout_sec_) {
      return;
    }

    const double x_d = last_dock_pose_.pose.position.x;
    const double y_d = last_dock_pose_.pose.position.y;
    const double yaw_d = quaternion_to_yaw(last_dock_pose_.pose.orientation);
    const double x_r = last_odom_.pose.pose.position.x;
    const double y_r = last_odom_.pose.pose.position.y;
    const double yaw_r = quaternion_to_yaw(last_odom_.pose.pose.orientation);

    const double dx = x_r - x_d;
    const double dy = y_r - y_d;
    const double cos_d = std::cos(yaw_d);
    const double sin_d = std::sin(yaw_d);

    // odom frame -> marker frame
    geometry_msgs::msg::Pose2D pose_m;
    pose_m.x = cos_d * dx + sin_d * dy;
    pose_m.y = -sin_d * dx + cos_d * dy;
    pose_m.theta = normalize_angle(yaw_r - yaw_d);
    current_pose_marker_pub_->publish(pose_m);
  }

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr dock_pose_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pose_viz_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr current_pose_marker_pub_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr v_marker_roi_lock_reset_client_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr move_srv_;
  rclcpp::Service<example_interfaces::srv::AddTwoInts>::SharedPtr move_add_two_ints_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr drive_finish_srv_;
  rclcpp::TimerBase::SharedPtr current_pose_timer_;
  rclcpp::TimerBase::SharedPtr goal_publish_timer_;

  geometry_msgs::msg::PoseStamped last_dock_pose_;
  nav_msgs::msg::Odometry last_odom_;
  geometry_msgs::msg::PoseStamped frozen_goal_;
  rclcpp::Time last_dock_pose_rx_time_;
  rclcpp::Time last_odom_rx_time_;

  bool dock_pose_received_;
  bool odom_received_;
  bool has_target_selection_;
  int selected_target_id_;
  bool goal_frozen_;
  bool yaw_republished_once_;
  bool enable_goal_freeze_;
  double goal_freeze_distance_;
  double yaw_republish_distance_;
  double dock_pose_timeout_sec_;
  double odom_timeout_sec_;
  double goal_publish_rate_hz_;
  std::string frame_id_;

  double a_x_;
  double a_y_;
  double a_yaw_;
  double b_x_;
  double b_y_;
  double b_yaw_;
  double c_x_;
  double c_y_;
  double c_yaw_;
};

}  // namespace v_marker_estimation

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<v_marker_estimation::MarkerAbGoalRouter>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

