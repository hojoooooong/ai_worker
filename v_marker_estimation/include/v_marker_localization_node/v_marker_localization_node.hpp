// Copyright 2023 ROBOTIS CO., LTD.
// Authors: Sungho Woo

#ifndef V_MARKER_LOCALIZATION_NODE__V_MARKER_LOCALIZATION_NODE_HPP_
#define V_MARKER_LOCALIZATION_NODE__V_MARKER_LOCALIZATION_NODE_HPP_

#include <cmath>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "std_srvs/srv/trigger.hpp"

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include "v_marker_estimation/msg/laser_sample.hpp"
#include "v_marker_estimation/msg/laser_sample_set.hpp"
#include "v_marker_estimation/msg/laser_sample_set_list.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

struct LinearFit
{
  double slope;
  double intercept;
  bool valid;
};

struct InitialPoseSample
{
  double x;
  double y;
  double yaw;
};

class VMarkerLocalizationNode : public rclcpp::Node
{
public:
  explicit VMarkerLocalizationNode(const std::string & node_name);
  virtual ~VMarkerLocalizationNode();

private:
  // Callbacks
  void clusters_callback(
    const v_marker_estimation::msg::LaserSampleSetList::SharedPtr clusters);

  void activation_callback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response);
  void reset_roi_lock_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);

  // Filter pipeline
  std::vector<v_marker_estimation::msg::LaserSampleSet> stitch_filter(
    const std::vector<v_marker_estimation::msg::LaserSampleSet> & src_clusters);

  std::vector<v_marker_estimation::msg::LaserSampleSet> dist_filter(
    const std::vector<v_marker_estimation::msg::LaserSampleSet> & src_clusters);

  std::vector<v_marker_estimation::msg::LaserSampleSet> relative_filter(
    const std::vector<v_marker_estimation::msg::LaserSampleSet> & src_clusters);

  std::vector<v_marker_estimation::msg::LaserSampleSet> tracking_filter(
    const std::vector<v_marker_estimation::msg::LaserSampleSet> & src_clusters);

  std::vector<v_marker_estimation::msg::LaserSampleSet> detect_lvl_pattern(
    const std::vector<v_marker_estimation::msg::LaserSampleSet> & src_clusters);

  // Cluster segmentation
  std::vector<v_marker_estimation::msg::LaserSampleSet> segment_cluster(
    const v_marker_estimation::msg::LaserSampleSet & cl);

  // Dock pose calculation
  bool calculate_dock_pose(
    const std::vector<v_marker_estimation::msg::LaserSampleSet> & src_clusters);

  // Helper functions
  double normalize_angle(double val, double min_val, double max_val);
  double circular_mean(const std::vector<double> & angles) const;
  bool build_stable_initial_pose(
    double cand_x, double cand_y, double cand_yaw,
    double & out_x, double & out_y, double & out_yaw);
  LinearFit fit_line(
    const std::vector<double> & x, const std::vector<double> & y);

  // Visualization helpers (build markers into pending_markers_ array)
  void add_point_marker(
    const std::string & ns, int id, double px, double py,
    double r, double g, double b, double scale);
  void add_segment_marker(
    const std::string & ns, int id,
    const v_marker_estimation::msg::LaserSampleSet & cl,
    double r, double g, double b);
  void flush_markers();  // publish all pending markers at once

  // ROS2 interfaces
  rclcpp::Subscription<v_marker_estimation::msg::LaserSampleSetList>::SharedPtr clusters_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr dock_pose_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr relative_marker_pub_;
  visualization_msgs::msg::MarkerArray pending_markers_;  // accumulated for batch publish
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr activation_server_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_roi_lock_server_;

  // State variables
  bool is_running_;
  bool activation_flag_;
  bool noise_once_trigger_;
  bool has_previous_detection_;

  // Robot state (from clusters->robot_pose: x, y, theta_as_z)
  double rx_, ry_, r_theta_;

  // Parameters
  std::string frame_id_;
  double split_threshold_;
  double angle_min_deg_;         // relative filter: min angle [deg]
  double angle_max_deg_;         // relative filter: max angle [deg]
  double angle_min_rad_;         // cached radian conversion
  double angle_max_rad_;         // cached radian conversion
  double detection_range_;
  double mark_size_lvl_0_;   // Marker size: left L [m]
  double mark_size_lvl_1_;   // Marker size: center-left V [m]
  double mark_size_lvl_2_;   // Marker size: center-right V [m]
  double mark_size_lvl_3_;   // Marker size: right L [m]
  double x_offset_;
  double y_offset_;
  double stitch_euclidian_dist_;
  double outer_parallel_max_angle_deg_;   // max allowed angle between outer L segments [deg]
  double outer_parallel_min_cos_;         // cached cos(max_angle)
  double seg_size_margin_ratio_;          // per-segment size tolerance ratio (e.g. 0.10 = +/-10%)
  bool inner_v_must_face_outward_;        // true: reject patterns whose V points toward robot
  double noise_xy_threshold_;
  double noise_theta_threshold_;
  double filter_log_period_sec_;
  int warn_log_throttle_ms_;
  int initial_pose_sample_count_;
  int initial_pose_min_inliers_;
  double initial_pose_max_xy_dev_;
  double initial_pose_max_yaw_dev_rad_;
  bool enable_first_detection_roi_lock_;
  double first_detection_roi_radius_;

  // Dock pose state
  double dp_x_g_, dp_y_g_;      // global dock pose (for tracking_filter)
  double dock_theta_;
  double pre_global_x_, pre_global_y_, pre_global_theta_;
  bool first_detection_roi_locked_;
  double first_detection_roi_center_x_g_;
  double first_detection_roi_center_y_g_;
  bool initial_pose_ready_;
  std::vector<InitialPoseSample> initial_pose_samples_;

  // EMA (Exponential Moving Average) for smoothing output pose
  double ema_alpha_;             // smoothing factor: 0.0=freeze, 1.0=no smoothing
  double ema_global_x_, ema_global_y_, ema_global_theta_;
  bool ema_initialized_;

  // Mutex
  std::shared_ptr<std::mutex> cluster_mutex_;

  // Header for timestamps
  std_msgs::msg::Header current_header_;

  // Throttled logging
  rclcpp::Time last_log_time_;
};

#endif  // V_MARKER_LOCALIZATION_NODE__V_MARKER_LOCALIZATION_NODE_HPP_
