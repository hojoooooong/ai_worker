// Copyright 2023 ROBOTIS CO., LTD.
// Authors: Sungho Woo
//
// V-Marker Localization Node (ROS2 C++ port of Python LVEdgeFilter)
// Detects L-V-V-L shaped 4-segment reflective markers from lidar clusters
// and estimates the docking pose.

#include <v_marker_localization_node/v_marker_localization_node.hpp>
#include <array>

// =============================================================================
// Constructor / Destructor
// =============================================================================

VMarkerLocalizationNode::VMarkerLocalizationNode(const std::string & node_name)
: Node(node_name)
{
  // Declare parameters
  this->declare_parameter("frame_id", "odom");
  this->declare_parameter("detection_range", 2.0);
  this->declare_parameter("angle_min_deg", -180.0);
  this->declare_parameter("angle_max_deg", 180.0);
  this->declare_parameter("split_threshold", 0.5);
  this->declare_parameter("stitch_euclidian_dist", 0.04);
  this->declare_parameter("outer_parallel_max_angle_deg", 20.0);
  this->declare_parameter("seg_size_margin_ratio", 0.10);
  this->declare_parameter("inner_v_must_face_outward", true);
  this->declare_parameter("x_offset", 0.0);
  this->declare_parameter("y_offset", 0.0);
  this->declare_parameter("mark_size_lvl_0", 0.25);
  this->declare_parameter("mark_size_lvl_1", 0.115);
  this->declare_parameter("mark_size_lvl_2", 0.115);
  this->declare_parameter("mark_size_lvl_3", 0.25);
  this->declare_parameter("noise_xy_threshold", 0.2);
  this->declare_parameter("noise_theta_threshold", 0.174);
  this->declare_parameter("filter_log_period_sec", 5.0);
  this->declare_parameter("warn_log_throttle_ms", 3000);
  this->declare_parameter("initial_pose_sample_count", 10);
  this->declare_parameter("initial_pose_min_inliers", 6);
  this->declare_parameter("initial_pose_max_xy_dev", 0.08);
  this->declare_parameter("initial_pose_max_yaw_dev_deg", 15.0);
  this->declare_parameter("ema_alpha", 0.3);
  this->declare_parameter("enable_first_detection_roi_lock", true);
  this->declare_parameter("first_detection_roi_radius", 0.35);
  this->declare_parameter("qos_depth", 1);

  // Get parameters
  frame_id_ = this->get_parameter("frame_id").as_string();
  detection_range_ = this->get_parameter("detection_range").as_double();
  angle_min_deg_ = this->get_parameter("angle_min_deg").as_double();
  angle_max_deg_ = this->get_parameter("angle_max_deg").as_double();
  angle_min_rad_ = angle_min_deg_ * M_PI / 180.0;
  angle_max_rad_ = angle_max_deg_ * M_PI / 180.0;
  split_threshold_ = this->get_parameter("split_threshold").as_double();
  stitch_euclidian_dist_ = this->get_parameter("stitch_euclidian_dist").as_double();
  outer_parallel_max_angle_deg_ = this->get_parameter("outer_parallel_max_angle_deg").as_double();
  outer_parallel_min_cos_ = std::cos(outer_parallel_max_angle_deg_ * M_PI / 180.0);
  seg_size_margin_ratio_ = this->get_parameter("seg_size_margin_ratio").as_double();
  seg_size_margin_ratio_ = std::max(0.0, std::min(seg_size_margin_ratio_, 0.49));
  inner_v_must_face_outward_ = this->get_parameter("inner_v_must_face_outward").as_bool();
  x_offset_ = this->get_parameter("x_offset").as_double();
  y_offset_ = this->get_parameter("y_offset").as_double();
  mark_size_lvl_0_ = this->get_parameter("mark_size_lvl_0").as_double();
  mark_size_lvl_1_ = this->get_parameter("mark_size_lvl_1").as_double();
  mark_size_lvl_2_ = this->get_parameter("mark_size_lvl_2").as_double();
  mark_size_lvl_3_ = this->get_parameter("mark_size_lvl_3").as_double();
  noise_xy_threshold_ = this->get_parameter("noise_xy_threshold").as_double();
  noise_theta_threshold_ = this->get_parameter("noise_theta_threshold").as_double();
  filter_log_period_sec_ = this->get_parameter("filter_log_period_sec").as_double();
  warn_log_throttle_ms_ = this->get_parameter("warn_log_throttle_ms").as_int();
  initial_pose_sample_count_ = this->get_parameter("initial_pose_sample_count").as_int();
  initial_pose_min_inliers_ = this->get_parameter("initial_pose_min_inliers").as_int();
  initial_pose_max_xy_dev_ = this->get_parameter("initial_pose_max_xy_dev").as_double();
  initial_pose_max_yaw_dev_rad_ =
    this->get_parameter("initial_pose_max_yaw_dev_deg").as_double() * M_PI / 180.0;
  ema_alpha_ = this->get_parameter("ema_alpha").as_double();
  enable_first_detection_roi_lock_ =
    this->get_parameter("enable_first_detection_roi_lock").as_bool();
  first_detection_roi_radius_ = this->get_parameter("first_detection_roi_radius").as_double();
  int qos_depth = static_cast<int>(this->get_parameter("qos_depth").as_int());

  // Log parameters
  RCLCPP_INFO(this->get_logger(), "V-Marker Localization Node initialized");
  RCLCPP_INFO(this->get_logger(), "  frame_id: %s", frame_id_.c_str());
  RCLCPP_INFO(this->get_logger(), "  detection_range: %.2f m", detection_range_);
  RCLCPP_INFO(this->get_logger(), "  angle range: [%.1f, %.1f] deg", angle_min_deg_, angle_max_deg_);
  RCLCPP_INFO(this->get_logger(), "  split_threshold: %.4f m", split_threshold_);
  RCLCPP_INFO(this->get_logger(), "  outer_parallel_max_angle: %.1f deg",
    outer_parallel_max_angle_deg_);
  RCLCPP_INFO(this->get_logger(), "  seg_size_margin_ratio: %.2f", seg_size_margin_ratio_);
  RCLCPP_INFO(this->get_logger(), "  inner_v_must_face_outward: %s",
    inner_v_must_face_outward_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "  marker sizes: [%.3f, %.3f, %.3f, %.3f] m",
    mark_size_lvl_0_, mark_size_lvl_1_, mark_size_lvl_2_, mark_size_lvl_3_);
  RCLCPP_INFO(this->get_logger(),
    "  initial_pose_stabilizer: samples=%d, min_inliers=%d, max_xy_dev=%.3f m, max_yaw_dev=%.1f deg",
    initial_pose_sample_count_, initial_pose_min_inliers_, initial_pose_max_xy_dev_,
    initial_pose_max_yaw_dev_rad_ * 180.0 / M_PI);
  RCLCPP_INFO(this->get_logger(),
    "  logging: filter_period=%.1f s, warn_throttle=%d ms",
    filter_log_period_sec_, warn_log_throttle_ms_);
  RCLCPP_INFO(this->get_logger(), "  first_detection_roi_lock: %s (radius=%.2f)",
    enable_first_detection_roi_lock_ ? "on" : "off",
    first_detection_roi_radius_);

  const auto QOS_RKL10V =
    rclcpp::QoS(rclcpp::KeepLast(qos_depth)).reliable().durability_volatile();

  // Subscribers
  clusters_sub_ = this->create_subscription<v_marker_estimation::msg::LaserSampleSetList>(
    "clusters", QOS_RKL10V,
    std::bind(&VMarkerLocalizationNode::clusters_callback, this, std::placeholders::_1));

  // Publishers
  dock_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
    "dock_pose", QOS_RKL10V);
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "v_marker_viz", QOS_RKL10V);
  relative_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "relative_filtered_viz", QOS_RKL10V);

  // Services
  activation_server_ = this->create_service<std_srvs::srv::SetBool>(
    "v_marker_localization_activate",
    std::bind(&VMarkerLocalizationNode::activation_callback, this,
      std::placeholders::_1, std::placeholders::_2));
  reset_roi_lock_server_ = this->create_service<std_srvs::srv::Trigger>(
    "v_marker_roi_lock_reset",
    std::bind(&VMarkerLocalizationNode::reset_roi_lock_callback, this,
      std::placeholders::_1, std::placeholders::_2));

  // Initialize state
  is_running_ = false;
  activation_flag_ = false;
  noise_once_trigger_ = false;
  has_previous_detection_ = false;
  rx_ = 0.0;
  ry_ = 0.0;
  r_theta_ = 0.0;
  dp_x_g_ = 0.0;
  dp_y_g_ = 0.0;
  dock_theta_ = 0.0;
  pre_global_x_ = 0.0;
  pre_global_y_ = 0.0;
  pre_global_theta_ = 0.0;
  ema_global_x_ = 0.0;
  ema_global_y_ = 0.0;
  ema_global_theta_ = 0.0;
  ema_initialized_ = false;
  first_detection_roi_locked_ = false;
  first_detection_roi_center_x_g_ = 0.0;
  first_detection_roi_center_y_g_ = 0.0;
  initial_pose_ready_ = false;
  initial_pose_samples_.clear();
  last_log_time_ = this->now();
  cluster_mutex_ = std::make_shared<std::mutex>();
}

VMarkerLocalizationNode::~VMarkerLocalizationNode()
{
}

// =============================================================================
// Service Callback
// =============================================================================

void VMarkerLocalizationNode::activation_callback(
  const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
  std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
  activation_flag_ = request->data;

  if (activation_flag_) {
    is_running_ = true;
    noise_once_trigger_ = false;
    has_previous_detection_ = false;
    ema_initialized_ = false;
    dp_x_g_ = 0.0;
    dp_y_g_ = 0.0;
    // Re-read dynamic parameters on activation
    angle_min_deg_ = this->get_parameter("angle_min_deg").as_double();
    angle_max_deg_ = this->get_parameter("angle_max_deg").as_double();
    angle_min_rad_ = angle_min_deg_ * M_PI / 180.0;
    angle_max_rad_ = angle_max_deg_ * M_PI / 180.0;
    x_offset_ = this->get_parameter("x_offset").as_double();
    y_offset_ = this->get_parameter("y_offset").as_double();
    detection_range_ = this->get_parameter("detection_range").as_double();
    outer_parallel_max_angle_deg_ =
      this->get_parameter("outer_parallel_max_angle_deg").as_double();
    outer_parallel_min_cos_ = std::cos(outer_parallel_max_angle_deg_ * M_PI / 180.0);
    seg_size_margin_ratio_ = this->get_parameter("seg_size_margin_ratio").as_double();
    seg_size_margin_ratio_ = std::max(0.0, std::min(seg_size_margin_ratio_, 0.49));
    inner_v_must_face_outward_ = this->get_parameter("inner_v_must_face_outward").as_bool();
    mark_size_lvl_0_ = this->get_parameter("mark_size_lvl_0").as_double();
    mark_size_lvl_1_ = this->get_parameter("mark_size_lvl_1").as_double();
    mark_size_lvl_2_ = this->get_parameter("mark_size_lvl_2").as_double();
    mark_size_lvl_3_ = this->get_parameter("mark_size_lvl_3").as_double();
    ema_alpha_ = this->get_parameter("ema_alpha").as_double();
    filter_log_period_sec_ = this->get_parameter("filter_log_period_sec").as_double();
    warn_log_throttle_ms_ = this->get_parameter("warn_log_throttle_ms").as_int();
    initial_pose_sample_count_ = this->get_parameter("initial_pose_sample_count").as_int();
    initial_pose_min_inliers_ = this->get_parameter("initial_pose_min_inliers").as_int();
    initial_pose_max_xy_dev_ = this->get_parameter("initial_pose_max_xy_dev").as_double();
    initial_pose_max_yaw_dev_rad_ =
      this->get_parameter("initial_pose_max_yaw_dev_deg").as_double() * M_PI / 180.0;
    enable_first_detection_roi_lock_ =
      this->get_parameter("enable_first_detection_roi_lock").as_bool();
    first_detection_roi_radius_ = this->get_parameter("first_detection_roi_radius").as_double();
    first_detection_roi_locked_ = false;
    first_detection_roi_center_x_g_ = 0.0;
    first_detection_roi_center_y_g_ = 0.0;
    initial_pose_ready_ = false;
    initial_pose_samples_.clear();

    RCLCPP_INFO(this->get_logger(), "V-Marker localization activated");
  } else {
    is_running_ = false;
    RCLCPP_INFO(this->get_logger(), "V-Marker localization deactivated");
  }

  response->success = true;
  response->message = activation_flag_ ?
    "V-Marker localization activated" : "V-Marker localization deactivated";
}

void VMarkerLocalizationNode::reset_roi_lock_callback(
  const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
  std::lock_guard<std::mutex> lock(*cluster_mutex_);
  noise_once_trigger_ = false;
  has_previous_detection_ = false;
  first_detection_roi_locked_ = false;
  first_detection_roi_center_x_g_ = 0.0;
  first_detection_roi_center_y_g_ = 0.0;
  initial_pose_ready_ = false;
  initial_pose_samples_.clear();
  response->success = true;
  response->message = "ROI lock, noise, tracking, initial-pose reset";
  RCLCPP_INFO(this->get_logger(),
    "ROI lock + noise_once_trigger + has_previous_detection reset by service request");
}

// =============================================================================
// Main Cluster Callback
// =============================================================================

void VMarkerLocalizationNode::clusters_callback(
  const v_marker_estimation::msg::LaserSampleSetList::SharedPtr clusters)
{
  if (!is_running_ || !activation_flag_) {
    return;
  }

  std::lock_guard<std::mutex> lock(*cluster_mutex_);

  current_header_ = clusters->header;
  rx_ = clusters->robot_pose.x;
  ry_ = clusters->robot_pose.y;
  r_theta_ = clusters->robot_pose.z;

  auto src_clusters = clusters->sampleset_list;

  // Clear pending markers for this frame
  pending_markers_.markers.clear();

  // Filter pipeline
  auto stitched = stitch_filter(src_clusters);
  auto dist_filtered = dist_filter(stitched);
  auto relative_filtered = relative_filter(dist_filtered);
  auto tracking_filtered = tracking_filter(relative_filtered);
  auto edge_filtered = detect_lvl_pattern(tracking_filtered);

  if (!edge_filtered.empty()) {
    calculate_dock_pose(edge_filtered);
  }

  // Throttled filter summary log
  auto now = this->now();
  if ((now - last_log_time_).seconds() >= std::max(0.5, filter_log_period_sec_)) {
    RCLCPP_INFO(this->get_logger(),
      "[filter] input=%zu stitch=%zu dist=%zu rel=%zu track=%zu detect=%zu",
      src_clusters.size(), stitched.size(), dist_filtered.size(),
      relative_filtered.size(), tracking_filtered.size(),
      edge_filtered.empty() ? 0u : edge_filtered.size());
    last_log_time_ = now;
  }

  // Publish all accumulated markers in a single MarkerArray message
  flush_markers();
}

// =============================================================================
// Filter: Stitch Filter
// Merges clusters split at the lidar scan boundary (±π)
// =============================================================================

std::vector<v_marker_estimation::msg::LaserSampleSet>
VMarkerLocalizationNode::stitch_filter(
  const std::vector<v_marker_estimation::msg::LaserSampleSet> & src_clusters)
{
  if (src_clusters.size() < 2) {
    return src_clusters;
  }

  const auto & clusters = src_clusters;  // avoid copy; merge builds new vector if needed

  struct EdgeInfo {
    v_marker_estimation::msg::LaserSample min_sample;
    v_marker_estimation::msg::LaserSample max_sample;
    double min_angle;
    double max_angle;
    size_t cluster_idx;
  };

  std::vector<EdgeInfo> edges;

  for (size_t i = 0; i < clusters.size(); i++) {
    double min_angle = 5.0, max_angle = -5.0;
    v_marker_estimation::msg::LaserSample min_sample, max_sample;

    for (const auto & sample : clusters[i].sample_set) {
      double angle = std::atan2(sample.y, sample.x);
      if (angle > max_angle) {
        max_angle = angle;
        max_sample = sample;
      }
      if (angle < min_angle) {
        min_angle = angle;
        min_sample = sample;
      }
    }

    // Check if cluster has samples near ±π boundary
    if (std::abs(normalize_angle(min_angle - M_PI, -M_PI, M_PI)) < 0.15 ||
      std::abs(normalize_angle(max_angle - M_PI, -M_PI, M_PI)) < 0.15)
    {
      EdgeInfo info;
      info.min_sample = min_sample;
      info.max_sample = max_sample;
      info.min_angle = min_angle;
      info.max_angle = max_angle;
      info.cluster_idx = i;
      edges.push_back(info);
    }
  }

  if (edges.size() < 2) {
    return clusters;
  }

  // Try to merge first and last edge clusters (scan wrapping)
  double dx_stitch = edges.front().min_sample.x - edges.back().max_sample.x;
  double dy_stitch = edges.front().min_sample.y - edges.back().max_sample.y;
  double dist = std::sqrt(dx_stitch * dx_stitch + dy_stitch * dy_stitch);

  if (dist < stitch_euclidian_dist_) {
    size_t first_idx = edges.front().cluster_idx;
    size_t last_idx = edges.back().cluster_idx;

    v_marker_estimation::msg::LaserSampleSet merged;
    merged.cluster_id = clusters[first_idx].cluster_id;

    // Reverse both clusters (to maintain sample ordering after merge)
    auto first_samples = clusters[first_idx].sample_set;
    auto last_samples = clusters[last_idx].sample_set;
    std::reverse(first_samples.begin(), first_samples.end());
    std::reverse(last_samples.begin(), last_samples.end());

    double sum_x = 0.0, sum_y = 0.0;
    for (const auto & s : first_samples) {
      merged.sample_set.push_back(s);
      sum_x += s.x;
      sum_y += s.y;
    }
    for (const auto & s : last_samples) {
      merged.sample_set.push_back(s);
      sum_x += s.x;
      sum_y += s.y;
    }

    size_t n = merged.sample_set.size();
    merged.mean_p.x = sum_x / static_cast<double>(n);
    merged.mean_p.y = sum_y / static_cast<double>(n);
    merged.mean_p.z = 0.0;

    // Build result: remove original two, add merged
    std::vector<v_marker_estimation::msg::LaserSampleSet> result;
    for (size_t i = 0; i < clusters.size(); i++) {
      if (i != first_idx && i != last_idx) {
        result.push_back(clusters[i]);
      }
    }
    result.push_back(merged);
    return result;
  }

  return clusters;
}

// =============================================================================
// Filter: Distance Filter
// Removes clusters beyond detection_range
// =============================================================================

std::vector<v_marker_estimation::msg::LaserSampleSet>
VMarkerLocalizationNode::dist_filter(
  const std::vector<v_marker_estimation::msg::LaserSampleSet> & src_clusters)
{
  std::vector<v_marker_estimation::msg::LaserSampleSet> result;

  for (const auto & cl : src_clusters) {
    double dx = cl.mean_p.x;
    double dy = cl.mean_p.y;
    double dist = std::sqrt(dx * dx + dy * dy);
    if (dist <= detection_range_) {
      result.push_back(cl);
    }
  }

  return result;
}

// =============================================================================
// Filter: Relative Angle Filter
// Keeps only clusters whose bearing angle falls within [angle_min, angle_max] degrees
// =============================================================================

std::vector<v_marker_estimation::msg::LaserSampleSet>
VMarkerLocalizationNode::relative_filter(
  const std::vector<v_marker_estimation::msg::LaserSampleSet> & src_clusters)
{
  std::vector<v_marker_estimation::msg::LaserSampleSet> result;
  visualization_msgs::msg::MarkerArray rel_markers;

  int marker_id = 0;
  for (const auto & cl : src_clusters) {
    double angle = std::atan2(cl.mean_p.y, cl.mean_p.x);  // bearing angle [rad]
    if (angle >= angle_min_rad_ && angle <= angle_max_rad_) {
      result.push_back(cl);

      // Visualize each passing cluster (orange)
      visualization_msgs::msg::Marker m;
      m.header.stamp = current_header_.stamp;
      m.header.frame_id = current_header_.frame_id;
      m.ns = "cluster";
      m.id = marker_id++;
      m.type = visualization_msgs::msg::Marker::SPHERE_LIST;
      m.lifetime = rclcpp::Duration::from_seconds(1.0);
      m.scale.x = 0.03;
      m.scale.y = 0.03;
      m.scale.z = 0.03;
      m.color.a = 0.7;
      m.color.r = 1.0;
      m.color.g = 0.5;
      m.color.b = 0.0;
      for (const auto & sample : cl.sample_set) {
        geometry_msgs::msg::Point p;
        p.x = sample.x;
        p.y = sample.y;
        p.z = 0.0;
        m.points.push_back(p);
      }
      rel_markers.markers.push_back(std::move(m));
    }
  }

  if (!rel_markers.markers.empty()) {
    relative_marker_pub_->publish(rel_markers);
  }

  return result;
}

// =============================================================================
// Filter: Tracking Filter
// Uses previous detection to narrow search area
// =============================================================================

std::vector<v_marker_estimation::msg::LaserSampleSet>
VMarkerLocalizationNode::tracking_filter(
  const std::vector<v_marker_estimation::msg::LaserSampleSet> & src_clusters)
{
  // If no previous detection, pass through all clusters
  if (!has_previous_detection_) {
    return src_clusters;
  }

  // Transform global reference center to local frame.
  // If ROI lock is enabled and initialized, use the first-detection locked center.
  // Otherwise keep existing behavior using latest tracked dock pose.
  double ref_x_g = dp_x_g_;
  double ref_y_g = dp_y_g_;
  if (enable_first_detection_roi_lock_ && first_detection_roi_locked_) {
    ref_x_g = first_detection_roi_center_x_g_;
    ref_y_g = first_detection_roi_center_y_g_;
  }

  std::vector<v_marker_estimation::msg::LaserSampleSet> result;
  double roi_radius;
  if (enable_first_detection_roi_lock_ && first_detection_roi_locked_) {
    roi_radius = std::max(0.05, first_detection_roi_radius_);
  } else {
    // Total marker width (sum of all 4 segments) as search bound
    double marker_total_width = mark_size_lvl_0_ + mark_size_lvl_1_ + mark_size_lvl_2_ + mark_size_lvl_3_;
    roi_radius = marker_total_width * 2.0;
  }

  // Visualize tracking ROI circle in global frame (goal-pose centered)
  visualization_msgs::msg::Marker roi;
  roi.header.stamp = current_header_.stamp;
  roi.header.frame_id = frame_id_;
  roi.ns = "tracking_roi";
  roi.id = 700;
  roi.type = visualization_msgs::msg::Marker::LINE_STRIP;
  roi.action = visualization_msgs::msg::Marker::ADD;
  roi.lifetime = rclcpp::Duration::from_seconds(1.0);
  roi.scale.x = 0.015;
  roi.color.a = 1.0;
  roi.color.r = 0.0;
  roi.color.g = enable_first_detection_roi_lock_ && first_detection_roi_locked_ ? 1.0 : 0.6;
  roi.color.b = enable_first_detection_roi_lock_ && first_detection_roi_locked_ ? 0.2 : 1.0;
  constexpr int kCircleSegments = 48;
  for (int i = 0; i <= kCircleSegments; ++i) {
    const double a = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(kCircleSegments);
    geometry_msgs::msg::Point p;
    p.z = 0.04;
    p.x = ref_x_g + roi_radius * std::cos(a);
    p.y = ref_y_g + roi_radius * std::sin(a);
    roi.points.push_back(p);
  }
  pending_markers_.markers.push_back(roi);

  // ROI center marker
  visualization_msgs::msg::Marker roi_center;
  roi_center.header.stamp = current_header_.stamp;
  roi_center.header.frame_id = frame_id_;
  roi_center.ns = "tracking_roi";
  roi_center.id = 701;
  roi_center.type = visualization_msgs::msg::Marker::SPHERE;
  roi_center.action = visualization_msgs::msg::Marker::ADD;
  roi_center.lifetime = rclcpp::Duration::from_seconds(1.0);
  roi_center.scale.x = 0.05;
  roi_center.scale.y = 0.05;
  roi_center.scale.z = 0.05;
  roi_center.color.a = 0.95;
  roi_center.color.r = enable_first_detection_roi_lock_ && first_detection_roi_locked_ ? 0.1 : 0.0;
  roi_center.color.g = enable_first_detection_roi_lock_ && first_detection_roi_locked_ ? 1.0 : 0.8;
  roi_center.color.b = enable_first_detection_roi_lock_ && first_detection_roi_locked_ ? 0.1 : 1.0;
  roi_center.pose.position.x = ref_x_g;
  roi_center.pose.position.y = ref_y_g;
  roi_center.pose.position.z = 0.05;
  pending_markers_.markers.push_back(roi_center);

  // ROI status text marker
  visualization_msgs::msg::Marker roi_text;
  roi_text.header.stamp = current_header_.stamp;
  roi_text.header.frame_id = frame_id_;
  roi_text.ns = "tracking_roi";
  roi_text.id = 702;
  roi_text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  roi_text.action = visualization_msgs::msg::Marker::ADD;
  roi_text.lifetime = rclcpp::Duration::from_seconds(1.0);
  roi_text.scale.z = 0.06;
  roi_text.color.a = 1.0;
  roi_text.color.r = 1.0;
  roi_text.color.g = 1.0;
  roi_text.color.b = 1.0;
  roi_text.pose.position.x = ref_x_g;
  roi_text.pose.position.y = ref_y_g;
  roi_text.pose.position.z = 0.14;
  roi_text.text =
    std::string("ROI(GLOBAL) ") +
    ((enable_first_detection_roi_lock_ && first_detection_roi_locked_) ? "LOCKED" : "SEARCH") +
    " r=" + std::to_string(roi_radius);
  pending_markers_.markers.push_back(roi_text);

  for (const auto & cl : src_clusters) {
    // Cluster center is in robot-local frame; convert to global for global ROI test.
    const double cl_x_g =
      cl.mean_p.x * std::cos(r_theta_) - cl.mean_p.y * std::sin(r_theta_) + rx_;
    const double cl_y_g =
      cl.mean_p.x * std::sin(r_theta_) + cl.mean_p.y * std::cos(r_theta_) + ry_;
    const double dist_x = cl_x_g - ref_x_g;
    const double dist_y = cl_y_g - ref_y_g;
    if (std::sqrt(dist_x * dist_x + dist_y * dist_y) <= roi_radius) {
      result.push_back(cl);
    }
  }

  // In ROI-lock mode, do NOT pass through outside clusters once locked.
  if (enable_first_detection_roi_lock_ && first_detection_roi_locked_) {
    if (result.empty()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), warn_log_throttle_ms_,
        "tracking_filter: ROI locked and no cluster inside circle (radius=%.3f)",
        roi_radius);
    }
    return result;
  }

  // If tracking lost all clusters, fall back to passing all through (legacy behavior)
  if (result.empty()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), warn_log_throttle_ms_,
      "tracking_filter: lost tracking (radius=%.3f), passing all %zu clusters",
      roi_radius, src_clusters.size());
    return src_clusters;
  }

  return result;
}

// =============================================================================
// Segment Cluster (IEPF - Iterative End-Point Fit)
// Recursively splits a cluster by finding the point with maximum perpendicular
// distance from the line connecting segment endpoints. Naturally finds the most
// significant geometric transitions first (e.g., V vertex before L-V edges).
// =============================================================================

// Helper: perpendicular distance from point (px, py) to line through (x1,y1)-(x2,y2)
static inline double perpendicular_distance(
  double px, double py, double x1, double y1, double x2, double y2)
{
  double dx = x2 - x1;
  double dy = y2 - y1;
  double line_len_sq = dx * dx + dy * dy;
  if (line_len_sq < 1e-12) {
    return std::sqrt((px - x1) * (px - x1) + (py - y1) * (py - y1));
  }
  return std::abs(dy * px - dx * py + x2 * y1 - y2 * x1) / std::sqrt(line_len_sq);
}

// IEPF recursive split (free function to avoid std::function heap overhead)
static void iepf_recursive(
  const std::vector<v_marker_estimation::msg::LaserSample> & samples,
  int start, int end, double threshold,
  std::vector<int> & split_indices)
{
  if (end - start < 2) return;

  const double x1 = samples[start].x, y1 = samples[start].y;
  const double x2 = samples[end].x, y2 = samples[end].y;

  double max_dist = 0.0;
  int max_idx = -1;

  for (int i = start + 1; i < end; i++) {
    double dist = perpendicular_distance(samples[i].x, samples[i].y, x1, y1, x2, y2);
    if (dist > max_dist) {
      max_dist = dist;
      max_idx = i;
    }
  }

  if (max_dist > threshold && max_idx >= 0) {
    split_indices.push_back(max_idx);
    iepf_recursive(samples, start, max_idx, threshold, split_indices);
    iepf_recursive(samples, max_idx, end, threshold, split_indices);
  }
}

// Compute principal axes (u: major axis, v: minor axis) for a group of clusters.
// This makes pattern classification robust to marker rotation in the lidar frame.
static void compute_principal_axes(
  const std::vector<v_marker_estimation::msg::LaserSampleSet> & clusters,
  double & cx, double & cy, double & ux, double & uy, double & vx, double & vy)
{
  cx = 0.0;
  cy = 0.0;
  double count = 0.0;

  for (const auto & cl : clusters) {
    for (const auto & s : cl.sample_set) {
      cx += s.x;
      cy += s.y;
      count += 1.0;
    }
  }

  if (count < 2.0) {
    // Fallback axes
    ux = 1.0;
    uy = 0.0;
    vx = 0.0;
    vy = 1.0;
    return;
  }

  cx /= count;
  cy /= count;

  double cxx = 0.0;
  double cxy = 0.0;
  double cyy = 0.0;
  for (const auto & cl : clusters) {
    for (const auto & s : cl.sample_set) {
      const double dx = s.x - cx;
      const double dy = s.y - cy;
      cxx += dx * dx;
      cxy += dx * dy;
      cyy += dy * dy;
    }
  }
  cxx /= count;
  cxy /= count;
  cyy /= count;

  const double trace = cxx + cyy;
  const double det = cxx * cyy - cxy * cxy;
  const double disc = std::max(0.0, trace * trace - 4.0 * det);
  const double lambda_max = 0.5 * (trace + std::sqrt(disc));

  // Eigenvector for lambda_max
  double ex = cxy;
  double ey = lambda_max - cxx;
  const double n = std::sqrt(ex * ex + ey * ey);

  if (n < 1e-10) {
    // Degenerate covariance, pick axis-aligned fallback
    if (cxx >= cyy) {
      ux = 1.0;
      uy = 0.0;
    } else {
      ux = 0.0;
      uy = 1.0;
    }
  } else {
    ux = ex / n;
    uy = ey / n;
  }

  // Minor axis: perpendicular to major axis
  vx = -uy;
  vy = ux;
}

std::vector<v_marker_estimation::msg::LaserSampleSet>
VMarkerLocalizationNode::segment_cluster(
  const v_marker_estimation::msg::LaserSampleSet & cl)
{
  const int n = static_cast<int>(cl.sample_set.size());
  if (n < 4) {
    return {};
  }

  // --- IEPF: find split indices recursively ---
  std::vector<int> split_indices;
  split_indices.reserve(8);  // typical V-marker has ~3 splits
  iepf_recursive(cl.sample_set, 0, n - 1, split_threshold_, split_indices);

  std::sort(split_indices.begin(), split_indices.end());

  // --- Build sub-clusters from split indices ---
  const size_t num_segments = split_indices.size() + 1;
  std::vector<v_marker_estimation::msg::LaserSampleSet> list_cs;
  list_cs.reserve(num_segments);

  // Build boundaries: [0, split0, split1, ..., n]
  int seg_start = 0;
  auto build_segment = [&](int seg_end) {
    if (seg_end <= seg_start) return;

    v_marker_estimation::msg::LaserSampleSet segment;
    segment.cluster_id = cl.cluster_id;
    segment.sample_set.reserve(seg_end - seg_start);

    double x_sum = 0.0, y_sum = 0.0;
    for (int i = seg_start; i < seg_end; i++) {
      segment.sample_set.push_back(cl.sample_set[i]);
      x_sum += cl.sample_set[i].x;
      y_sum += cl.sample_set[i].y;
    }

    const double cnt = static_cast<double>(segment.sample_set.size());
    segment.mean_p.x = x_sum / cnt;
    segment.mean_p.y = y_sum / cnt;
    segment.mean_p.z = 0.0;

    list_cs.push_back(std::move(segment));
    seg_start = seg_end;
  };

  for (int idx : split_indices) {
    build_segment(idx);
  }
  build_segment(n);

  return list_cs;
}

// =============================================================================
// Detect LVL Pattern
// For each cluster, segments into sub-clusters and validates the
// L-V-V-L 4-segment pattern by checking marker sizes (±25% tolerance).
// If multiple valid candidates, selects the closest one.
// =============================================================================

std::vector<v_marker_estimation::msg::LaserSampleSet>
VMarkerLocalizationNode::detect_lvl_pattern(
  const std::vector<v_marker_estimation::msg::LaserSampleSet> & src_clusters)
{
  std::vector<v_marker_estimation::msg::LaserSampleSet> result;
  std::vector<std::vector<v_marker_estimation::msg::LaserSampleSet>> candidate_groups;
  std::vector<double> candidate_distances;

  for (const auto & cl : src_clusters) {
    auto sub_clusters = segment_cluster(cl);

    // L-V-V-L pattern requires at least 4 sub-clusters
    if (sub_clusters.size() < 4) {
      continue;
    }

    // Sliding window: try every consecutive 4 sub-clusters
    for (size_t w = 0; w + 3 < sub_clusters.size(); w++) {
      std::vector<v_marker_estimation::msg::LaserSampleSet> window_clusters = {
        sub_clusters[w], sub_clusters[w + 1], sub_clusters[w + 2], sub_clusters[w + 3]
      };

      // Rotation-invariant ordering:
      // 1) estimate principal axes of the 4-segment candidate
      // 2) sort segments along major axis u
      // 3) evaluate V-depth on minor axis v
      double cx = 0.0, cy = 0.0;
      double ux = 1.0, uy = 0.0, vx = 0.0, vy = 1.0;
      compute_principal_axes(window_clusters, cx, cy, ux, uy, vx, vy);

      std::array<double, 4> proj_u {};
      std::array<double, 4> proj_v {};
      for (size_t i = 0; i < 4; i++) {
        const double dx = window_clusters[i].mean_p.x - cx;
        const double dy = window_clusters[i].mean_p.y - cy;
        proj_u[i] = dx * ux + dy * uy;
        proj_v[i] = dx * vx + dy * vy;
      }

      std::vector<size_t> indices = {0, 1, 2, 3};
      std::sort(indices.begin(), indices.end(),
        [&proj_u](size_t a, size_t b) {
          return proj_u[a] < proj_u[b];
        });

      std::vector<v_marker_estimation::msg::LaserSampleSet> sorted_clusters;
      sorted_clusters.reserve(4);
      std::array<double, 4> sorted_v {};
      size_t sorted_idx = 0;
      for (auto idx : indices) {
        sorted_clusters.push_back(window_clusters[idx]);
        sorted_v[sorted_idx] = proj_v[idx];
        sorted_idx++;
      }

      // Check each segment has enough samples
      bool valid = true;
      for (int i = 0; i < 4; i++) {
        if (sorted_clusters[i].sample_set.size() < 2) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        continue;
      }

      // --- Validation 1: Total marker span check ---
      // Measure total span from leftmost to rightmost point across all 4 segments
      double min_y = 1e9, max_y = -1e9;
      double min_x = 1e9, max_x = -1e9;
      for (int i = 0; i < 4; i++) {
        for (const auto & s : sorted_clusters[i].sample_set) {
          if (s.y < min_y) min_y = s.y;
          if (s.y > max_y) max_y = s.y;
          if (s.x < min_x) min_x = s.x;
          if (s.x > max_x) max_x = s.x;
        }
      }
      double total_span = std::sqrt((max_x - min_x) * (max_x - min_x) +
                                     (max_y - min_y) * (max_y - min_y));
      double expected_total = mark_size_lvl_0_ + mark_size_lvl_1_ + mark_size_lvl_2_ + mark_size_lvl_3_;

      if (total_span < expected_total * 0.50 || total_span > expected_total * 1.5) {
        continue;
      }

      // --- Validation 2: Rotation-invariant V-shape classification ---
      // Inner segments should lie on the same side of outer baseline (minor axis v)
      // and show deeper indentation than outers.
      const double v_outer_center = (sorted_v[0] + sorted_v[3]) / 2.0;
      const double v_inner_1 = sorted_v[1] - v_outer_center;
      const double v_inner_2 = sorted_v[2] - v_outer_center;
      const double v_outer_0 = sorted_v[0] - v_outer_center;
      const double v_outer_3 = sorted_v[3] - v_outer_center;

      // Inner pair must be on the same side to form a V valley/ridge.
      if (v_inner_1 * v_inner_2 <= 0.0) {
        continue;
      }

      // Inner pair should be deeper than outer pair (with a small margin).
      const double inner_depth = std::min(std::abs(v_inner_1), std::abs(v_inner_2));
      const double outer_depth = std::max(std::abs(v_outer_0), std::abs(v_outer_3));
      if (inner_depth <= outer_depth + 0.005) {
        continue;
      }

      // Optional directionality check:
      // V should face outward from robot (inner center farther from robot than outer center).
      if (inner_v_must_face_outward_) {
        const double outer_cx = (sorted_clusters[0].mean_p.x + sorted_clusters[3].mean_p.x) / 2.0;
        const double outer_cy = (sorted_clusters[0].mean_p.y + sorted_clusters[3].mean_p.y) / 2.0;
        const double inner_cx = (sorted_clusters[1].mean_p.x + sorted_clusters[2].mean_p.x) / 2.0;
        const double inner_cy = (sorted_clusters[1].mean_p.y + sorted_clusters[2].mean_p.y) / 2.0;

        // Compare projection of V protrusion onto outward radial direction from robot.
        // dot > 0: inner pair is farther from robot than outer pair (outward-facing V).
        const double protrude_x = inner_cx - outer_cx;
        const double protrude_y = inner_cy - outer_cy;
        const double outward_x = outer_cx;
        const double outward_y = outer_cy;
        const double outward_norm = std::sqrt(outward_x * outward_x + outward_y * outward_y);
        if (outward_norm < 1e-6) {
          continue;
        }
        const double dot = protrude_x * outward_x + protrude_y * outward_y;
        if (dot <= 0.0) {
          continue;
        }
      }

      // --- Validation 3: Segment size ratios ---
      // Outer L segments should be similar size to each other; inner V segments similar to each other
      double seg_sizes[4];
      for (int i = 0; i < 4; i++) {
        const auto & sc = sorted_clusters[i];
        double dx = sc.sample_set.front().x - sc.sample_set.back().x;
        double dy = sc.sample_set.front().y - sc.sample_set.back().y;
        seg_sizes[i] = std::sqrt(dx * dx + dy * dy);
      }

      // Each segment length should match configured L-V-V-L size within +/-10%.
      const double expected_sizes[4] = {
        mark_size_lvl_0_, mark_size_lvl_1_, mark_size_lvl_2_, mark_size_lvl_3_
      };
      bool size_match = true;
      for (int i = 0; i < 4; i++) {
        const double min_len = expected_sizes[i] * (1.0 - seg_size_margin_ratio_);
        const double max_len = expected_sizes[i] * (1.0 + seg_size_margin_ratio_);
        if (seg_sizes[i] < min_len || seg_sizes[i] > max_len) {
          size_match = false;
          break;
        }
      }
      if (!size_match) {
        continue;
      }

      // --- Validation 3.5: Outer segment parallelism ---
      // Use direction vectors of outer segments (0,3). Ignore sign with abs(dot).
      const auto & outer0 = sorted_clusters[0].sample_set;
      const auto & outer3 = sorted_clusters[3].sample_set;
      const double o0_dx = outer0.back().x - outer0.front().x;
      const double o0_dy = outer0.back().y - outer0.front().y;
      const double o3_dx = outer3.back().x - outer3.front().x;
      const double o3_dy = outer3.back().y - outer3.front().y;
      const double o0_len = std::sqrt(o0_dx * o0_dx + o0_dy * o0_dy);
      const double o3_len = std::sqrt(o3_dx * o3_dx + o3_dy * o3_dy);
      if (o0_len < 1e-6 || o3_len < 1e-6) {
        continue;
      }
      const double parallel_cos =
        std::abs((o0_dx * o3_dx + o0_dy * o3_dy) / (o0_len * o3_len));
      if (parallel_cos < outer_parallel_min_cos_) {
        continue;
      }

      // L segments ratio (seg0 vs seg3): should be within 3:1
      double l_ratio = (seg_sizes[0] > seg_sizes[3]) ?
        seg_sizes[0] / std::max(seg_sizes[3], 0.001) :
        seg_sizes[3] / std::max(seg_sizes[0], 0.001);
      // V segments ratio (seg1 vs seg2): should be within 3:1
      double v_ratio = (seg_sizes[1] > seg_sizes[2]) ?
        seg_sizes[1] / std::max(seg_sizes[2], 0.001) :
        seg_sizes[2] / std::max(seg_sizes[1], 0.001);

      if (l_ratio > 3.0 || v_ratio > 3.0) {
        continue;
      }

      candidate_groups.push_back(std::move(sorted_clusters));
      const double center_x =
        (candidate_groups.back()[0].mean_p.x + candidate_groups.back()[3].mean_p.x) / 2.0;
      const double center_y =
        (candidate_groups.back()[0].mean_p.y + candidate_groups.back()[3].mean_p.y) / 2.0;
      double dist_from_robot = std::sqrt(center_x * center_x + center_y * center_y);
      candidate_distances.push_back(dist_from_robot);
      break;  // Found valid pattern in this cluster, move to next cluster
    }
  }

  if (candidate_groups.empty()) {
    return result;
  }

  // Select closest candidate
  size_t closest_idx = 0;
  if (candidate_groups.size() > 1) {
    double min_dist = candidate_distances[0];
    for (size_t i = 1; i < candidate_distances.size(); i++) {
      if (candidate_distances[i] < min_dist) {
        min_dist = candidate_distances[i];
        closest_idx = i;
      }
    }
  }

  result = candidate_groups[closest_idx];

  // Add L-V-V-L segment markers with distinct colors per segment:
  // seg[0]=L (green), seg[1]=V (cyan), seg[2]=V (magenta), seg[3]=L (yellow)
  double seg_colors[4][3] = {
    {0.0, 1.0, 0.0},   // L0: green
    {0.0, 1.0, 1.0},   // V1: cyan
    {1.0, 0.0, 1.0},   // V2: magenta
    {1.0, 1.0, 0.0}    // L3: yellow
  };
  for (int i = 0; i < 4; i++) {
    add_segment_marker("segment", i, result[i],
      seg_colors[i][0], seg_colors[i][1], seg_colors[i][2]);
  }

  return result;
}

// =============================================================================
// Linear Regression Helper (OLS)
// =============================================================================

LinearFit VMarkerLocalizationNode::fit_line(
  const std::vector<double> & x, const std::vector<double> & y)
{
  LinearFit fit = {0.0, 0.0, false};
  size_t n = x.size();
  if (n < 2 || n != y.size()) {
    return fit;
  }

  double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
  for (size_t i = 0; i < n; i++) {
    sum_x += x[i];
    sum_y += y[i];
    sum_xy += x[i] * y[i];
    sum_xx += x[i] * x[i];
  }

  double dn = static_cast<double>(n);
  double denom = dn * sum_xx - sum_x * sum_x;
  if (std::abs(denom) < 1e-10) {
    return fit;
  }

  fit.slope = (dn * sum_xy - sum_x * sum_y) / denom;
  fit.intercept = (sum_y - fit.slope * sum_x) / dn;
  fit.valid = true;
  return fit;
}

// =============================================================================
// Calculate Dock Pose
// Uses line fitting on inner V clusters for intersection point,
// and OLS regression on outer L clusters for dock angle.
// =============================================================================

bool VMarkerLocalizationNode::calculate_dock_pose(
  const std::vector<v_marker_estimation::msg::LaserSampleSet> & src_clusters)
{
  if (src_clusters.size() != 4) {
    return false;
  }

  // --- Step 1: Fit lines to each of the 4 sub-clusters ---
  std::vector<double> slopes(4), intercepts(4);
  for (int c = 0; c < 4; c++) {
    std::vector<double> xs, ys;
    const auto & samples = src_clusters[c].sample_set;
    // Use all samples except the last (matching Python: range(len-1))
    for (size_t j = 0; j + 1 < samples.size(); j++) {
      xs.push_back(samples[j].x);
      ys.push_back(samples[j].y);
    }
    auto fit = fit_line(xs, ys);
    if (!fit.valid) {
      RCLCPP_DEBUG(this->get_logger(), "Line fitting failed for sub-cluster %d", c);
      return false;
    }
    slopes[c] = fit.slope;
    intercepts[c] = fit.intercept;
  }

  // --- Step 2: Calculate intersection of inner V clusters [1] and [2] ---
  double slope_diff = slopes[1] - slopes[2];
  if (std::abs(slope_diff) < 1e-10) {
    RCLCPP_DEBUG(this->get_logger(),
      "Inner V clusters are parallel, cannot find intersection");
    return false;
  }

  double pre_x = (intercepts[2] - intercepts[1]) / slope_diff;
  double pre_y = ((slopes[1] + slopes[2]) * pre_x + intercepts[1] + intercepts[2]) / 2.0;

  // Add intersection point marker to pending array
  add_point_marker("cross_point", 1, pre_x, pre_y, 0.0, 0.0, 1.0, 0.05);

  // --- Step 3: Calculate dock angle from outer L clusters [0] and [3] ---
  // Single-pass: accumulate sums for OLS and variance simultaneously
  double n_pts = 0.0;
  double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0, sum_yy = 0.0;

  for (int ci : {0, 3}) {
    for (const auto & sample : src_clusters[ci].sample_set) {
      n_pts += 1.0;
      sum_x += sample.x;
      sum_y += sample.y;
      sum_xy += sample.x * sample.y;
      sum_xx += sample.x * sample.x;
      sum_yy += sample.y * sample.y;
    }
  }

  if (n_pts < 2.0) {
    return false;
  }

  // OLS regression slopes (also used for verticality via variance = E[x^2] - E[x]^2)
  double denom_x = n_pts * sum_xx - sum_x * sum_x;  // = n^2 * var_x
  double denom_y = n_pts * sum_yy - sum_y * sum_y;  // = n^2 * var_y
  if (std::abs(denom_x) < 1e-10 && std::abs(denom_y) < 1e-10) {
    return false;
  }

  // Verticality from variance ratio (no second pass needed)
  double var_x = denom_x / (n_pts * n_pts);
  double var_y = denom_y / (n_pts * n_pts);
  double std_x = std::sqrt(std::max(var_x, 0.0));
  double std_y = std::sqrt(std::max(var_y, 0.0));
  double verticality = (std_x > 1e-10) ? std_y / std_x : 999.0;

  double slope_based_x = (std::abs(denom_x) > 1e-10) ?
    (n_pts * sum_xy - sum_x * sum_y) / denom_x : 0.0;
  double slope_based_y = (std::abs(denom_y) > 1e-10) ?
    (n_pts * sum_xy - sum_x * sum_y) / denom_y : 0.0;

  // Calculate dock angle (choosing regression axis based on data verticality)
  double dock_th;
  if (verticality > 1.5) {
    dock_th = -std::atan2(-slope_based_y, -1.0);
  } else {
    dock_th = -std::atan2(-1.0, -slope_based_x);
  }

  // Resolve 180-deg ambiguity by choosing heading aligned with robot->marker direction.
  // This prevents arrow flip toward robot when marker is behind.
  double th0 = normalize_angle(dock_th, -M_PI, M_PI);
  double th1 = normalize_angle(dock_th + M_PI, -M_PI, M_PI);
  double marker_bearing = std::atan2(pre_y, pre_x);
  double score0 = std::cos(normalize_angle(th0 - marker_bearing, -M_PI, M_PI));
  double score1 = std::cos(normalize_angle(th1 - marker_bearing, -M_PI, M_PI));
  dock_theta_ = (score0 >= score1) ? th0 : th1;

  // --- Step 4: Apply offset ---
  double cos_th = std::cos(dock_theta_);
  double sin_th = std::sin(dock_theta_);
  double x_coordinate = pre_x + cos_th * x_offset_ + sin_th * y_offset_;
  double y_coordinate = pre_y + sin_th * x_offset_ - cos_th * y_offset_;

  // --- Step 5: Transform to global frame ---
  double global_dp_x =
    x_coordinate * std::cos(r_theta_) - y_coordinate * std::sin(r_theta_) + rx_;
  double global_dp_y =
    x_coordinate * std::sin(r_theta_) + y_coordinate * std::cos(r_theta_) + ry_;

  double global_theta = r_theta_ + dock_theta_;

  // --- Step 6: Initial pose stabilizer (collect multiple detections and fuse robustly) ---
  if (!initial_pose_ready_) {
    double stable_x = global_dp_x;
    double stable_y = global_dp_y;
    double stable_theta = global_theta;
    if (!build_stable_initial_pose(global_dp_x, global_dp_y, global_theta,
      stable_x, stable_y, stable_theta))
    {
      return true;
    }
    global_dp_x = stable_x;
    global_dp_y = stable_y;
    global_theta = stable_theta;
    initial_pose_ready_ = true;
    RCLCPP_INFO(this->get_logger(),
      "Initial dock pose stabilized with %zu samples", initial_pose_samples_.size());
  }

  // --- Step 7: Noise filter (reject large jumps) ---
  if (!noise_once_trigger_) {
    pre_global_x_ = global_dp_x;
    pre_global_y_ = global_dp_y;
    pre_global_theta_ = global_theta;
    noise_once_trigger_ = true;
  }

  double dx_noise = pre_global_x_ - global_dp_x;
  double dy_noise = pre_global_y_ - global_dp_y;
  double pre_dist = std::sqrt(dx_noise * dx_noise + dy_noise * dy_noise);

  if (pre_dist > noise_xy_threshold_) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), warn_log_throttle_ms_,
      "XY change exceeds %.2f m (%.3f m), skipping update",
      noise_xy_threshold_, pre_dist);
    return true;
  }
  pre_global_x_ = global_dp_x;
  pre_global_y_ = global_dp_y;

  double pre_theta_diff = std::abs(normalize_angle(global_theta - pre_global_theta_, -M_PI, M_PI));
  if (pre_theta_diff > noise_theta_threshold_) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), warn_log_throttle_ms_,
      "Theta change exceeds %.1f deg (%.1f deg), skipping update",
      noise_theta_threshold_ * 180.0 / M_PI, pre_theta_diff * 180.0 / M_PI);
    return true;
  }
  pre_global_theta_ = global_theta;

  // --- Step 8: Update tracking variables for tracking filter ---
  dp_x_g_ = global_dp_x;
  dp_y_g_ = global_dp_y;
  has_previous_detection_ = true;
  if (enable_first_detection_roi_lock_ && !first_detection_roi_locked_) {
    first_detection_roi_center_x_g_ = global_dp_x;
    first_detection_roi_center_y_g_ = global_dp_y;
    first_detection_roi_locked_ = true;
    RCLCPP_INFO(this->get_logger(),
      "First detection ROI locked at global (%.3f, %.3f) with radius %.3f",
      first_detection_roi_center_x_g_, first_detection_roi_center_y_g_,
      first_detection_roi_radius_);
  }

  // --- Step 9: EMA (Exponential Moving Average) smoothing ---
  if (!ema_initialized_) {
    ema_global_x_ = global_dp_x;
    ema_global_y_ = global_dp_y;
    ema_global_theta_ = global_theta;
    ema_initialized_ = true;
  } else {
    ema_global_x_ = ema_alpha_ * global_dp_x + (1.0 - ema_alpha_) * ema_global_x_;
    ema_global_y_ = ema_alpha_ * global_dp_y + (1.0 - ema_alpha_) * ema_global_y_;

    // Angle EMA: handle wrapping via shortest angular distance
    double angle_diff = normalize_angle(global_theta - ema_global_theta_, -M_PI, M_PI);
    ema_global_theta_ = normalize_angle(
      ema_global_theta_ + ema_alpha_ * angle_diff, -M_PI, M_PI);
  }

  // --- Step 10: Publish smoothed dock pose ---
  geometry_msgs::msg::PoseStamped dp;
  dp.header.frame_id = frame_id_;
  dp.header.stamp = current_header_.stamp;
  dp.pose.position.x = ema_global_x_;
  dp.pose.position.y = ema_global_y_;

  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, ema_global_theta_);
  tf2::convert(q, dp.pose.orientation);

  dock_pose_pub_->publish(dp);

  return true;
}

// =============================================================================
// Helper Functions
// =============================================================================

double VMarkerLocalizationNode::normalize_angle(
  double val, double min_val, double max_val)
{
  double norm = 0.0;
  if (val >= min_val) {
    norm = min_val + std::fmod((val - min_val), (max_val - min_val));
  } else {
    norm = max_val - std::fmod((min_val - val), (max_val - min_val));
  }
  return norm;
}

double VMarkerLocalizationNode::circular_mean(const std::vector<double> & angles) const
{
  if (angles.empty()) {
    return 0.0;
  }
  double s = 0.0;
  double c = 0.0;
  for (const double a : angles) {
    s += std::sin(a);
    c += std::cos(a);
  }
  return std::atan2(s, c);
}

bool VMarkerLocalizationNode::build_stable_initial_pose(
  double cand_x, double cand_y, double cand_yaw,
  double & out_x, double & out_y, double & out_yaw)
{
  initial_pose_samples_.push_back({cand_x, cand_y, cand_yaw});
  const size_t required = static_cast<size_t>(std::max(3, initial_pose_sample_count_));
  if (initial_pose_samples_.size() < required) {
    return false;
  }

  std::vector<double> xs;
  std::vector<double> ys;
  std::vector<double> yaws;
  xs.reserve(initial_pose_samples_.size());
  ys.reserve(initial_pose_samples_.size());
  yaws.reserve(initial_pose_samples_.size());
  for (const auto & s : initial_pose_samples_) {
    xs.push_back(s.x);
    ys.push_back(s.y);
    yaws.push_back(s.yaw);
  }

  auto median = [](std::vector<double> v) -> double {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
  const double mx = median(xs);
  const double my = median(ys);
  const double myaw = circular_mean(yaws);

  std::vector<InitialPoseSample> inliers;
  inliers.reserve(initial_pose_samples_.size());
  for (const auto & s : initial_pose_samples_) {
    const double dxy = std::hypot(s.x - mx, s.y - my);
    const double dyaw = std::abs(normalize_angle(s.yaw - myaw, -M_PI, M_PI));
    if (dxy <= initial_pose_max_xy_dev_ && dyaw <= initial_pose_max_yaw_dev_rad_) {
      inliers.push_back(s);
    }
  }

  const size_t min_inliers = static_cast<size_t>(std::max(3, initial_pose_min_inliers_));
  if (inliers.size() < min_inliers) {
    // Keep sliding window size bounded
    if (initial_pose_samples_.size() > required) {
      initial_pose_samples_.erase(initial_pose_samples_.begin());
    }
    return false;
  }

  double sx = 0.0;
  double sy = 0.0;
  std::vector<double> iyaws;
  iyaws.reserve(inliers.size());
  for (const auto & s : inliers) {
    sx += s.x;
    sy += s.y;
    iyaws.push_back(s.yaw);
  }

  out_x = sx / static_cast<double>(inliers.size());
  out_y = sy / static_cast<double>(inliers.size());
  out_yaw = circular_mean(iyaws);
  initial_pose_samples_.clear();
  return true;
}

// =============================================================================
// Visualization Helpers
// =============================================================================

void VMarkerLocalizationNode::add_point_marker(
  const std::string & ns, int id, double px, double py,
  double r, double g, double b, double scale)
{
  visualization_msgs::msg::Marker m;
  m.header.stamp = current_header_.stamp;
  m.header.frame_id = current_header_.frame_id;
  m.ns = ns;
  m.id = id;
  m.type = visualization_msgs::msg::Marker::SPHERE;
  m.lifetime = rclcpp::Duration::from_seconds(1.0);
  m.scale.x = scale;
  m.scale.y = scale;
  m.scale.z = scale;
  m.color.a = 1.0;
  m.color.r = r;
  m.color.g = g;
  m.color.b = b;
  m.pose.position.x = px;
  m.pose.position.y = py;
  m.pose.position.z = 0.05;

  pending_markers_.markers.push_back(m);
}

void VMarkerLocalizationNode::add_segment_marker(
  const std::string & ns, int id,
  const v_marker_estimation::msg::LaserSampleSet & cl,
  double r, double g, double b)
{
  visualization_msgs::msg::Marker m;
  m.header.stamp = current_header_.stamp;
  m.header.frame_id = current_header_.frame_id;
  m.ns = ns;
  m.id = id;
  m.type = visualization_msgs::msg::Marker::SPHERE_LIST;
  m.lifetime = rclcpp::Duration::from_seconds(1.0);
  m.scale.x = 0.04;
  m.scale.y = 0.04;
  m.scale.z = 0.04;
  m.color.a = 1.0;
  m.color.r = r;
  m.color.g = g;
  m.color.b = b;

  for (const auto & sample : cl.sample_set) {
    geometry_msgs::msg::Point p;
    p.x = sample.x;
    p.y = sample.y;
    p.z = 0.05;
    m.points.push_back(p);
  }

  pending_markers_.markers.push_back(m);
}

void VMarkerLocalizationNode::flush_markers()
{
  if (!pending_markers_.markers.empty()) {
    marker_pub_->publish(pending_markers_);
  }
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VMarkerLocalizationNode>("v_marker_localization_node");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
