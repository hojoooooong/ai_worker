#include <chrono>
#include <cmath>
#include <memory>
#include <vector>
#include <iostream>

#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include "std_msgs/msg/header.hpp"
#include "geometry_msgs/msg/point.hpp"
#include <v_marker_estimation/laser_processor.hpp>
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "nav_msgs/msg/odometry.hpp"

#include "v_marker_estimation/msg/laser_sample_set_list.hpp"
#include "v_marker_estimation/msg/laser_sample_set.hpp"
#include "v_marker_estimation/msg/laser_sample.hpp"
#include <rclcpp/rclcpp.hpp>
#include "std_srvs/srv/set_bool.hpp"


namespace laser_processor {

class LaserClusters : public rclcpp::Node {
public:
  LaserClusters(const std::string node_name): Node(node_name),
  num_prev_markers_published_(0),
  activation_flag_(false)
  {
    // Declare and get ROS parameters
    this->declare_parameter("fixed_frame", "base_link");
    this->declare_parameter("cluster_dist_euclid", 0.13);
    this->declare_parameter("min_points_per_cluster", 3);
    this->declare_parameter("max_detect_distance", 10.0);
    this->declare_parameter("qos_depth", 1);
    fixed_frame_ = this->get_parameter("fixed_frame").as_string();
    cluster_dist_euclid_ = this->get_parameter("cluster_dist_euclid").as_double();
    min_points_per_cluster_ = this->get_parameter("min_points_per_cluster").as_int();
    max_detect_distance_ = this->get_parameter("max_detect_distance").as_double();
    int8_t qos_depth = this->get_parameter("qos_depth").as_int();

    RCLCPP_INFO(this->get_logger(), "fixed_frame: %s", fixed_frame_.c_str());
    RCLCPP_INFO(this->get_logger(), "cluster_dist_euclid: %.2f", cluster_dist_euclid_);
    RCLCPP_INFO(this->get_logger(), "min_points_per_cluster: %d", min_points_per_cluster_);
    RCLCPP_INFO(this->get_logger(), "max_detect_distance: %.2f", max_detect_distance_);
    RCLCPP_INFO(this->get_logger(), "qos_depth: %d", qos_depth);

    const auto QOS_RKL10V =
      rclcpp::QoS(rclcpp::KeepLast(qos_depth)).reliable().durability_volatile();

    // Sensor QoS: best_effort for lidar drivers that publish with best_effort
    const auto QOS_SENSOR =
      rclcpp::QoS(rclcpp::KeepLast(qos_depth)).best_effort().durability_volatile();

    timer_ = this->create_wall_timer(std::chrono::milliseconds(30), std::bind(&LaserClusters::publish_clusters, this));

    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", QOS_SENSOR, std::bind(&LaserClusters::laserCallback, this, std::placeholders::_1));
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", QOS_SENSOR, std::bind(&LaserClusters::subscribe_odom, this, std::placeholders::_1));
    markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("clustered_viz", QOS_RKL10V);
    clusters_pub_ = this->create_publisher<v_marker_estimation::msg::LaserSampleSetList>("clusters", QOS_RKL10V);
    
    // Create service for docking perception activation
    docking_server_ = this->create_service<std_srvs::srv::SetBool>(
      "docking_perception_activate",
      std::bind(&LaserClusters::docking_activation_callback, this,
        std::placeholders::_1, std::placeholders::_2));
    
    // Create client to activate v_marker_localization_node
    marker_activation_client_ = this->create_client<std_srvs::srv::SetBool>(
      "v_marker_localization_activate");
    
    // Initial state: deactivated (waiting for docking_perception_activate service call)
    activation_flag_ = false;
    desired_marker_activation_ = false;
    marker_activation_synced_ = false;
    marker_activation_request_in_flight_ = false;
    marker_activation_retry_sec_ = 0.5;
    last_marker_activation_request_t_ = this->now();
    RCLCPP_INFO(this->get_logger(), "Cluster node initialized (waiting for docking_perception_activate trigger)");

    last_update_t_ = this->now();

    robot_x_ =0.0;
    robot_y_ =0.0;
    robot_theta_ =0.0;
  }

  ~LaserClusters()
  {
  }

private:
  int scan_num_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
  rclcpp::Publisher<v_marker_estimation::msg::LaserSampleSetList>::SharedPtr clusters_pub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr docking_server_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr marker_activation_client_;
  bool activation_flag_;
  bool marker_activation_synced_;
  bool marker_activation_request_in_flight_;
  bool desired_marker_activation_;
  rclcpp::Time last_marker_activation_request_t_;
  double marker_activation_retry_sec_;
  std::string fixed_frame_;
  double cluster_dist_euclid_;
  int min_points_per_cluster_;
  double max_detect_distance_;
  int num_prev_markers_published_;
  double robot_x_ ;
  double robot_y_ ;
  double robot_theta_ ;
  rclcpp::Time last_update_t_;
  visualization_msgs::msg::MarkerArray markers;
  sensor_msgs::msg::LaserScan scan_;
  std::mutex scan_mutex_;
  double robot_x_scan_;
  double robot_y_scan_;
  double robot_theta_scan_;

  void docking_activation_callback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response)
  {
    activation_flag_ = request->data;
    desired_marker_activation_ = activation_flag_;
    marker_activation_synced_ = false;
    
    // Forward desired state to v_marker_localization_node.
    // If service is not ready yet, periodic retry is handled in publish_clusters().
    activate_marker_localization();
    
    RCLCPP_INFO(this->get_logger(), "Docking perception activation: %s",
      activation_flag_ ? "activated" : "deactivated");
    
    response->success = true;
    response->message = "Docking perception activation set to " +
      std::string(activation_flag_ ? "true" : "false");
  }

  void activate_marker_localization()
  {
    if (marker_activation_synced_ || marker_activation_request_in_flight_) {
      return;
    }

    auto now = this->now();
    if ((now - last_marker_activation_request_t_).seconds() < marker_activation_retry_sec_) {
      return;
    }

    if (!marker_activation_client_->service_is_ready()) {
      RCLCPP_DEBUG(this->get_logger(),
        "Marker localization activation service not available yet, retrying");
      return;
    }

    auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
    request->data = desired_marker_activation_;
    const bool requested_state = desired_marker_activation_;
    marker_activation_request_in_flight_ = true;
    last_marker_activation_request_t_ = now;

    // Async request + retry state machine (no blocking wait)
    marker_activation_client_->async_send_request(request,
      [this, requested_state](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture future) {
        marker_activation_request_in_flight_ = false;
        auto response = future.get();
        if (response->success && requested_state == desired_marker_activation_) {
          marker_activation_synced_ = true;
          RCLCPP_INFO(this->get_logger(),
            "Marker localization %s", requested_state ? "activated" : "deactivated");
        } else {
          marker_activation_synced_ = false;
          RCLCPP_WARN(this->get_logger(),
            "Marker localization activation sync failed, retrying");
        }
      });
  }

  void laserCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan)
  {
    {
      std::lock_guard<std::mutex> lock(scan_mutex_);
      robot_x_scan_ = robot_x_;
      robot_y_scan_ = robot_y_;
      robot_theta_scan_ = robot_theta_;
      scan_ = *scan;
    }
  }

  void subscribe_odom(const nav_msgs::msg::Odometry::SharedPtr odom_data)
  {
    geometry_msgs::msg::Quaternion orientation = odom_data->pose.pose.orientation;
    geometry_msgs::msg::Point position = odom_data->pose.pose.position;

    tf2::Quaternion tf_orientation(orientation.x, orientation.y, orientation.z, orientation.w);
    tf2::Matrix3x3 tf_matrix(tf_orientation);
    double roll, pitch, yaw;
    tf_matrix.getRPY(roll, pitch, yaw);

    robot_x_ = position.x;
    robot_y_ = position.y;
    robot_theta_ = yaw;
  }

  void publish_clusters(){
    // Keep v_marker_localization activation synchronized with requested docking state.
    activate_marker_localization();

    // Only publish clusters when activated
    if (!activation_flag_) {
      // Publish empty markers to clear visualization when deactivated
      markers.markers.clear();
      markers_pub_->publish(markers);
      return;
    }

    sensor_msgs::msg::LaserScan scan;
    geometry_msgs::msg::Point robot_pose;

    {
      std::lock_guard<std::mutex> lock(scan_mutex_);
      scan = scan_;
      robot_pose.x = robot_x_scan_;
      robot_pose.y = robot_y_scan_;
      robot_pose.z = robot_theta_scan_;
    }

    laser_processor::ScanProcessor processor(scan);
    processor.splitConnected(cluster_dist_euclid_, 0.0);
    processor.removeLessThan(min_points_per_cluster_);
    processor.mergeClusters(cluster_dist_euclid_);

    markers.markers.clear();
    uint32_t id_num = 1;

    for (auto& cluster : processor.getClusters()) {
        geometry_msgs::msg::Point point = cluster->getPosition();
        tf2::Vector3 position(point.x, point.y, point.z);

        float rel_dist = std::sqrt(position.x() * position.x() + position.y() * position.y());

        if (rel_dist < max_detect_distance_) {
            auto points = cluster->getPoints();

            visualization_msgs::msg::Marker m;
            m.header.stamp = scan.header.stamp;
            m.header.frame_id = fixed_frame_;
            m.ns = "clusters";
            m.id = id_num;
            m.type = visualization_msgs::msg::Marker::POINTS;
            m.scale.x = 0.08;
            m.scale.y = 0.08;
            m.scale.z = 0.08;
            m.color.a = 0.5;
            m.color.r = (id_num % 3) == 0 ? 1 : 0;
            m.color.g = (id_num % 3) == 1 ? 1 : 0;
            m.color.b = (id_num % 3) == 2 ? 1 : 0;
            m.points.assign(points.begin(), points.end());


            visualization_msgs::msg::Marker m2;
            m2.header.stamp = scan.header.stamp;
            m2.header.frame_id = fixed_frame_;
            m2.ns = "mean_points";
            m2.id = id_num++;
            m2.type = visualization_msgs::msg::Marker::SPHERE;
            m2.pose.position.x = position.x();
            m2.pose.position.y = position.y();
            m2.pose.position.z = 0.2;
            m2.scale.x = 0.13;
            m2.scale.y = 0.13;
            m2.scale.z = 0.13;
            m2.color.a = 1;
            m2.color.r = 0;
            m2.color.g = 0;
            m2.color.b = 1;

            markers.markers.push_back(m);
            markers.markers.push_back(m2);
        }
    }

    for (int id_num_diff = num_prev_markers_published_ - id_num; id_num_diff > 0; id_num_diff--) {

      visualization_msgs::msg::Marker m;
      m.header.stamp = scan.header.stamp;
      m.header.frame_id = fixed_frame_;
      m.ns = "clusters";
      m.id = id_num_diff + id_num-1;
      m.action = visualization_msgs::msg::Marker::DELETE;
      markers.markers.push_back(m);

      m.header.stamp = scan.header.stamp;
      m.header.frame_id = fixed_frame_;
      m.ns = "mean_points";
      m.id = id_num_diff + id_num-1;
      m.action = visualization_msgs::msg::Marker::DELETE;
      markers.markers.push_back(m);
    }

    auto samplesets = std::make_unique<v_marker_estimation::msg::LaserSampleSetList>();
    v_marker_estimation::msg::LaserSampleSetList sampleset_;
    std::vector<v_marker_estimation::msg::LaserSampleSet> clusters_;
    sampleset_.sampleset_list = processor.getSampleSets();

    for (size_t i = 0; i < sampleset_.sampleset_list.size(); i++) {
      double rel_dist =
          std::pow(std::pow(sampleset_.sampleset_list[i].mean_p.x, 2.) + std::pow(sampleset_.sampleset_list[i].mean_p.y, 2.), 1. / 2.);
      if (rel_dist < max_detect_distance_) {
        sampleset_.sampleset_list[i].cluster_id = clusters_.size() + 1;
        clusters_.push_back(sampleset_.sampleset_list[i]);
      }
    }

    samplesets->header.stamp = scan.header.stamp;
    samplesets->header.frame_id = fixed_frame_;
    samplesets->sampleset_list = clusters_;
    samplesets->robot_pose = robot_pose;
    clusters_pub_->publish(std::move(samplesets));
    markers_pub_->publish(markers);

    num_prev_markers_published_ = id_num;

  }

};
}

  // namespace laser_processor

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<laser_processor::LaserClusters>("cluster_node");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
