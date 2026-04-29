#pragma once

#include <Eigen/Dense>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "ffw_safety_qp/capsule_spec.hpp"
#include "ffw_safety_qp/robot_kinematics.hpp"
#include "ffw_safety_qp/shield_qp.hpp"

namespace ffw_safety_qp
{

class LeaderShieldNode : public rclcpp::Node
{
public:
  enum class Side { Left, Right };

  explicit LeaderShieldNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions{});

private:
  // Per-side state.
  struct ArmContext
  {
    std::vector<std::string> arm_joint_names;     // 7 names in v-vector order
    std::vector<int> v_idx;                       // 7 indices into pinocchio v
    Eigen::VectorXd q_lower_arm;                  // 7
    Eigen::VectorXd q_upper_arm;                  // 7
    Eigen::VectorXd v_max_arm;                    // 7
    CapsuleList capsules;                         // for this side only
    std::unique_ptr<ShieldQP> qp;
    rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr leader_sub;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr shielded_pub;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr min_dist_pub;
  };

  // Lifecycle helpers.
  void declareParameters();
  void initFromRobotDescription(const std::string & urdf_xml);
  void buildArmContext(Side side, const std::vector<std::string> & arm_joint_names);

  // Subscriber callbacks.
  void onJointStates(const sensor_msgs::msg::JointState::SharedPtr msg);
  void onCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void onRobotDescription(const std_msgs::msg::String::SharedPtr msg);
  void onLeaderTraj(Side side, const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);

  // Periodic helpers.
  void publishMarkers();

  // Core control step (called per-side on each leader trajectory message).
  void controlStep(
    Side side,
    const trajectory_msgs::msg::JointTrajectory & leader_msg);

  // Helpers.
  ArmContext & ctx(Side side) {return side == Side::Left ? left_ : right_;}
  const std::string & sideName(Side side) const
  {
    return side == Side::Left ? left_name_ : right_name_;
  }

  // Lookup q-vector index given a joint name (Pinocchio model). Returns -1 if missing.
  int vIndex(const std::string & joint_name) const;

  // Pull current full-DOF q from cached joint_states.
  Eigen::VectorXd buildFullQ() const;

  // ---------------- members ----------------
  std::shared_ptr<RobotKinematics> kin_;       // built once /robot_description arrives
  std::atomic<bool> kinematics_ready_{false};

  ArmContext left_;
  ArmContext right_;
  const std::string left_name_{"left"};
  const std::string right_name_{"right"};

  // Cached joint state, full-DOF size.
  std::mutex js_mtx_;
  Eigen::VectorXd q_meas_;
  bool joint_state_ready_{false};
  std::unordered_map<std::string, int> js_name_to_v_idx_;  // /joint_states.name → v idx

  // Cloud + KdTree.
  std::mutex cloud_mtx_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
  std::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZ>> kdtree_;
  rclcpp::Time last_cloud_stamp_;
  bool cloud_ready_{false};

  // Subscriptions / publishers / timer.
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr js_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_desc_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::TimerBase::SharedPtr marker_timer_;

  // Parameters.
  std::string urdf_path_param_;          // optional fallback if /robot_description unavailable
  std::string base_frame_;
  double control_dt_;
  double buffer_;
  double search_margin_;
  ShieldQPParams qp_params_;
  int max_capsules_per_side_;
  std::vector<std::string> link_whitelist_left_;
  std::vector<std::string> link_whitelist_right_;
  std::vector<std::string> arm_joint_names_left_;
  std::vector<std::string> arm_joint_names_right_;
};

}  // namespace ffw_safety_qp
