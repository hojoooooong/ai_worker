#pragma once

#include <rclcpp/rclcpp.hpp>

#include <urdf/model.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/bool.hpp>

#include <fcl/fcl.h>

#include <unordered_map>
#include <vector>
#include <set>
#include <string>

namespace ffw_collision
{

class SelfCollisionNode : public rclcpp::Node
{
public:
  explicit SelfCollisionNode();

private:
  /* ========================== */
  /* ===== Internal Types ===== */
  /* ========================== */

  struct CollisionModel
  {
    std::shared_ptr<fcl::CollisionGeometryd> geometry;
    fcl::Transform3d offset;   // T_link_collision
  };

  /* ========================== */
  /* ===== Initialization ===== */
  /* ========================== */

  void load_urdf_collision_model();

  /* ========================== */
  /* ===== Main Loop ========== */
  /* ========================== */

  void update();
  bool update_link_transform(const std::string & link);

  /* ========================== */
  /* ===== Visualization ===== */
  /* ========================== */

  void publish_markers(const std::set<std::string> & colliding_links);
  void append_capsule_marker(
    const std::string & link,
    bool is_collision,
    int id,
    visualization_msgs::msg::MarkerArray & array);

  /* ========================== */
  /* ===== Parameters ========= */
  /* ========================== */

  std::string robot_description_;
  std::string world_frame_;
  bool enable_marker_;
  std::string left_arm_keyword_;
  std::string right_arm_keyword_;
  int tf_timeout_ms_;

  /* ========================== */
  /* ===== TF ================ */
  /* ========================== */

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  /* ========================== */
  /* ===== Collision Data ===== */
  /* ========================== */

  std::unordered_map<std::string, CollisionModel> collision_models_;
  std::unordered_map<
    std::string,
    std::shared_ptr<fcl::CollisionObjectd>> fcl_objects_;

  std::vector<std::string> left_arm_links_;
  std::vector<std::string> right_arm_links_;
  std::vector<std::string> body_links_;

  fcl::CollisionRequestd collision_request_;

  /* ========================== */
  /* ===== ROS Interfaces ==== */
  /* ========================== */

  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr collision_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace ffw_collision
