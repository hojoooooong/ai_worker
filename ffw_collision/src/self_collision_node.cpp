// Copyright 2024 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Author: Sungho Woo

#include "ffw_collision/self_collision_node.hpp"

#include <tf2_eigen/tf2_eigen.hpp>

namespace ffw_collision
{

SelfCollisionNode::SelfCollisionNode()
: Node("self_collision_node"),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_)
{
  declare_parameter("robot_description", "");
  declare_parameter("world_frame", "base_link");
  declare_parameter("enable_marker", true);
  declare_parameter("left_arm_keyword", "arm_l_");
  declare_parameter("right_arm_keyword", "arm_r_");
  declare_parameter("tf_timeout_ms", 50);

  get_parameter("robot_description", robot_description_);
  get_parameter("world_frame", world_frame_);
  get_parameter("enable_marker", enable_marker_);
  get_parameter("left_arm_keyword", left_arm_keyword_);
  get_parameter("right_arm_keyword", right_arm_keyword_);
  get_parameter("tf_timeout_ms", tf_timeout_ms_);

  if (robot_description_.empty()) {
    RCLCPP_FATAL(get_logger(), "robot_description is empty");
    rclcpp::shutdown();
    return;
  }

  collision_pub_ =
    create_publisher<std_msgs::msg::Bool>("/collision/is_colliding", 10);

  if (enable_marker_) {
    marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>(
        "/collision/markers", 10);
  }

  load_urdf_collision_model();

  timer_ = create_wall_timer(
    std::chrono::milliseconds(20),
    std::bind(&SelfCollisionNode::update, this));

  RCLCPP_INFO(
    get_logger(),
    "FFW Self Collision Node started (left ↔ right arms, body)");
}

/* ========================== */
/* ===== URDF LOADING ======= */
/* ========================== */

void SelfCollisionNode::load_urdf_collision_model()
{
  urdf::Model model;
  if (!model.initString(robot_description_)) {
    RCLCPP_FATAL(get_logger(), "Failed to parse URDF");
    rclcpp::shutdown();
    return;
  }

  RCLCPP_DEBUG(get_logger(), "Scanning URDF links for collision models...");

  int total_links_with_collision = 0;
  int relevant_links_found = 0;

  for (const auto & kv : model.links_) {
    const std::string & link_name = kv.first;
    const auto & link = kv.second;

    if (!link->collision || !link->collision->geometry) {
      continue;
    }

    total_links_with_collision++;

    // Classify links: left arm, right arm, or body
    bool is_body = (link_name.find("arm_base_link") != std::string::npos);
    bool is_left_arm = (link_name.find(left_arm_keyword_) != std::string::npos) ||
                        (link_name.find("gripper_l_") != std::string::npos) ||
                        (link_name.find("camera_l_") != std::string::npos);
    bool is_right_arm = (link_name.find(right_arm_keyword_) != std::string::npos) ||
                        (link_name.find("gripper_r_") != std::string::npos) ||
                        (link_name.find("camera_r_") != std::string::npos);

    if (!is_left_arm && !is_right_arm && !is_body) {
      continue;
    }

    relevant_links_found++;

    // Get geometry type string for logging
    std::string geom_type_str = "unknown";
    if (link->collision->geometry->type == urdf::Geometry::CYLINDER) {
      geom_type_str = "CYLINDER";
    } else if (link->collision->geometry->type == urdf::Geometry::BOX) {
      geom_type_str = "BOX";
    } else if (link->collision->geometry->type == urdf::Geometry::SPHERE) {
      geom_type_str = "SPHERE";
    } else if (link->collision->geometry->type == urdf::Geometry::MESH) {
      geom_type_str = "MESH";
    }

    RCLCPP_INFO(
      get_logger(),
      "Found link: %s (type: %s, left=%d, right=%d, body=%d)",
      link_name.c_str(), geom_type_str.c_str(), is_left_arm, is_right_arm, is_body);

    // Convert geometry to capsule (supports CYLINDER and BOX)
    std::shared_ptr<fcl::Capsuled> capsule;

    if (link->collision->geometry->type == urdf::Geometry::CYLINDER) {
      auto cyl = std::dynamic_pointer_cast<urdf::Cylinder>(link->collision->geometry);
      capsule = std::make_shared<fcl::Capsuled>(cyl->radius, cyl->length);
    } else if (link->collision->geometry->type == urdf::Geometry::BOX) {
      // Approximate BOX as capsule: longest axis as length, average of other two as radius
      auto box = std::dynamic_pointer_cast<urdf::Box>(link->collision->geometry);
      const double x = box->dim.x;
      const double y = box->dim.y;
      const double z = box->dim.z;

      const double length = std::max({x, y, z});
      double radius = 0.0;
      if (length == x) {
        radius = (y + z) / 4.0;  // average/2 = radius
      } else if (length == y) {
        radius = (x + z) / 4.0;
      } else {
        radius = (x + y) / 4.0;
      }

      capsule = std::make_shared<fcl::Capsuled>(radius, length);
      RCLCPP_DEBUG(
        get_logger(),
        "Converting BOX to capsule for %s: box=(%f,%f,%f) -> capsule(r=%f, l=%f)",
        link_name.c_str(), x, y, z, radius, length);
    } else {
      RCLCPP_WARN(
        get_logger(),
        "Link %s has %s collision geometry, skipping (only CYLINDER and BOX supported)",
        link_name.c_str(), geom_type_str.c_str());
      continue;
    }

    CollisionModel cm;
    cm.geometry = capsule;

    // Compute collision model offset transform
    fcl::Transform3d T = fcl::Transform3d::Identity();
    const auto & origin = link->collision->origin;

    T.translation() = fcl::Vector3d(origin.position.x, origin.position.y, origin.position.z);

    double roll, pitch, yaw;
    origin.rotation.getRPY(roll, pitch, yaw);
    T.linear() = (Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
                  Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                  Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()))
                     .toRotationMatrix();

    cm.offset = T;
    collision_models_[link_name] = cm;

    // Create FCL collision object
    auto obj = std::make_shared<fcl::CollisionObjectd>(cm.geometry);
    obj->setTransform(fcl::Transform3d::Identity());
    obj->computeAABB();
    fcl_objects_[link_name] = obj;

    // Categorize links into groups
    if (is_left_arm) {
      left_arm_links_.push_back(link_name);
    }
    if (is_right_arm) {
      right_arm_links_.push_back(link_name);
    }
    if (is_body) {
      body_links_.push_back(link_name);
    }
  }

  RCLCPP_INFO(
    get_logger(),
    "URDF scan complete: %d total links with collision, %d relevant links found",
    total_links_with_collision, relevant_links_found);
  RCLCPP_INFO(
    get_logger(),
    "Collision capsules loaded: left=%zu right=%zu body=%zu",
    left_arm_links_.size(), right_arm_links_.size(), body_links_.size());

  // Log loaded links for debugging
  if (left_arm_links_.empty() && right_arm_links_.empty() && body_links_.empty()) {
    RCLCPP_ERROR(
      get_logger(),
      "No links loaded! Check URDF and keyword matching.");
    RCLCPP_INFO(get_logger(), "Left arm keyword: '%s'", left_arm_keyword_.c_str());
    RCLCPP_INFO(get_logger(), "Right arm keyword: '%s'", right_arm_keyword_.c_str());
  } else {
    RCLCPP_INFO(get_logger(), "Left arm/gripper/camera links (%zu):", left_arm_links_.size());
    for (const auto & link : left_arm_links_) {
      RCLCPP_INFO(get_logger(), "  - %s", link.c_str());
    }
    RCLCPP_INFO(get_logger(), "Right arm/gripper/camera links (%zu):", right_arm_links_.size());
    for (const auto & link : right_arm_links_) {
      RCLCPP_INFO(get_logger(), "  - %s", link.c_str());
    }
    RCLCPP_INFO(get_logger(), "Body links (%zu):", body_links_.size());
    for (const auto & link : body_links_) {
      RCLCPP_INFO(get_logger(), "  - %s", link.c_str());
    }
  }
}

/* ========================== */
/* ===== MAIN UPDATE ======== */
/* ========================== */

void SelfCollisionNode::update()
{
  std::set<std::string> colliding_links;
  bool collision = false;

  // Update transforms for all links
  bool all_tf_ok = true;
  auto update_group = [&](const std::vector<std::string> & links) {
    for (const auto & link : links) {
      if (!update_link_transform(link)) {
        all_tf_ok = false;
      }
    }
  };

  update_group(left_arm_links_);
  update_group(right_arm_links_);
  update_group(body_links_);

  if (!all_tf_ok) {
    return;
  }

  // Helper function to check collision between two links
  auto check_collision = [&](const std::string & link1, const std::string & link2) {
    fcl::CollisionResultd res;
    fcl::collide(
      fcl_objects_[link1].get(),
      fcl_objects_[link2].get(),
      collision_request_,
      res);

    if (res.isCollision()) {
      collision = true;
      colliding_links.insert(link1);
      colliding_links.insert(link2);
    }
  };

  // 1. Check collisions between left and right arms
  for (const auto & l : left_arm_links_) {
    for (const auto & r : right_arm_links_) {
      check_collision(l, r);
    }
  }

  // 2. Check collisions between left arm and body
  for (const auto & l : left_arm_links_) {
    for (const auto & b : body_links_) {
      check_collision(l, b);
    }
  }

  // 3. Check collisions between right arm and body
  for (const auto & r : right_arm_links_) {
    for (const auto & b : body_links_) {
      check_collision(r, b);
    }
  }

  // Publish collision status
  std_msgs::msg::Bool msg;
  msg.data = collision;
  collision_pub_->publish(msg);

  if (enable_marker_) {
    publish_markers(colliding_links);
  }
}

bool SelfCollisionNode::update_link_transform(const std::string & link)
{
  geometry_msgs::msg::TransformStamped tf;
  try {
    tf = tf_buffer_.lookupTransform(
      world_frame_, link,
      tf2::TimePointZero,
      tf2::durationFromSec(tf_timeout_ms_ / 1000.0));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "TF lookup failed for %s: %s", link.c_str(), ex.what());
    return false;
  }

  Eigen::Isometry3d T_world_link =
    tf2::transformToEigen(tf.transform);

  Eigen::Isometry3d T_world_collision =
    T_world_link * collision_models_[link].offset;

  fcl_objects_[link]->setTransform(T_world_collision);
  fcl_objects_[link]->computeAABB();
  return true;
}

/* ========================== */
/* ===== MARKERS ============ */
/* ========================== */

void SelfCollisionNode::publish_markers(const std::set<std::string> & colliding)
{
  visualization_msgs::msg::MarkerArray array;
  int id = 0;

  // Helper lambda to append markers for a group of links
  auto append_group = [&](const std::vector<std::string> & links) {
    for (const auto & link : links) {
      append_capsule_marker(link, colliding.count(link) > 0, id, array);
      id += 3;
    }
  };

  append_group(left_arm_links_);
  append_group(right_arm_links_);
  append_group(body_links_);

  marker_pub_->publish(array);
}

void SelfCollisionNode::append_capsule_marker(
  const std::string & link,
  bool is_collision,
  int id,
  visualization_msgs::msg::MarkerArray & array)
{
  auto capsule = std::dynamic_pointer_cast<fcl::Capsuled>(collision_models_[link].geometry);
  if (!capsule) {
    return;
  }

  const auto & tf = fcl_objects_[link]->getTransform();
  const Eigen::Vector3d pos = tf.translation();
  const Eigen::Quaterniond q(tf.linear());
  const double radius = capsule->radius;
  const double half_length = capsule->lz / 2.0;

  // Set color: red for collision, green for safe
  std_msgs::msg::ColorRGBA color;
  color.r = is_collision ? 1.0 : 0.0;
  color.g = is_collision ? 0.0 : 1.0;
  color.b = 0.0;
  color.a = 0.3;

  // Create base marker template
  visualization_msgs::msg::Marker base_marker;
  base_marker.header.frame_id = world_frame_;
  base_marker.header.stamp = now();
  base_marker.ns = link;
  base_marker.action = visualization_msgs::msg::Marker::ADD;
  base_marker.pose.position.x = pos.x();
  base_marker.pose.position.y = pos.y();
  base_marker.pose.position.z = pos.z();
  base_marker.pose.orientation.x = q.x();
  base_marker.pose.orientation.y = q.y();
  base_marker.pose.orientation.z = q.z();
  base_marker.pose.orientation.w = q.w();
  base_marker.color = color;
  base_marker.lifetime = rclcpp::Duration::from_seconds(0.1);

  // Cylinder marker
  visualization_msgs::msg::Marker cyl = base_marker;
  cyl.id = id;
  cyl.type = visualization_msgs::msg::Marker::CYLINDER;
  cyl.scale.x = cyl.scale.y = 2.0 * radius;
  cyl.scale.z = capsule->lz;
  array.markers.push_back(cyl);

  // Top sphere marker
  Eigen::Vector3d top_offset = tf.linear() * Eigen::Vector3d(0, 0, half_length);
  visualization_msgs::msg::Marker top = base_marker;
  top.id = id + 1;
  top.type = visualization_msgs::msg::Marker::SPHERE;
  top.scale.x = top.scale.y = top.scale.z = 2.0 * radius;
  top.pose.position.x += top_offset.x();
  top.pose.position.y += top_offset.y();
  top.pose.position.z += top_offset.z();
  array.markers.push_back(top);

  // Bottom sphere marker
  Eigen::Vector3d bottom_offset = tf.linear() * Eigen::Vector3d(0, 0, -half_length);
  visualization_msgs::msg::Marker bottom = base_marker;
  bottom.id = id + 2;
  bottom.type = visualization_msgs::msg::Marker::SPHERE;
  bottom.scale.x = bottom.scale.y = bottom.scale.z = 2.0 * radius;
  bottom.pose.position.x += bottom_offset.x();
  bottom.pose.position.y += bottom_offset.y();
  bottom.pose.position.z += bottom_offset.z();
  array.markers.push_back(bottom);
}


}  // namespace ffw_collision

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ffw_collision::SelfCollisionNode>());
  rclcpp::shutdown();
  return 0;
}
