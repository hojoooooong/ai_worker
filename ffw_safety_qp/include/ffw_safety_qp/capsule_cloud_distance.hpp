#pragma once

#include "ffw_safety_qp/capsule_spec.hpp"

#include <Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <limits>
#include <memory>
#include <vector>

namespace ffw_safety_qp
{

// Result of querying one capsule against a kdtree of cloud points.
struct CapsuleQueryResult
{
  // Signed surface distance: ||P* - Q*|| - radius. Positive = clear, negative = penetrating.
  double distance{std::numeric_limits<double>::infinity()};

  // Whether distance ≤ buffer (the CBF should activate this slot).
  bool active{false};

  // Closest cloud point P* (world coordinates).
  Eigen::Vector3d closest_cloud_point{Eigen::Vector3d::Zero()};

  // Closest point on capsule axis Q* (world coordinates).
  Eigen::Vector3d closest_capsule_point{Eigen::Vector3d::Zero()};

  // Unit normal pointing from cloud point toward capsule axis (= (Q*-P*) / |Q*-P*|).
  // Used as the gradient direction: ∂d/∂q = nᵀ · J(Q*).
  Eigen::Vector3d normal{Eigen::Vector3d::Zero()};

  // Parameter t ∈ [0,1] of Q* along the capsule axis (0 = p1, 1 = p2).
  double t{0.0};
};

// Closest-point query of a single capsule (in world coordinates) against a kdtree.
// `buffer` controls the activation threshold; `search_margin` extends the radius
// search beyond (radius + buffer) to be safe.
CapsuleQueryResult queryCapsule(
  const Eigen::Vector3d & p1_world,
  const Eigen::Vector3d & p2_world,
  double radius,
  const pcl::KdTreeFLANN<pcl::PointXYZ> & kdtree,
  const pcl::PointCloud<pcl::PointXYZ> & cloud,
  double buffer,
  double search_margin = 0.05);

}  // namespace ffw_safety_qp
