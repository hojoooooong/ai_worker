#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace ffw_safety_qp
{

struct CapsuleSpec
{
  std::string frame;            // URDF link name (e.g. "arm_l_link5")
  Eigen::Vector3d p1_local;     // segment endpoint 1, in link-local frame
  Eigen::Vector3d p2_local;     // segment endpoint 2 (== p1_local for sphere)
  double radius;
};

// One side's capsule list (typically left or right arm).
using CapsuleList = std::vector<CapsuleSpec>;

}  // namespace ffw_safety_qp
