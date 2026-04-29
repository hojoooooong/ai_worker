#include "ffw_safety_qp/capsule_cloud_distance.hpp"

#include <algorithm>
#include <cmath>

namespace ffw_safety_qp
{

CapsuleQueryResult queryCapsule(
  const Eigen::Vector3d & p1_world,
  const Eigen::Vector3d & p2_world,
  double radius,
  const pcl::KdTreeFLANN<pcl::PointXYZ> & kdtree,
  const pcl::PointCloud<pcl::PointXYZ> & cloud,
  double buffer,
  double search_margin)
{
  CapsuleQueryResult res;
  if (cloud.empty()) {return res;}

  const Eigen::Vector3d ab = p2_world - p1_world;
  const double seg2 = ab.squaredNorm();
  const Eigen::Vector3d mid = 0.5 * (p1_world + p2_world);
  const double half_len = 0.5 * std::sqrt(seg2);
  const double R = half_len + radius + buffer + search_margin;

  pcl::PointXYZ query;
  query.x = static_cast<float>(mid.x());
  query.y = static_cast<float>(mid.y());
  query.z = static_cast<float>(mid.z());

  std::vector<int> idxs;
  std::vector<float> sq;
  // const_cast: PCL's radiusSearch is non-const but does not mutate the tree.
  auto & kdt_mut = const_cast<pcl::KdTreeFLANN<pcl::PointXYZ> &>(kdtree);
  const int found = kdt_mut.radiusSearch(query, R, idxs, sq);
  if (found <= 0) {return res;}

  double best_d = std::numeric_limits<double>::infinity();
  Eigen::Vector3d best_P = Eigen::Vector3d::Zero();
  Eigen::Vector3d best_Q = Eigen::Vector3d::Zero();
  double best_t = 0.0;

  for (int k : idxs) {
    const auto & pt = cloud.points[k];
    Eigen::Vector3d P(pt.x, pt.y, pt.z);
    double t = 0.0;
    if (seg2 > 1e-12) {
      t = (P - p1_world).dot(ab) / seg2;
      t = std::clamp(t, 0.0, 1.0);
    }
    Eigen::Vector3d Q = p1_world + t * ab;
    const double dist = (P - Q).norm() - radius;
    if (dist < best_d) {
      best_d = dist;
      best_P = P;
      best_Q = Q;
      best_t = t;
    }
  }

  res.distance = best_d;
  res.closest_cloud_point = best_P;
  res.closest_capsule_point = best_Q;
  res.t = best_t;
  res.active = (best_d <= buffer);

  // Normal: from P* toward Q* (i.e. the direction the capsule surface should move
  // to increase d). ∂d/∂q = nᵀ · J(Q*).
  const Eigen::Vector3d delta = best_Q - best_P;
  const double nlen = delta.norm();
  if (nlen > 1e-9) {
    res.normal = delta / nlen;
  } else {
    // Capsule axis exactly on the cloud point — pick a unit perpendicular to ab.
    if (seg2 > 1e-12) {
      Eigen::Vector3d any(1, 0, 0);
      if (std::abs(ab.normalized().dot(any)) > 0.9) {any = Eigen::Vector3d(0, 1, 0);}
      res.normal = ab.cross(any).normalized();
    } else {
      res.normal = Eigen::Vector3d::UnitZ();
    }
  }

  return res;
}

}  // namespace ffw_safety_qp
