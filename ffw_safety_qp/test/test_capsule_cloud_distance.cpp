#include "ffw_safety_qp/capsule_cloud_distance.hpp"

#include <gtest/gtest.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace
{

pcl::PointCloud<pcl::PointXYZ>::Ptr makeWallCloud(double z, int n_per_side = 21)
{
  auto pc = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  for (int i = 0; i < n_per_side; ++i) {
    for (int j = 0; j < n_per_side; ++j) {
      const float x = -0.5f + 1.0f * i / (n_per_side - 1);
      const float y = -0.5f + 1.0f * j / (n_per_side - 1);
      pc->push_back(pcl::PointXYZ(x, y, static_cast<float>(z)));
    }
  }
  return pc;
}

}  // namespace

TEST(CapsuleCloudDistance, EmptyCloudReturnsInactive)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::KdTreeFLANN<pcl::PointXYZ> kdt;
  // Empty cloud → kdtree is empty → query short-circuits.
  Eigen::Vector3d a(0, 0, 0.5), b(0, 0, 0.7);
  auto r = ffw_safety_qp::queryCapsule(a, b, 0.05, kdt, *pc, 0.10);
  EXPECT_FALSE(r.active);
  EXPECT_TRUE(std::isinf(r.distance));
}

TEST(CapsuleCloudDistance, SphereAboveWall)
{
  // Wall at z=0; sphere capsule (p1==p2) at z=0.10, radius 0.04 → expect d ≈ 0.06.
  auto pc = makeWallCloud(0.0);
  pcl::KdTreeFLANN<pcl::PointXYZ> kdt;
  kdt.setInputCloud(pc);

  Eigen::Vector3d c(0.0, 0.0, 0.10);
  auto r = ffw_safety_qp::queryCapsule(c, c, 0.04, kdt, *pc, 0.20);
  EXPECT_TRUE(r.active);
  EXPECT_NEAR(r.distance, 0.06, 1e-3);
  // Normal should point upward (+z) since the wall is below the sphere.
  EXPECT_NEAR(r.normal.z(), 1.0, 1e-3);
}

TEST(CapsuleCloudDistance, CylinderSegmentAboveWall)
{
  auto pc = makeWallCloud(0.0);
  pcl::KdTreeFLANN<pcl::PointXYZ> kdt;
  kdt.setInputCloud(pc);

  // Horizontal segment from (-0.1,0,0.1) to (0.1,0,0.1), radius 0.03 → d ≈ 0.07.
  Eigen::Vector3d a(-0.1, 0.0, 0.10), b(0.1, 0.0, 0.10);
  auto r = ffw_safety_qp::queryCapsule(a, b, 0.03, kdt, *pc, 0.20);
  EXPECT_TRUE(r.active);
  EXPECT_NEAR(r.distance, 0.07, 1e-3);
  // closest point on the capsule axis must lie between A and B (i.e. t in [0,1]).
  EXPECT_GE(r.t, 0.0);
  EXPECT_LE(r.t, 1.0);
}

TEST(CapsuleCloudDistance, PenetrationGivesNegativeDistance)
{
  auto pc = makeWallCloud(0.0);
  pcl::KdTreeFLANN<pcl::PointXYZ> kdt;
  kdt.setInputCloud(pc);

  // Sphere at z=0.02 with radius 0.05 → axis-to-wall distance is 0.02,
  // so signed surface distance = 0.02 - 0.05 = -0.03 (penetrating).
  Eigen::Vector3d c(0.0, 0.0, 0.02);
  auto r = ffw_safety_qp::queryCapsule(c, c, 0.05, kdt, *pc, 0.20);
  EXPECT_TRUE(r.active);
  EXPECT_NEAR(r.distance, -0.03, 1e-3);
}
