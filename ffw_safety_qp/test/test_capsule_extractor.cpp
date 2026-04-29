#include "ffw_safety_qp/capsule_extractor.hpp"

#include <gtest/gtest.h>

#include <string>

namespace
{

// A minimal URDF with two arm-style links: one with cyl+sphere,
// the other with sphere only. base_link is included but NOT in whitelist
// so it should not appear in the output.
const char * kURDFXml = R"(
<robot name="test_arm">
  <link name="base_link"/>
  <link name="arm_l_link3">
    <collision>
      <origin rpy="0 0 0" xyz="0.0 0.0 -0.05"/>
      <geometry><cylinder length="0.20" radius="0.04"/></geometry>
    </collision>
    <collision>
      <origin rpy="0 0 0" xyz="0.0 0.0 0.05"/>
      <geometry><sphere radius="0.04"/></geometry>
    </collision>
  </link>
  <link name="arm_l_link4">
    <collision>
      <origin rpy="0 0 0" xyz="0.10 0.0 0.0"/>
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <joint name="j3" type="revolute">
    <parent link="base_link"/>
    <child link="arm_l_link3"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="10" velocity="2"/>
  </joint>
  <joint name="j4" type="revolute">
    <parent link="arm_l_link3"/>
    <child link="arm_l_link4"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="10" velocity="2"/>
  </joint>
</robot>
)";

}  // namespace

TEST(CapsuleExtractor, ExtractsCylinderAndSphere)
{
  const auto caps = ffw_safety_qp::extractCapsulesFromURDFString(
    kURDFXml, {"arm_l_link3", "arm_l_link4"});

  ASSERT_EQ(caps.size(), 3u);

  // First: cylinder on link3.  Endpoints are origin + (0,0,±L/2) → (0,0,-0.15) and (0,0,0.05)
  EXPECT_EQ(caps[0].frame, "arm_l_link3");
  EXPECT_NEAR(caps[0].radius, 0.04, 1e-9);
  EXPECT_NEAR(caps[0].p1_local.z(), -0.15, 1e-9);
  EXPECT_NEAR(caps[0].p2_local.z(), 0.05, 1e-9);

  // Second: sphere on link3 at xyz=(0,0,0.05). Degenerate.
  EXPECT_EQ(caps[1].frame, "arm_l_link3");
  EXPECT_NEAR(caps[1].radius, 0.04, 1e-9);
  EXPECT_NEAR(caps[1].p1_local.z(), 0.05, 1e-9);
  EXPECT_NEAR((caps[1].p1_local - caps[1].p2_local).norm(), 0.0, 1e-9);

  // Third: sphere on link4 at (0.10, 0, 0).
  EXPECT_EQ(caps[2].frame, "arm_l_link4");
  EXPECT_NEAR(caps[2].radius, 0.05, 1e-9);
  EXPECT_NEAR(caps[2].p1_local.x(), 0.10, 1e-9);
}

TEST(CapsuleExtractor, IgnoresMissingLinks)
{
  const auto caps = ffw_safety_qp::extractCapsulesFromURDFString(
    kURDFXml, {"nonexistent_link"});
  EXPECT_EQ(caps.size(), 0u);
}

TEST(CapsuleExtractor, ThrowsOnBadURDF)
{
  EXPECT_THROW(
    ffw_safety_qp::extractCapsulesFromURDFString("not xml", {}),
    std::runtime_error);
}
