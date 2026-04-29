// Quick smoke test: load real SG2 follower URDF, extract capsules with the
// production whitelist, build the pinocchio model, and exercise the QP shield
// once with an empty cloud. Not part of the gtest run; built and invoked
// manually in CI/dev.

#include "ffw_safety_qp/capsule_extractor.hpp"
#include "ffw_safety_qp/robot_kinematics.hpp"
#include "ffw_safety_qp/shield_qp.hpp"

#include <Eigen/Dense>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv)
{
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <path_to_ffw_sg2_follower.urdf>\n", argv[0]);
    return 2;
  }
  const std::string urdf_path = argv[1];

  std::ifstream in(urdf_path);
  if (!in) {
    std::fprintf(stderr, "Cannot open %s\n", urdf_path.c_str());
    return 2;
  }
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string urdf_xml = ss.str();

  const std::vector<std::string> wl_left{
    "arm_l_link3", "arm_l_link4", "arm_l_link5",
    "arm_l_link6", "arm_l_link7", "end_effector_l_link"};
  const std::vector<std::string> wl_right{
    "arm_r_link3", "arm_r_link4", "arm_r_link5",
    "arm_r_link6", "arm_r_link7", "end_effector_r_link"};

  auto caps_l = ffw_safety_qp::extractCapsulesFromURDFString(urdf_xml, wl_left);
  auto caps_r = ffw_safety_qp::extractCapsulesFromURDFString(urdf_xml, wl_right);
  std::printf("[smoke] left capsules: %zu, right capsules: %zu\n",
    caps_l.size(), caps_r.size());

  auto kin = ffw_safety_qp::RobotKinematics::fromXML(urdf_xml);
  std::printf("[smoke] pinocchio: nq=%d nv=%d\n", kin->nq(), kin->nv());

  // Resolve the 7 left-arm joints in the v vector.
  const std::vector<std::string> arm_l{
    "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
    "arm_l_joint5", "arm_l_joint6", "arm_l_joint7"};
  std::vector<int> v_idx;
  for (const auto & jn : arm_l) {
    int v = kin->vIndexOfJoint(jn);
    if (v < 0) {
      std::fprintf(stderr, "[smoke] missing joint: %s\n", jn.c_str());
      return 1;
    }
    v_idx.push_back(v);
  }

  // FK at q=0 to confirm capsule frames resolve.
  Eigen::VectorXd q = Eigen::VectorXd::Zero(kin->nq());
  kin->updateFK(q);
  for (const auto & cap : caps_l) {
    auto T = kin->framePose(cap.frame);
    std::printf("[smoke] %s @ q=0 -> world (%.3f, %.3f, %.3f)\n",
      cap.frame.c_str(), T.translation().x(), T.translation().y(),
      T.translation().z());
  }

  // QP solve with empty cloud / no obstacles: should track desired exactly (damping=0).
  ffw_safety_qp::ShieldQP qp(7, 24);
  ffw_safety_qp::ShieldQPParams qparams;
  qparams.damping = 0.0;
  qparams.alpha_jl = 10.0;
  qparams.slack_penalty = 1.0e4;
  qp.setParams(qparams);

  Eigen::VectorXd q_arm(7);
  Eigen::VectorXd q_lo(7), q_hi(7), v_max(7);
  for (int i = 0; i < 7; ++i) {
    q_arm(i) = q(v_idx[i]);
    q_lo(i) = kin->qLower()(v_idx[i]);
    q_hi(i) = kin->qUpper()(v_idx[i]);
    v_max(i) = std::max(0.5, kin->vMax()(v_idx[i]));
  }

  Eigen::VectorXd qdot_des(7);
  qdot_des << 0.1, -0.1, 0.05, 0.0, -0.05, 0.1, 0.0;

  qp.setState(q_arm, q_lo, q_hi, v_max);
  qp.setDesiredJointVel(qdot_des);
  qp.setObstacles({});

  Eigen::VectorXd qdot_safe;
  if (!qp.solve(qdot_safe)) {
    std::fprintf(stderr, "[smoke] QP solve failed\n");
    return 1;
  }
  const double err = (qdot_safe - qdot_des).norm();
  std::printf("[smoke] QP empty-cloud tracking error: %.6f (expect ~0)\n", err);
  if (err > 1e-3) {
    std::fprintf(stderr, "[smoke] tracking error too large\n");
    return 1;
  }

  std::printf("[smoke] OK\n");
  return 0;
}
