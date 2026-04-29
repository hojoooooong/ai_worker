#include "ffw_safety_qp/shield_qp.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

namespace
{

ffw_safety_qp::ShieldQPParams defaultParams()
{
  ffw_safety_qp::ShieldQPParams p;
  p.dt = 0.02;
  p.alpha_obs = 5.0;
  p.alpha_jl = 10.0;
  p.damping = 0.01;
  p.slack_penalty = 1.0e4;
  p.safe_distance = 0.02;
  return p;
}

}  // namespace

TEST(ShieldQP, NoConstraintsTracksDesired)
{
  const int n = 7;
  ffw_safety_qp::ShieldQP qp(n, /*max_capsules=*/4);
  auto p = defaultParams();
  p.damping = 0.0;  // exact tracking
  qp.setParams(p);

  Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd q_lo = Eigen::VectorXd::Constant(n, -3.0);
  Eigen::VectorXd q_hi = Eigen::VectorXd::Constant(n, +3.0);
  Eigen::VectorXd v_max = Eigen::VectorXd::Constant(n, 5.0);
  Eigen::VectorXd qdot_des(n);
  qdot_des << 0.1, -0.2, 0.0, 0.3, -0.05, 0.0, 0.15;

  qp.setState(q, q_lo, q_hi, v_max);
  qp.setDesiredJointVel(qdot_des);
  qp.setObstacles({});

  Eigen::VectorXd qdot_safe;
  ASSERT_TRUE(qp.solve(qdot_safe));
  EXPECT_NEAR((qdot_safe - qdot_des).norm(), 0.0, 1e-3);
}

TEST(ShieldQP, DampingShrinksDesired)
{
  // With damping > 0, the cost ½‖qdot - qdot_des‖² + ½ damping ‖qdot‖² shrinks
  // the unconstrained solution to qdot_des / (1 + damping).
  const int n = 3;
  ffw_safety_qp::ShieldQP qp(n, /*max_capsules=*/0);
  auto p = defaultParams();
  p.damping = 0.1;
  qp.setParams(p);

  Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd q_lo = Eigen::VectorXd::Constant(n, -3.0);
  Eigen::VectorXd q_hi = Eigen::VectorXd::Constant(n, +3.0);
  Eigen::VectorXd v_max = Eigen::VectorXd::Constant(n, 5.0);
  Eigen::VectorXd qdot_des(n);
  qdot_des << 1.0, -0.5, 0.2;

  qp.setState(q, q_lo, q_hi, v_max);
  qp.setDesiredJointVel(qdot_des);
  qp.setObstacles({});

  Eigen::VectorXd qdot_safe;
  ASSERT_TRUE(qp.solve(qdot_safe));
  Eigen::VectorXd expected = qdot_des / (1.0 + p.damping);
  EXPECT_NEAR((qdot_safe - expected).norm(), 0.0, 1e-3);
}

TEST(ShieldQP, ObstacleAlongDesiredBlocksMotion)
{
  const int n = 3;
  ffw_safety_qp::ShieldQP qp(n, /*max_capsules=*/2);
  auto p = defaultParams();
  p.slack_penalty = 1.0e6;  // make slack expensive so the QP really tracks the constraint
  qp.setParams(p);

  Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd q_lo = Eigen::VectorXd::Constant(n, -3.0);
  Eigen::VectorXd q_hi = Eigen::VectorXd::Constant(n, +3.0);
  Eigen::VectorXd v_max = Eigen::VectorXd::Constant(n, 5.0);
  Eigen::VectorXd qdot_des(n);
  qdot_des << 1.0, 0.0, 0.0;  // want to move along axis 0

  // Distance is at the safety boundary; gradient says "axis 0 motion makes d smaller".
  ffw_safety_qp::ObstacleConstraint o;
  o.distance = p.safe_distance;
  o.grad_arm = Eigen::VectorXd(n);
  o.grad_arm << -1.0, 0.0, 0.0;  // ∂d/∂q0 = -1 → moving qdot0 positive decreases d

  qp.setState(q, q_lo, q_hi, v_max);
  qp.setDesiredJointVel(qdot_des);
  qp.setObstacles({o});

  Eigen::VectorXd qdot_safe;
  ASSERT_TRUE(qp.solve(qdot_safe));

  // Constraint at d == d_safe demands g·qdot ≥ 0 (modulo slack).
  // With g = (-1,0,0), this means -qdot0 ≥ 0 → qdot0 ≤ 0.
  // The desired was +1. So qdot_safe(0) should be at or below ~0 (with small slack-tolerated negative).
  EXPECT_LE(qdot_safe(0), 0.05);
}

TEST(ShieldQP, JointUpperLimitCBFBinds)
{
  const int n = 1;
  ffw_safety_qp::ShieldQP qp(n, /*max_capsules=*/0);
  auto p = defaultParams();
  p.slack_penalty = 1.0e6;
  qp.setParams(p);

  // Joint at upper limit; commanded velocity is positive → CBF must clamp to ≤ 0.
  Eigen::VectorXd q(1);            q << 1.0;
  Eigen::VectorXd q_lo(1);          q_lo << -1.0;
  Eigen::VectorXd q_hi(1);          q_hi <<  1.0;
  Eigen::VectorXd v_max(1);         v_max << 5.0;
  Eigen::VectorXd qdot_des(1);      qdot_des << 1.0;

  qp.setState(q, q_lo, q_hi, v_max);
  qp.setDesiredJointVel(qdot_des);
  qp.setObstacles({});

  Eigen::VectorXd qdot_safe;
  ASSERT_TRUE(qp.solve(qdot_safe));
  // qdot ≤ α_jl (q_max - q) = 10 * 0 = 0 (modulo slack)
  EXPECT_LE(qdot_safe(0), 0.05);
}
