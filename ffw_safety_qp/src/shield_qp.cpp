#include "ffw_safety_qp/shield_qp.hpp"

#include <algorithm>
#include <stdexcept>

namespace ffw_safety_qp
{

namespace
{

// QP layout:
//   x = [qdot (n), s_qmin (n), s_qmax (n), s_col (M)]   total nx = 3n + M
//
// Bound constraint count = nx (one box per variable).
// Inequality constraint count = 2n (joint-pos CBF) + M (obstacle CBF).

constexpr int kSlackPosCount = 2;  // s_qmin and s_qmax (each of size n)

}  // namespace

ShieldQP::ShieldQP(int n_arm, int max_capsules)
: n_arm_(n_arm), max_capsules_(max_capsules)
{
  if (n_arm_ <= 0) {
    throw std::invalid_argument("ShieldQP: n_arm must be positive");
  }
  if (max_capsules_ < 0) {
    throw std::invalid_argument("ShieldQP: max_capsules must be non-negative");
  }

  const int nx = (1 + kSlackPosCount) * n_arm_ + max_capsules_;
  const int nbc = nx;
  const int nineqc = 2 * n_arm_ + max_capsules_;
  const int neqc = 0;

  setQPsize(nx, nbc, nineqc, neqc);

  // Defaults so that the QP can be solved even before the user calls setState/etc.
  q_arm_ = Eigen::VectorXd::Zero(n_arm_);
  q_lower_arm_ = Eigen::VectorXd::Constant(n_arm_, -1.0e6);
  q_upper_arm_ = Eigen::VectorXd::Constant(n_arm_, +1.0e6);
  v_max_arm_ = Eigen::VectorXd::Constant(n_arm_, 1.0e6);
  qdot_des_arm_ = Eigen::VectorXd::Zero(n_arm_);
}

void ShieldQP::setState(
  const Eigen::VectorXd & q_arm,
  const Eigen::VectorXd & q_lower_arm,
  const Eigen::VectorXd & q_upper_arm,
  const Eigen::VectorXd & v_max_arm)
{
  if (q_arm.size() != n_arm_ || q_lower_arm.size() != n_arm_ ||
    q_upper_arm.size() != n_arm_ || v_max_arm.size() != n_arm_)
  {
    throw std::invalid_argument("ShieldQP::setState: vector size mismatch");
  }
  q_arm_ = q_arm;
  q_lower_arm_ = q_lower_arm;
  q_upper_arm_ = q_upper_arm;
  v_max_arm_ = v_max_arm.cwiseAbs();
}

void ShieldQP::setDesiredJointVel(const Eigen::VectorXd & qdot_des_arm)
{
  if (qdot_des_arm.size() != n_arm_) {
    throw std::invalid_argument("ShieldQP::setDesiredJointVel: size mismatch");
  }
  qdot_des_arm_ = qdot_des_arm;
}

void ShieldQP::setObstacles(const std::vector<ObstacleConstraint> & obstacles)
{
  active_obstacles_.clear();
  active_obstacles_.reserve(std::min<size_t>(obstacles.size(), max_capsules_));
  for (const auto & o : obstacles) {
    if (active_obstacles_.size() >= static_cast<size_t>(max_capsules_)) {break;}
    if (o.grad_arm.size() != n_arm_) {
      throw std::invalid_argument("ShieldQP::setObstacles: grad_arm size mismatch");
    }
    active_obstacles_.push_back(o);
  }
}

bool ShieldQP::solve(Eigen::VectorXd & qdot_safe)
{
  Eigen::MatrixXd sol;
  if (!solveQP(sol)) {return false;}
  // sol is nx x 1. Take the first n_arm components.
  qdot_safe = sol.topRows(n_arm_);
  return true;
}

void ShieldQP::setCost()
{
  P_ds_.setZero();
  q_ds_.setZero();

  // ½ ‖qdot - qdot_des‖² + ½ damping ‖qdot‖²
  //   P[qdot, qdot] = (1 + damping) I_n
  //   q[qdot]       = -qdot_des
  const double diag_q = 1.0 + params_.damping;
  for (int i = 0; i < n_arm_; ++i) {
    P_ds_(i, i) = diag_q;
    q_ds_(i) = -qdot_des_arm_(i);
  }

  // Slack penalty: quadratic ½ w_s ‖s‖² (L2-style soft constraint).
  // Quadratic-only keeps the full P PD for OSQP and pushes slacks toward 0 when
  // the constraint is feasible at slack=0.
  const int slack_start = n_arm_;
  const int slack_count = 2 * n_arm_ + max_capsules_;
  for (int i = 0; i < slack_count; ++i) {
    P_ds_(slack_start + i, slack_start + i) = params_.slack_penalty;
  }
}

void ShieldQP::setBoundConstraint()
{
  // Variable order: qdot (n), s_qmin (n), s_qmax (n), s_col (M)
  // Bounds:
  //   -v_max ≤ qdot ≤ v_max
  //   slacks ≥ 0  (no upper bound)
  const double inf = std::numeric_limits<double>::infinity();
  for (int i = 0; i < n_arm_; ++i) {
    l_bound_ds_(i) = -v_max_arm_(i);
    u_bound_ds_(i) = +v_max_arm_(i);
  }
  const int slack_start = n_arm_;
  const int slack_count = 2 * n_arm_ + max_capsules_;
  for (int i = 0; i < slack_count; ++i) {
    l_bound_ds_(slack_start + i) = 0.0;
    u_bound_ds_(slack_start + i) = inf;
  }
}

void ShieldQP::setIneqConstraint()
{
  // We encode all rows as `A x ≤ u`, i.e. lower bound = -inf.
  const double inf = std::numeric_limits<double>::infinity();
  A_ineq_ds_.setZero();
  l_ineq_ds_.setConstant(-inf);
  u_ineq_ds_.setConstant(inf);

  const int idx_qdot = 0;
  const int idx_s_qmin = n_arm_;
  const int idx_s_qmax = 2 * n_arm_;
  const int idx_s_col = 3 * n_arm_;

  // Row 0..n-1: joint upper limit CBF
  //   qdot ≤ α_jl (q_max - q) + s_qmax
  //   qdot - s_qmax ≤ α_jl (q_max - q)
  for (int i = 0; i < n_arm_; ++i) {
    A_ineq_ds_(i, idx_qdot + i) = +1.0;
    A_ineq_ds_(i, idx_s_qmax + i) = -1.0;
    u_ineq_ds_(i) = params_.alpha_jl * (q_upper_arm_(i) - q_arm_(i));
  }
  // Row n..2n-1: joint lower limit CBF
  //   -qdot ≤ α_jl (q - q_min) + s_qmin
  //   -qdot - s_qmin ≤ α_jl (q - q_min)
  for (int i = 0; i < n_arm_; ++i) {
    A_ineq_ds_(n_arm_ + i, idx_qdot + i) = -1.0;
    A_ineq_ds_(n_arm_ + i, idx_s_qmin + i) = -1.0;
    u_ineq_ds_(n_arm_ + i) = params_.alpha_jl * (q_arm_(i) - q_lower_arm_(i));
  }
  // Row 2n..2n+M-1: obstacle CBF for each capsule slot
  //   -gᵀ qdot - s_col_i ≤ α_obs (d_i - d_safe)
  // Inactive slots are filled with a row that's trivially satisfied: 0 ≤ +inf.
  for (int i = 0; i < max_capsules_; ++i) {
    const int row = 2 * n_arm_ + i;
    if (i < static_cast<int>(active_obstacles_.size())) {
      const auto & o = active_obstacles_[i];
      for (int j = 0; j < n_arm_; ++j) {
        A_ineq_ds_(row, idx_qdot + j) = -o.grad_arm(j);
      }
      A_ineq_ds_(row, idx_s_col + i) = -1.0;
      u_ineq_ds_(row) = params_.alpha_obs * (o.distance - params_.safe_distance);
    } else {
      // Inactive slot: 0 ≤ +inf
      A_ineq_ds_(row, idx_s_col + i) = -1.0;  // touches slack so OSQP keeps the row well-conditioned
      u_ineq_ds_(row) = +inf;
    }
  }
}

void ShieldQP::setEqConstraint()
{
  // No equality constraints.
}

}  // namespace ffw_safety_qp
