#pragma once

#include <Eigen/Dense>

#include <vector>

#include "optimization/qp_base.hpp"

namespace ffw_safety_qp
{

// One capsule's CBF input for the QP.
struct ObstacleConstraint
{
  // Distance d = ||P* - Q*|| - radius (signed; can be negative for penetration).
  double distance{0.0};
  // Gradient slice ∂d/∂q for the n controlled joints (size n_arm).
  Eigen::VectorXd grad_arm;
};

struct ShieldQPParams
{
  double dt{0.02};
  double alpha_obs{5.0};         // CBF gain for obstacle distance
  double alpha_jl{10.0};         // CBF gain for joint position limits
  double damping{0.01};          // ½ damping ‖qdot‖² regularizer
  double slack_penalty{1.0e4};   // penalty on each non-negative slack
  double safe_distance{0.02};    // d_safe in CBF formulation
};

// QP shield for a single arm (n_arm joints; default 7).
//
// Decision variables: x = [qdot (n_arm), s_qmin (n_arm), s_qmax (n_arm), s_col (max_capsules)]
//
// Cost:
//   ½ ‖qdot - qdot_des‖² + ½ damping · ‖qdot‖² + slack_penalty · 1ᵀ s
//
// Constraints (re-built each cycle):
//   - Bound (size n): -v_max ≤ qdot ≤ v_max
//   - Slack ≥ 0  (n_arm + n_arm + max_capsules rows)
//   - Joint upper CBF: qdot - α_jl(q_max - q) - s_qmax ≤ 0          (n rows)
//   - Joint lower CBF: -qdot - α_jl(q - q_min) - s_qmin ≤ 0          (n rows)
//   - Obstacle CBF (per slot, capped to max_capsules):
//       -gᵀ qdot - s_col_i ≤ α_obs (d_i - d_safe)
//     Inactive slots are filled with a trivially satisfied row.
class ShieldQP : public cyclo_motion_controller::optimization::QPBase
{
public:
  ShieldQP(int n_arm, int max_capsules);

  void setParams(const ShieldQPParams & params) {params_ = params;}

  // Per-cycle inputs.
  void setState(
    const Eigen::VectorXd & q_arm,
    const Eigen::VectorXd & q_lower_arm,
    const Eigen::VectorXd & q_upper_arm,
    const Eigen::VectorXd & v_max_arm);
  void setDesiredJointVel(const Eigen::VectorXd & qdot_des_arm);
  void setObstacles(const std::vector<ObstacleConstraint> & obstacles);

  // Solve. Returns true on success and writes qdot_safe (size n_arm).
  bool solve(Eigen::VectorXd & qdot_safe);

  int nArm() const {return n_arm_;}
  int maxCapsules() const {return max_capsules_;}
  int activeObstacleCount() const {return static_cast<int>(active_obstacles_.size());}

private:
  // QPBase declares these as private virtual; we override (private virtual is
  // overridable in C++ — the access specifier is checked at the call site only).
  void setCost() override;
  void setBoundConstraint() override;
  void setIneqConstraint() override;
  void setEqConstraint() override;

  int n_arm_;
  int max_capsules_;
  ShieldQPParams params_;

  Eigen::VectorXd q_arm_;
  Eigen::VectorXd q_lower_arm_;
  Eigen::VectorXd q_upper_arm_;
  Eigen::VectorXd v_max_arm_;
  Eigen::VectorXd qdot_des_arm_;
  std::vector<ObstacleConstraint> active_obstacles_;
};

}  // namespace ffw_safety_qp
