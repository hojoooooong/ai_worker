#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/multibody/model.hpp>

namespace ffw_safety_qp
{

// Lightweight Pinocchio wrapper used by ShieldQP and the distance routine.
// We don't use cyclo_motion_controller_core::KinematicsSolver here because we need
// direct access to model_/data_ for getJointJacobian + frame parent joint id.
class RobotKinematics
{
public:
  // Build from a URDF file path. Throws on parse failure.
  explicit RobotKinematics(const std::string & urdf_path);

  // Build from a URDF XML string.
  static std::unique_ptr<RobotKinematics> fromXML(const std::string & urdf_xml);

  // Number of generalized velocities (nv). For a fixed-base manipulator nv == nq.
  int nv() const {return model_.nv;}
  int nq() const {return model_.nq;}

  const pinocchio::Model & model() const {return model_;}
  pinocchio::Data & data() {return data_;}

  // Names of joints in the order Pinocchio uses for v / q vectors.
  // For a fixed-base model, returned size is nv (and skips the universe joint).
  std::vector<std::string> jointNamesInVOrder() const;

  // Joint position limits in q-vector order, size nq.
  Eigen::VectorXd qLower() const {return model_.lowerPositionLimit;}
  Eigen::VectorXd qUpper() const {return model_.upperPositionLimit;}
  // Velocity limits in v-vector order, size nv.
  Eigen::VectorXd vMax() const {return model_.velocityLimit;}

  // Look up a joint's index in the v vector. Returns -1 if not found.
  int vIndexOfJoint(const std::string & joint_name) const;

  // Look up frame id by name. Throws if missing.
  pinocchio::FrameIndex frameId(const std::string & frame_name) const;

  // Update FK (forwardKinematics + updateFramePlacements) for the given q.
  // Must be called before computePose / pointJacobian / linkJacobian for that q.
  void updateFK(const Eigen::VectorXd & q);

  // World-frame pose of a URDF link/frame at the q used in the most recent updateFK.
  Eigen::Affine3d framePose(const std::string & frame_name) const;

  // 6 x nv Jacobian of the joint that owns `frame_name`, expressed in
  // LOCAL_WORLD_ALIGNED (world rotation, frame translation).
  // Caller must have called updateFK(q) and computeJointJacobians.
  Eigen::MatrixXd jointJacobianLWA(const std::string & frame_name);

  // 3 x nv linear Jacobian of an arbitrary point P given in world coordinates,
  // attached rigidly to the joint that owns `frame_name`. Uses the rigid-body
  // Jacobian shift J_v(P) = J_v_origin + (-skew(r)) * J_w, where r = P - origin(joint).
  Eigen::MatrixXd pointLinearJacobian(
    const std::string & frame_name,
    const Eigen::Vector3d & p_world);

  // Recompute joint jacobians for current q (call after updateFK or together).
  void computeJointJacobians(const Eigen::VectorXd & q);

private:
  explicit RobotKinematics(pinocchio::Model && model);

  pinocchio::Model model_;
  pinocchio::Data data_;
  std::unordered_map<std::string, int> joint_name_to_v_idx_;
};

}  // namespace ffw_safety_qp
