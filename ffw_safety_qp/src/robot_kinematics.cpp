#include "ffw_safety_qp/robot_kinematics.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>

#include <stdexcept>
#include <utility>

namespace ffw_safety_qp
{

namespace
{

Eigen::Matrix3d skew(const Eigen::Vector3d & v)
{
  Eigen::Matrix3d S;
  S <<     0, -v.z(),  v.y(),
       v.z(),     0, -v.x(),
      -v.y(),  v.x(),     0;
  return S;
}

}  // namespace

RobotKinematics::RobotKinematics(pinocchio::Model && model)
: model_(std::move(model)), data_(model_)
{
  // Build joint name → v-index map. Skip the universe joint at index 0.
  for (pinocchio::JointIndex j = 1; j < static_cast<pinocchio::JointIndex>(model_.njoints); ++j) {
    const std::string & name = model_.names[j];
    joint_name_to_v_idx_[name] = model_.idx_vs[j];
  }
}

RobotKinematics::RobotKinematics(const std::string & urdf_path)
: RobotKinematics(
    [&]() {
      pinocchio::Model m;
      pinocchio::urdf::buildModel(urdf_path, m);
      return m;
    }())
{
}

std::unique_ptr<RobotKinematics> RobotKinematics::fromXML(const std::string & urdf_xml)
{
  pinocchio::Model m;
  pinocchio::urdf::buildModelFromXML(urdf_xml, m);
  return std::unique_ptr<RobotKinematics>(new RobotKinematics(std::move(m)));
}

std::vector<std::string> RobotKinematics::jointNamesInVOrder() const
{
  std::vector<std::string> out(model_.nv);
  for (pinocchio::JointIndex j = 1; j < static_cast<pinocchio::JointIndex>(model_.njoints); ++j) {
    const int v = model_.idx_vs[j];
    const int nv = model_.nvs[j];
    for (int k = 0; k < nv; ++k) {
      out[v + k] = model_.names[j];
    }
  }
  return out;
}

int RobotKinematics::vIndexOfJoint(const std::string & joint_name) const
{
  auto it = joint_name_to_v_idx_.find(joint_name);
  if (it == joint_name_to_v_idx_.end()) {return -1;}
  return it->second;
}

pinocchio::FrameIndex RobotKinematics::frameId(const std::string & frame_name) const
{
  if (!model_.existFrame(frame_name)) {
    throw std::runtime_error("Frame not found in URDF: " + frame_name);
  }
  return model_.getFrameId(frame_name);
}

void RobotKinematics::updateFK(const Eigen::VectorXd & q)
{
  pinocchio::forwardKinematics(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);
}

void RobotKinematics::computeJointJacobians(const Eigen::VectorXd & q)
{
  pinocchio::computeJointJacobians(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);
}

Eigen::Affine3d RobotKinematics::framePose(const std::string & frame_name) const
{
  const auto fid = model_.getFrameId(frame_name);
  const auto & oMf = data_.oMf[fid];
  Eigen::Affine3d T = Eigen::Affine3d::Identity();
  T.linear() = oMf.rotation();
  T.translation() = oMf.translation();
  return T;
}

Eigen::MatrixXd RobotKinematics::jointJacobianLWA(const std::string & frame_name)
{
  const auto fid = model_.getFrameId(frame_name);
  Eigen::MatrixXd J(6, model_.nv);
  J.setZero();
  pinocchio::getFrameJacobian(model_, data_, fid, pinocchio::LOCAL_WORLD_ALIGNED, J);
  return J;
}

Eigen::MatrixXd RobotKinematics::pointLinearJacobian(
  const std::string & frame_name,
  const Eigen::Vector3d & p_world)
{
  // 6 x nv Jacobian of the FRAME (world-aligned). Linear part is velocity of the frame
  // origin in world, angular part is the angular velocity in world.
  const auto fid = model_.getFrameId(frame_name);
  Eigen::MatrixXd J6(6, model_.nv);
  J6.setZero();
  pinocchio::getFrameJacobian(model_, data_, fid, pinocchio::LOCAL_WORLD_ALIGNED, J6);
  // Frame origin in world.
  const Eigen::Vector3d origin = data_.oMf[fid].translation();
  const Eigen::Vector3d r = p_world - origin;
  // J_v(P) = J_v_origin - skew(r) * J_w
  Eigen::MatrixXd Jv(3, model_.nv);
  Jv = J6.topRows<3>() - skew(r) * J6.bottomRows<3>();
  return Jv;
}

}  // namespace ffw_safety_qp
