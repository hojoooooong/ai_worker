#include "ffw_safety_qp/capsule_extractor.hpp"

#include <urdf_parser/urdf_parser.h>
#include <urdf_model/model.h>

#include <rclcpp/rclcpp.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ffw_safety_qp
{

namespace
{

Eigen::Affine3d poseToAffine(const urdf::Pose & pose)
{
  Eigen::Quaterniond q(pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z);
  Eigen::Affine3d T = Eigen::Affine3d::Identity();
  T.linear() = q.toRotationMatrix();
  T.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  return T;
}

CapsuleList extractFromModel(
  const urdf::ModelInterfaceSharedPtr & model,
  const std::vector<std::string> & link_whitelist)
{
  static rclcpp::Logger logger = rclcpp::get_logger("ffw_safety_qp.capsule_extractor");

  CapsuleList out;
  for (const auto & link_name : link_whitelist) {
    auto link = model->getLink(link_name);
    if (!link) {
      RCLCPP_WARN(logger, "Link '%s' not found in URDF; skipping.", link_name.c_str());
      continue;
    }

    // urdf::Link stores both a singular `collision` (first element) and
    // `collision_array`. We iterate the array; if it is empty but a singular
    // `collision` exists, fall back to that single element.
    std::vector<urdf::CollisionSharedPtr> cols(
      link->collision_array.begin(), link->collision_array.end());
    if (cols.empty() && link->collision) {
      cols.push_back(link->collision);
    }

    if (cols.empty()) {
      RCLCPP_WARN(logger, "Link '%s' has no <collision> elements; skipping.", link_name.c_str());
      continue;
    }

    for (const auto & col : cols) {
      if (!col || !col->geometry) {continue;}
      const Eigen::Affine3d T = poseToAffine(col->origin);
      const auto type = col->geometry->type;

      if (type == urdf::Geometry::CYLINDER) {
        auto cyl = std::dynamic_pointer_cast<urdf::Cylinder>(col->geometry);
        if (!cyl) {continue;}
        const double half = 0.5 * cyl->length;
        Eigen::Vector3d p1 = T * Eigen::Vector3d(0.0, 0.0, -half);
        Eigen::Vector3d p2 = T * Eigen::Vector3d(0.0, 0.0, +half);
        out.push_back(CapsuleSpec{link_name, p1, p2, cyl->radius});
      } else if (type == urdf::Geometry::SPHERE) {
        auto sph = std::dynamic_pointer_cast<urdf::Sphere>(col->geometry);
        if (!sph) {continue;}
        Eigen::Vector3d c = T.translation();
        out.push_back(CapsuleSpec{link_name, c, c, sph->radius});
      } else if (type == urdf::Geometry::BOX || type == urdf::Geometry::MESH) {
        RCLCPP_WARN(
          logger,
          "Link '%s' has a <box>/<mesh> collision element; skipped (only cylinder/sphere supported).",
          link_name.c_str());
      }
    }
  }
  return out;
}

std::string readFileToString(const std::string & path)
{
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open URDF file: " + path);
  }
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace

CapsuleList extractCapsulesFromURDF(
  const std::string & urdf_path,
  const std::vector<std::string> & link_whitelist)
{
  return extractCapsulesFromURDFString(readFileToString(urdf_path), link_whitelist);
}

CapsuleList extractCapsulesFromURDFString(
  const std::string & urdf_xml,
  const std::vector<std::string> & link_whitelist)
{
  auto model = urdf::parseURDF(urdf_xml);
  if (!model) {
    throw std::runtime_error("Failed to parse URDF XML");
  }
  return extractFromModel(model, link_whitelist);
}

}  // namespace ffw_safety_qp
