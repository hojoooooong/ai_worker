#pragma once

#include "ffw_safety_qp/capsule_spec.hpp"

#include <string>
#include <vector>

namespace ffw_safety_qp
{

// Extract capsules (cylinder + sphere primitives) from the <collision> blocks
// of the given links in a URDF. <box>/<mesh> primitives are skipped with a warning.
//
// Cylinder (length L, radius r) at origin T → segment from T·(0,0,-L/2) to T·(0,0,+L/2),
//   radius r.
// Sphere (radius r) at origin T → degenerate capsule with p1==p2==T.translation(),
//   radius r.
//
// Throws std::runtime_error if URDF cannot be parsed.
CapsuleList extractCapsulesFromURDF(
  const std::string & urdf_path,
  const std::vector<std::string> & link_whitelist);

// Same as above, but parses URDF from a string (e.g. /robot_description content).
CapsuleList extractCapsulesFromURDFString(
  const std::string & urdf_xml,
  const std::vector<std::string> & link_whitelist);

}  // namespace ffw_safety_qp
