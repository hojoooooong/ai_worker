#include "ffw_safety_qp/leader_shield_node.hpp"

#include "ffw_safety_qp/capsule_cloud_distance.hpp"
#include "ffw_safety_qp/capsule_extractor.hpp"

#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

#include <sensor_msgs/point_cloud2_iterator.hpp>

using namespace std::chrono_literals;

namespace ffw_safety_qp
{

namespace
{

std::string readFileToString(const std::string & path)
{
  std::ifstream in(path);
  if (!in) {return {};}
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace

LeaderShieldNode::LeaderShieldNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("leader_shield", options)
{
  declareParameters();

  cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

  // QoS: leader trajectories are RELIABLE; cloud + robot_description are TRANSIENT_LOCAL.
  rclcpp::QoS reliable_qos(rclcpp::KeepLast(10));
  reliable_qos.reliable();

  rclcpp::QoS latched_qos(rclcpp::KeepLast(1));
  latched_qos.transient_local().reliable();

  // Robot description: try parameter, file, then live subscribe.
  std::string urdf_xml = this->get_parameter("urdf_xml").as_string();
  if (urdf_xml.empty() && !urdf_path_param_.empty()) {
    urdf_xml = readFileToString(urdf_path_param_);
  }
  if (!urdf_xml.empty()) {
    initFromRobotDescription(urdf_xml);
  } else {
    robot_desc_sub_ = this->create_subscription<std_msgs::msg::String>(
      "/robot_description", latched_qos,
      std::bind(&LeaderShieldNode::onRobotDescription, this, std::placeholders::_1));
  }

  // /joint_states
  js_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", reliable_qos,
    std::bind(&LeaderShieldNode::onJointStates, this, std::placeholders::_1));

  // Accumulated cloud
  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/safety/accumulated_cloud", latched_qos,
    std::bind(&LeaderShieldNode::onCloud, this, std::placeholders::_1));

  // Diagnostics publishers (ArmContext-level pubs are created per side in buildArmContext).
  markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/safety/capsule_markers", latched_qos);
  status_pub_ = this->create_publisher<std_msgs::msg::String>(
    "/safety/qp_status", reliable_qos);

  // 10 Hz marker republish for RViz.
  marker_timer_ = this->create_wall_timer(
    100ms,
    std::bind(&LeaderShieldNode::publishMarkers, this));

  RCLCPP_INFO(this->get_logger(), "leader_shield node started.");
}

void LeaderShieldNode::declareParameters()
{
  urdf_path_param_ = this->declare_parameter<std::string>("urdf_path", "");
  this->declare_parameter<std::string>("urdf_xml", "");
  base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");

  control_dt_ = this->declare_parameter<double>("control_dt", 0.02);
  buffer_ = this->declare_parameter<double>("buffer", 0.10);
  search_margin_ = this->declare_parameter<double>("search_margin", 0.05);

  qp_params_.dt = control_dt_;
  qp_params_.alpha_obs = this->declare_parameter<double>("alpha_obs", 5.0);
  qp_params_.alpha_jl = this->declare_parameter<double>("alpha_jl", 10.0);
  qp_params_.damping = this->declare_parameter<double>("damping", 0.01);
  qp_params_.slack_penalty = this->declare_parameter<double>("slack_penalty", 1.0e4);
  qp_params_.safe_distance = this->declare_parameter<double>("safe_distance", 0.02);

  max_capsules_per_side_ = this->declare_parameter<int>("max_capsules_per_side", 24);

  // Default whitelists: link3..link7 + end_effector. (Shoulder links 1-2 don't reach
  // the desk; excluding them halves the QP capsule load.)
  link_whitelist_left_ = this->declare_parameter<std::vector<std::string>>(
    "link_whitelist_left",
    std::vector<std::string>{
        "arm_l_link3", "arm_l_link4", "arm_l_link5",
        "arm_l_link6", "arm_l_link7", "end_effector_l_link"});
  link_whitelist_right_ = this->declare_parameter<std::vector<std::string>>(
    "link_whitelist_right",
    std::vector<std::string>{
        "arm_r_link3", "arm_r_link4", "arm_r_link5",
        "arm_r_link6", "arm_r_link7", "end_effector_r_link"});

  arm_joint_names_left_ = this->declare_parameter<std::vector<std::string>>(
    "arm_joint_names_left",
    std::vector<std::string>{
        "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
        "arm_l_joint5", "arm_l_joint6", "arm_l_joint7"});
  arm_joint_names_right_ = this->declare_parameter<std::vector<std::string>>(
    "arm_joint_names_right",
    std::vector<std::string>{
        "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
        "arm_r_joint5", "arm_r_joint6", "arm_r_joint7"});
}

void LeaderShieldNode::onRobotDescription(const std_msgs::msg::String::SharedPtr msg)
{
  if (kinematics_ready_) {return;}
  initFromRobotDescription(msg->data);
}

void LeaderShieldNode::initFromRobotDescription(const std::string & urdf_xml)
{
  try {
    kin_ = std::shared_ptr<RobotKinematics>(RobotKinematics::fromXML(urdf_xml).release());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      this->get_logger(), "Failed to build pinocchio model from URDF: %s", e.what());
    return;
  }

  q_meas_ = Eigen::VectorXd::Zero(kin_->nq());

  // Build /joint_states.name → v-index map (assuming actuated joints match Pinocchio).
  for (const auto & jn : kin_->jointNamesInVOrder()) {
    js_name_to_v_idx_[jn] = kin_->vIndexOfJoint(jn);
  }

  buildArmContext(Side::Left, arm_joint_names_left_);
  buildArmContext(Side::Right, arm_joint_names_right_);

  // Extract capsules per side.
  try {
    left_.capsules = extractCapsulesFromURDFString(urdf_xml, link_whitelist_left_);
    right_.capsules = extractCapsulesFromURDFString(urdf_xml, link_whitelist_right_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Capsule extraction failed: %s", e.what());
    return;
  }
  RCLCPP_INFO(
    this->get_logger(),
    "Extracted capsules: left=%zu, right=%zu (max_per_side=%d)",
    left_.capsules.size(), right_.capsules.size(), max_capsules_per_side_);

  if (static_cast<int>(left_.capsules.size()) > max_capsules_per_side_ ||
    static_cast<int>(right_.capsules.size()) > max_capsules_per_side_)
  {
    RCLCPP_WARN(
      this->get_logger(),
      "Capsule count exceeds max_capsules_per_side; later capsules will be unconstrained.");
  }

  kinematics_ready_ = true;
}

void LeaderShieldNode::buildArmContext(
  Side side,
  const std::vector<std::string> & arm_joint_names)
{
  ArmContext & c = ctx(side);
  c.arm_joint_names = arm_joint_names;
  c.v_idx.clear();
  c.v_idx.reserve(arm_joint_names.size());
  for (const auto & jn : arm_joint_names) {
    int idx = kin_->vIndexOfJoint(jn);
    if (idx < 0) {
      RCLCPP_ERROR(
        this->get_logger(), "Joint '%s' not found in URDF for side %s.",
        jn.c_str(), sideName(side).c_str());
      return;
    }
    c.v_idx.push_back(idx);
  }

  const int n = static_cast<int>(c.v_idx.size());
  c.q_lower_arm = Eigen::VectorXd(n);
  c.q_upper_arm = Eigen::VectorXd(n);
  c.v_max_arm = Eigen::VectorXd(n);
  Eigen::VectorXd qlb = kin_->qLower();
  Eigen::VectorXd qub = kin_->qUpper();
  Eigen::VectorXd vmax = kin_->vMax();
  for (int i = 0; i < n; ++i) {
    c.q_lower_arm(i) = qlb(c.v_idx[i]);
    c.q_upper_arm(i) = qub(c.v_idx[i]);
    c.v_max_arm(i) = vmax(c.v_idx[i]);
    if (!std::isfinite(c.v_max_arm(i)) || c.v_max_arm(i) <= 0) {
      c.v_max_arm(i) = 3.0;  // sensible default if URDF lacks a velocity limit
    }
  }

  c.qp = std::make_unique<ShieldQP>(n, max_capsules_per_side_);
  c.qp->setParams(qp_params_);

  // Per-side topics.
  rclcpp::QoS reliable_qos(rclcpp::KeepLast(10));
  reliable_qos.reliable();

  if (side == Side::Left) {
    c.leader_sub = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
      reliable_qos,
      [this](const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
        onLeaderTraj(Side::Left, msg);
      });
    c.shielded_pub = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/safety/qp_left/joint_trajectory", reliable_qos);
    c.min_dist_pub = this->create_publisher<std_msgs::msg::Float32>(
      "/safety/min_distance_left", reliable_qos);
  } else {
    c.leader_sub = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
      reliable_qos,
      [this](const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
        onLeaderTraj(Side::Right, msg);
      });
    c.shielded_pub = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/safety/qp_right/joint_trajectory", reliable_qos);
    c.min_dist_pub = this->create_publisher<std_msgs::msg::Float32>(
      "/safety/min_distance_right", reliable_qos);
  }
}

void LeaderShieldNode::onJointStates(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  if (!kinematics_ready_) {return;}
  std::lock_guard<std::mutex> lk(js_mtx_);
  if (q_meas_.size() != kin_->nq()) {
    q_meas_ = Eigen::VectorXd::Zero(kin_->nq());
  }
  for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i) {
    auto it = js_name_to_v_idx_.find(msg->name[i]);
    if (it == js_name_to_v_idx_.end() || it->second < 0) {continue;}
    q_meas_(it->second) = msg->position[i];
  }
  joint_state_ready_ = true;
}

int LeaderShieldNode::vIndex(const std::string & joint_name) const
{
  auto it = js_name_to_v_idx_.find(joint_name);
  if (it == js_name_to_v_idx_.end()) {return -1;}
  return it->second;
}

Eigen::VectorXd LeaderShieldNode::buildFullQ() const
{
  return q_meas_;
}

void LeaderShieldNode::onCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  // Ignore if same stamp (the accumulator republishes at 2 Hz; we rebuild only on change).
  if (cloud_ready_ &&
    last_cloud_stamp_.nanoseconds() == rclcpp::Time(msg->header.stamp).nanoseconds())
  {
    return;
  }

  auto pc = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pc->reserve(msg->width * msg->height);

  sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iy(*msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iz(*msg, "z");
  for (; ix != ix.end(); ++ix, ++iy, ++iz) {
    if (!std::isfinite(*ix) || !std::isfinite(*iy) || !std::isfinite(*iz)) {continue;}
    pc->push_back(pcl::PointXYZ(*ix, *iy, *iz));
  }

  auto kdt = std::make_shared<pcl::KdTreeFLANN<pcl::PointXYZ>>();
  if (!pc->empty()) {kdt->setInputCloud(pc);}

  {
    std::lock_guard<std::mutex> lk(cloud_mtx_);
    cloud_ = pc;
    kdtree_ = kdt;
    last_cloud_stamp_ = rclcpp::Time(msg->header.stamp);
    cloud_ready_ = !pc->empty();
  }

  RCLCPP_INFO(
    this->get_logger(), "Accumulated cloud rebuilt (%zu points; frame='%s').",
    pc->size(), msg->header.frame_id.c_str());
}

void LeaderShieldNode::onLeaderTraj(
  Side side,
  const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  if (!kinematics_ready_) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Kinematics not ready; dropping leader trajectory.");
    return;
  }
  if (!joint_state_ready_) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "joint_states not yet received; passing leader trajectory through.");
    ctx(side).shielded_pub->publish(*msg);
    return;
  }
  controlStep(side, *msg);
}

void LeaderShieldNode::controlStep(
  Side side,
  const trajectory_msgs::msg::JointTrajectory & leader_msg)
{
  ArmContext & c = ctx(side);
  if (!c.qp || c.v_idx.empty() || leader_msg.points.empty()) {
    c.shielded_pub->publish(leader_msg);
    return;
  }

  const int n = static_cast<int>(c.v_idx.size());

  // Snapshot full-DOF q.
  Eigen::VectorXd q;
  {
    std::lock_guard<std::mutex> lk(js_mtx_);
    q = q_meas_;
  }
  if (q.size() != kin_->nq()) {
    c.shielded_pub->publish(leader_msg);
    return;
  }

  // Map leader trajectory's first point joint targets into a full-DOF q_leader.
  Eigen::VectorXd q_leader = q;
  const auto & pt = leader_msg.points.front();
  for (size_t i = 0; i < leader_msg.joint_names.size() && i < pt.positions.size(); ++i) {
    int idx = vIndex(leader_msg.joint_names[i]);
    if (idx < 0) {continue;}
    q_leader(idx) = pt.positions[i];
  }

  // qdot_des in arm-only space (size n).
  Eigen::VectorXd q_arm(n);
  Eigen::VectorXd q_leader_arm(n);
  for (int i = 0; i < n; ++i) {
    q_arm(i) = q(c.v_idx[i]);
    q_leader_arm(i) = q_leader(c.v_idx[i]);
  }
  Eigen::VectorXd qdot_des_arm = (q_leader_arm - q_arm) / control_dt_;
  // Saturate at v_max for cost-shape conditioning.
  for (int i = 0; i < n; ++i) {
    qdot_des_arm(i) = std::clamp(qdot_des_arm(i), -c.v_max_arm(i), +c.v_max_arm(i));
  }

  // FK & joint Jacobians at current q (full-DOF).
  kin_->computeJointJacobians(q);

  // Build obstacle constraints from capsules.
  std::vector<ObstacleConstraint> obstacles;
  obstacles.reserve(c.capsules.size());
  double min_d = std::numeric_limits<double>::infinity();
  bool have_cloud = false;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_local;
  std::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZ>> kdtree_local;
  {
    std::lock_guard<std::mutex> lk(cloud_mtx_);
    cloud_local = cloud_;
    kdtree_local = kdtree_;
    have_cloud = cloud_ready_ && cloud_local && !cloud_local->empty() && kdtree_local;
  }

  if (have_cloud) {
    for (const auto & cap : c.capsules) {
      Eigen::Affine3d T;
      try {
        T = kin_->framePose(cap.frame);
      } catch (const std::exception & e) {
        continue;
      }
      const Eigen::Vector3d A = T * cap.p1_local;
      const Eigen::Vector3d B = T * cap.p2_local;
      auto qres = queryCapsule(A, B, cap.radius, *kdtree_local, *cloud_local,
          buffer_, search_margin_);
      if (qres.distance < min_d) {min_d = qres.distance;}
      if (!qres.active) {continue;}

      // Closest point on capsule axis at parameter t.
      Eigen::Vector3d Q = qres.closest_capsule_point;
      // Linear Jacobian of Q (rigidly attached to cap.frame's parent joint).
      Eigen::MatrixXd Jq = kin_->pointLinearJacobian(cap.frame, Q);
      // ∂d/∂q = nᵀ J(Q), where n points from P* toward Q*.
      Eigen::VectorXd grad_full = (qres.normal.transpose() * Jq).transpose();

      // Slice to arm columns only.
      ObstacleConstraint oc;
      oc.distance = qres.distance;
      oc.grad_arm = Eigen::VectorXd(n);
      for (int i = 0; i < n; ++i) {
        oc.grad_arm(i) = grad_full(c.v_idx[i]);
      }
      obstacles.push_back(oc);
    }
  }

  // Push to QP.
  c.qp->setState(q_arm, c.q_lower_arm, c.q_upper_arm, c.v_max_arm);
  c.qp->setDesiredJointVel(qdot_des_arm);
  c.qp->setObstacles(obstacles);

  Eigen::VectorXd qdot_safe;
  const auto t0 = std::chrono::steady_clock::now();
  bool ok = c.qp->solve(qdot_safe);
  const auto t1 = std::chrono::steady_clock::now();
  const double solve_ms =
    std::chrono::duration<double, std::milli>(t1 - t0).count();

  if (!ok) {
    qdot_safe = Eigen::VectorXd::Zero(n);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "[%s] QP infeasible/failed; freezing arm.", sideName(side).c_str());
  }

  // Integrate one step.
  Eigen::VectorXd q_target_arm = q_arm + qdot_safe * control_dt_;

  // Publish trajectory (one point at time_from_start = 2*dt, JTC-friendly).
  trajectory_msgs::msg::JointTrajectory out;
  out.header.stamp = this->now();
  out.header.frame_id = base_frame_;
  out.joint_names = c.arm_joint_names;
  trajectory_msgs::msg::JointTrajectoryPoint outpt;
  outpt.positions.assign(q_target_arm.data(), q_target_arm.data() + n);
  outpt.velocities.assign(qdot_safe.data(), qdot_safe.data() + n);
  outpt.time_from_start = rclcpp::Duration::from_seconds(2.0 * control_dt_);
  out.points.push_back(outpt);
  c.shielded_pub->publish(out);

  // Diagnostics.
  std_msgs::msg::Float32 dmin_msg;
  dmin_msg.data = static_cast<float>(std::isfinite(min_d) ? min_d : 1.0e3);
  c.min_dist_pub->publish(dmin_msg);

  std_msgs::msg::String s;
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(4)
     << "{\"side\":\"" << sideName(side) << "\""
     << ",\"solve_ms\":" << solve_ms
     << ",\"min_d\":" << (std::isfinite(min_d) ? min_d : 1.0e3)
     << ",\"active\":" << c.qp->activeObstacleCount()
     << ",\"infeasible\":" << (ok ? "false" : "true")
     << ",\"have_cloud\":" << (have_cloud ? "true" : "false")
     << "}";
  s.data = ss.str();
  status_pub_->publish(s);
}

void LeaderShieldNode::publishMarkers()
{
  if (!kinematics_ready_) {return;}

  visualization_msgs::msg::MarkerArray arr;
  int id = 0;

  Eigen::VectorXd q;
  {
    std::lock_guard<std::mutex> lk(js_mtx_);
    if (!joint_state_ready_) {return;}
    q = q_meas_;
  }
  kin_->updateFK(q);

  auto add_capsule = [&](const CapsuleSpec & cap, const std::string & ns,
      double r_color, double g_color, double b_color) {
      Eigen::Affine3d T;
      try {
        T = kin_->framePose(cap.frame);
      } catch (...) {return;}
      const Eigen::Vector3d A = T * cap.p1_local;
      const Eigen::Vector3d B = T * cap.p2_local;

      visualization_msgs::msg::Marker m;
      m.header.frame_id = base_frame_;
      m.header.stamp = this->now();
      m.ns = ns;
      m.id = id++;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.color.r = r_color;
      m.color.g = g_color;
      m.color.b = b_color;
      m.color.a = 0.4f;

      const double seg_len = (B - A).norm();
      if (seg_len < 1e-6) {
        // Sphere only.
        m.type = visualization_msgs::msg::Marker::SPHERE;
        m.pose.position.x = A.x();
        m.pose.position.y = A.y();
        m.pose.position.z = A.z();
        m.pose.orientation.w = 1.0;
        const double d = 2.0 * cap.radius;
        m.scale.x = d; m.scale.y = d; m.scale.z = d;
        arr.markers.push_back(m);
      } else {
        // Cylinder body.
        m.type = visualization_msgs::msg::Marker::CYLINDER;
        Eigen::Vector3d mid = 0.5 * (A + B);
        m.pose.position.x = mid.x();
        m.pose.position.y = mid.y();
        m.pose.position.z = mid.z();
        // Quaternion from z-axis to (B-A) direction.
        Eigen::Vector3d zaxis(0, 0, 1);
        Eigen::Vector3d dir = (B - A).normalized();
        Eigen::Quaterniond qrot;
        qrot.setFromTwoVectors(zaxis, dir);
        m.pose.orientation.x = qrot.x();
        m.pose.orientation.y = qrot.y();
        m.pose.orientation.z = qrot.z();
        m.pose.orientation.w = qrot.w();
        const double d = 2.0 * cap.radius;
        m.scale.x = d; m.scale.y = d; m.scale.z = seg_len;
        arr.markers.push_back(m);

        // End-cap spheres.
        visualization_msgs::msg::Marker s1 = m;
        s1.id = id++;
        s1.type = visualization_msgs::msg::Marker::SPHERE;
        s1.pose.position.x = A.x();
        s1.pose.position.y = A.y();
        s1.pose.position.z = A.z();
        s1.pose.orientation.w = 1.0;
        s1.scale.x = d; s1.scale.y = d; s1.scale.z = d;
        arr.markers.push_back(s1);

        visualization_msgs::msg::Marker s2 = s1;
        s2.id = id++;
        s2.pose.position.x = B.x();
        s2.pose.position.y = B.y();
        s2.pose.position.z = B.z();
        arr.markers.push_back(s2);
      }
    };

  for (const auto & cap : left_.capsules) {
    add_capsule(cap, "shield_capsules_left", 0.0, 0.7, 0.2);
  }
  for (const auto & cap : right_.capsules) {
    add_capsule(cap, "shield_capsules_right", 0.0, 0.4, 0.9);
  }
  markers_pub_->publish(arr);
}

}  // namespace ffw_safety_qp
