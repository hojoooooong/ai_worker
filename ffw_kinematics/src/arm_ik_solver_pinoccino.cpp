#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/spatial/explog.hpp>

#include <Eigen/Dense>
#include <chrono>

class PinocchioIKSolver : public rclcpp::Node
{
public:
    PinocchioIKSolver() : Node("pinocchio_ik_solver")
    {
        // Parameters
        this->declare_parameter("use_robot_description", true);
        this->declare_parameter("urdf_path", "/root/ros2_ws/src/ai_worker/ffw_description/urdf/ffw_bg2_rev4_follower/ffw_bg2_follower.urdf");
        this->declare_parameter("base_link", "base_link");
        this->declare_parameter("end_effector_link", "arm_r_link7");
        this->declare_parameter("max_iterations", 1000);
        this->declare_parameter("tolerance", 0.3);  // Relaxed tolerance
        this->declare_parameter("step_size", 0.01);
        this->declare_parameter("ik_damping", 1e-4);  // Damped least squares lambda
        this->declare_parameter("dt", 0.1);           // Integration step (s)
        // New tunables for flexibility
        this->declare_parameter("max_joint_velocity", 0.5); // rad/s equivalent per iteration
        this->declare_parameter("min_progress", 1e-6);      // convergence progress threshold
        this->declare_parameter("max_stagnation", 10);      // iterations with no progress before giving up
        this->declare_parameter("pos_weight", 1.0);          // weight for position error in log6 vector
        this->declare_parameter("ori_weight", 1.0);          // weight for orientation error in log6 vector

        use_robot_description_ = this->get_parameter("use_robot_description").as_bool();
        urdf_path_ = this->get_parameter("urdf_path").as_string();
        base_link_ = this->get_parameter("base_link").as_string();
        end_effector_link_ = this->get_parameter("end_effector_link").as_string();
        max_iterations_ = this->get_parameter("max_iterations").as_int();
        tolerance_ = this->get_parameter("tolerance").as_double();
        step_size_ = this->get_parameter("step_size").as_double();
        ik_damping_ = this->get_parameter("ik_damping").as_double();
        dt_ = this->get_parameter("dt").as_double();
        max_joint_velocity_ = this->get_parameter("max_joint_velocity").as_double();
        min_progress_ = this->get_parameter("min_progress").as_double();
        max_stagnation_ = this->get_parameter("max_stagnation").as_int();
        pos_weight_ = this->get_parameter("pos_weight").as_double();
        ori_weight_ = this->get_parameter("ori_weight").as_double();

        // Initialize Pinocchio model
        if (!initializePinocchioModel()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize Pinocchio model");
            return;
        }

        // Publishers and subscribers
        joint_trajectory_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory", 10);

        target_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/target_pose", 10,
            std::bind(&PinocchioIKSolver::targetPoseCallback, this, std::placeholders::_1));

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&PinocchioIKSolver::jointStateCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Pinocchio IK Solver initialized");
        RCLCPP_INFO(this->get_logger(), "Model has %ld DOFs", model_.nv);
        RCLCPP_INFO(this->get_logger(), "End effector frame ID: %ld", ee_frame_id_);
    }

private:
    bool initializePinocchioModel()
    {
        try {
            if (use_robot_description_) {
                // Try to get robot_description parameter
                if (!this->has_parameter("robot_description")) {
                    RCLCPP_WARN(this->get_logger(), "robot_description parameter not found, trying to declare it...");
                    this->declare_parameter("robot_description", "");
                }

                std::string robot_description;
                if (this->get_parameter("robot_description", robot_description) && !robot_description.empty()) {
                    RCLCPP_INFO(this->get_logger(), "Loading URDF from robot_description parameter");
                    // Load URDF from string
                    pinocchio::urdf::buildModelFromXML(robot_description, model_);
                } else {
                    RCLCPP_WARN(this->get_logger(), "robot_description parameter is empty, falling back to file path");
                    // Fallback to file path
                    pinocchio::urdf::buildModel(urdf_path_, model_);
                }
            } else {
                RCLCPP_INFO(this->get_logger(), "Loading URDF from file: %s", urdf_path_.c_str());
                // Load URDF from file
                pinocchio::urdf::buildModel(urdf_path_, model_);
            }

            data_ = pinocchio::Data(model_);

            RCLCPP_INFO(this->get_logger(), "Loaded URDF model with %ld joints", model_.njoints);

            // Find end effector frame
            if (model_.existFrame(end_effector_link_)) {
                ee_frame_id_ = model_.getFrameId(end_effector_link_);
                RCLCPP_INFO(this->get_logger(), "Found end effector frame: %s (ID: %ld)",
                           end_effector_link_.c_str(), ee_frame_id_);
            } else {
                RCLCPP_ERROR(this->get_logger(), "End effector frame '%s' not found in model",
                            end_effector_link_.c_str());
                return false;
            }

            // Print joint names for debugging
            for (size_t i = 1; i < model_.names.size(); ++i) {  // Skip universe joint (index 0)
                RCLCPP_INFO(this->get_logger(), "Joint %ld: %s", i, model_.names[i].c_str());
            }

            // Initialize joint configuration
            q_ = pinocchio::neutral(model_);

            // Define joints to exclude from IK (only use right arm joints for IK)
            excluded_joints_ = {
                "lift_joint",
                "wheel_fl_joint", "wheel_fr_joint", "wheel_bl_joint", "wheel_br_joint",
                "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4", "arm_l_joint5", "arm_l_joint6", "arm_l_joint7",
                "gripper_l_joint1", "gripper_l_joint2", "gripper_l_joint3", "gripper_l_joint4",
                "gripper_r_joint1", "gripper_r_joint2", "gripper_r_joint3", "gripper_r_joint4",
                "head_joint1", "head_joint2"
            };

            // Build mapping from joint names to indices
            for (size_t i = 1; i < model_.names.size(); ++i) {  // Skip universe joint
                joint_name_to_index_[model_.names[i]] = i;
                // Also map to configuration index (idx_q)
                joint_name_to_qidx_[model_.names[i]] = model_.joints[i].idx_q();

                // Check if this joint should be excluded from IK
                if (std::find(excluded_joints_.begin(), excluded_joints_.end(), model_.names[i]) == excluded_joints_.end()) {
                    active_joint_indices_.push_back(i);
                }
            }

            RCLCPP_INFO(this->get_logger(), "Active joints for IK: %ld", active_joint_indices_.size());
            for (auto idx : active_joint_indices_) {
                RCLCPP_INFO(this->get_logger(), "  - %s (index %ld)", model_.names[idx].c_str(), idx);
            }

            return true;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error initializing Pinocchio model: %s", e.what());
            return false;
        }
    }

    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        // Update current joint configuration (use configuration indices, not joint indices)
        for (size_t i = 0; i < msg->name.size(); ++i) {
            auto it = joint_name_to_index_.find(msg->name[i]);
            if (it != joint_name_to_index_.end()) {
                size_t joint_idx = it->second;
                size_t q_idx = model_.joints[joint_idx].idx_q();
                if (q_idx < static_cast<size_t>(q_.size())) {
                    q_[q_idx] = msg->position[i];
                }
            }
        }

        current_joint_state_ = *msg;
        has_joint_state_ = true;

        // Debug: Print received joint state for right arm joints
        static int count = 0;
        if (++count % 50 == 0) {  // Print every 50 messages to avoid spam
            RCLCPP_INFO(this->get_logger(), "Received joint states. Right arm positions:");
            for (const std::string& joint_name : {"arm_r_joint1", "arm_r_joint2", "arm_r_joint3",
                                                 "arm_r_joint4", "arm_r_joint5", "arm_r_joint6", "arm_r_joint7"}) {
                auto it = joint_name_to_index_.find(joint_name);
                if (it != joint_name_to_index_.end()) {
                    size_t q_idx = model_.joints[it->second].idx_q();
                    RCLCPP_INFO(this->get_logger(), "  %s: %.6f", joint_name.c_str(), q_[q_idx]);
                }
            }
        }
    }

    void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        if (!has_joint_state_) {
            RCLCPP_WARN(this->get_logger(), "No joint state received yet, ignoring target pose");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Joint state received: %s", has_joint_state_ ? "YES" : "NO");

        // First, compute forward kinematics with current joint state to see where end effector is
        pinocchio::forwardKinematics(model_, data_, q_);
        pinocchio::updateFramePlacements(model_, data_);
        pinocchio::SE3 current_ee_pose = data_.oMf[ee_frame_id_];

        RCLCPP_INFO(this->get_logger(), "Current end effector pose:");
        RCLCPP_INFO(this->get_logger(), "  Position: [%.3f, %.3f, %.3f]",
                   current_ee_pose.translation().x(), current_ee_pose.translation().y(), current_ee_pose.translation().z());

        Eigen::Quaterniond current_quat(current_ee_pose.rotation());
        RCLCPP_INFO(this->get_logger(), "  Orientation: [%.3f, %.3f, %.3f, %.3f]",
                   current_quat.w(), current_quat.x(), current_quat.y(), current_quat.z());

        auto start_time = std::chrono::high_resolution_clock::now();

        // Convert ROS pose to Pinocchio SE3
        pinocchio::SE3 target_pose;
        target_pose.translation() << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
        target_pose.rotation() = Eigen::Quaterniond(
            msg->pose.orientation.w,
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z
        ).toRotationMatrix();

        RCLCPP_INFO(this->get_logger(), "Received target pose:");
        RCLCPP_INFO(this->get_logger(), "  Position: [%.3f, %.3f, %.3f]",
                   msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        RCLCPP_INFO(this->get_logger(), "  Orientation: [%.3f, %.3f, %.3f, %.3f]",
                   msg->pose.orientation.w, msg->pose.orientation.x,
                   msg->pose.orientation.y, msg->pose.orientation.z);

    // Solve IK
    Eigen::VectorXd solution = q_;  // Start from current configuration
    bool success = solveIK(target_pose, solution);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        if (success) {
            RCLCPP_INFO(this->get_logger(), "IK solved successfully in %ld µs", duration.count());

            // Update current joint state with the solution
            q_ = solution;

            publishJointTrajectory(solution);

            // Verify the solution
            verifyIKSolution(solution, target_pose);
        } else {
            RCLCPP_ERROR(this->get_logger(), "IK failed to converge in %ld µs", duration.count());
        }
    }

    bool solveIK(const pinocchio::SE3& target_pose, Eigen::VectorXd& q_solution)
    {
        const double eps = tolerance_;
        const int max_iter = max_iterations_;

        Eigen::VectorXd q_iter = q_solution;
        double prev_error_norm = std::numeric_limits<double>::max();
        int stagnation_count = 0;
    const int max_stagnation = max_stagnation_;
    const double min_progress = min_progress_;

        // Build a mask over velocity space for active joints
        std::vector<uint8_t> vmask(model_.nv, 0);
        for (size_t j = 0; j < active_joint_indices_.size(); ++j) {
            size_t joint_idx = active_joint_indices_[j];
            size_t v_idx = model_.joints[joint_idx].idx_v();
            int nv = model_.joints[joint_idx].nv();
            for (int t = 0; t < nv; ++t) {
                if (v_idx + static_cast<size_t>(t) < vmask.size()) vmask[v_idx + static_cast<size_t>(t)] = 1;
            }
        }

        for (int i = 0; i < max_iter; ++i) {
            // Update kinematics
            pinocchio::forwardKinematics(model_, data_, q_iter);
            pinocchio::updateFramePlacements(model_, data_);

            // Current end-effector pose in world
            const pinocchio::SE3& oMe = data_.oMf[ee_frame_id_];

            // Error in LOCAL/body frame: iMd = oMe^{-1} * oMd
            pinocchio::SE3 iMd = oMe.inverse() * target_pose;
            Eigen::Matrix<double, 6, 1> err = pinocchio::log6(iMd).toVector();
            // Apply user-configurable weighting to position/orientation parts
            err.segment<3>(0) *= pos_weight_;
            err.segment<3>(3) *= ori_weight_;

            double error_norm = err.norm();
            if (i < 5 || i % 100 == 0) {
                RCLCPP_INFO(this->get_logger(), "Iteration %d: error norm = %.6f", i + 1, error_norm);
                RCLCPP_INFO(this->get_logger(), "  Position error: [%.6f, %.6f, %.6f]", err[0], err[1], err[2]);
                RCLCPP_INFO(this->get_logger(), "  Orientation error: [%.6f, %.6f, %.6f]", err[3], err[4], err[5]);
            }

            if (error_norm < eps) {
                q_solution = q_iter;
                RCLCPP_INFO(this->get_logger(), "🎯 IK CONVERGED in %d iterations! Final error: %.6f (tolerance: %.6f)", i + 1, error_norm, eps);
                return true;
            }

            if (std::abs(prev_error_norm - error_norm) < min_progress) {
                if (++stagnation_count >= max_stagnation) {
                    RCLCPP_WARN(this->get_logger(), "IK stagnated after %d iterations, error norm: %.6f", i + 1, error_norm);
                    return false;
                }
            } else {
                stagnation_count = 0;
            }
            prev_error_norm = error_norm;

            // Jacobian in LOCAL frame (matches body-frame error above)
            Eigen::MatrixXd J(6, model_.nv);
            pinocchio::computeFrameJacobian(model_, data_, q_iter, ee_frame_id_, pinocchio::LOCAL, J);

            if (i == 0) {
                RCLCPP_INFO(this->get_logger(), "Jacobian shape: %ld x %ld", J.rows(), J.cols());
            }

            // Damped least squares: v = J^T (J J^T + lambda I)^{-1} err
            Eigen::Matrix<double, 6, 6> JJt = J * J.transpose();
            JJt.diagonal().array() += ik_damping_;
            Eigen::Matrix<double, 6, 1> y = JJt.ldlt().solve(err);
            Eigen::VectorXd v = J.transpose() * y; // size nv

            // Mask velocities for non-active joints
            for (int k = 0; k < v.size(); ++k) {
                if (!vmask[static_cast<size_t>(k)]) v[k] = 0.0;
            }

            // Limit per-joint velocities (safety)
            const double max_joint_velocity = max_joint_velocity_;
            for (int k = 0; k < v.size(); ++k) {
                if (v[k] > max_joint_velocity) v[k] = max_joint_velocity;
                else if (v[k] < -max_joint_velocity) v[k] = -max_joint_velocity;
            }

            // Integrate in tangent space
            Eigen::VectorXd v_dt = v * dt_ * step_size_; // step_size_ acts as extra gain similar to Python's dt scaling
            q_iter = pinocchio::integrate(model_, q_iter, v_dt);
        }

        RCLCPP_WARN(this->get_logger(), "IK failed to converge after %d iterations", max_iter);
        return false;
    }

    void verifyIKSolution(const Eigen::VectorXd& q_solution, const pinocchio::SE3& target_pose)
    {
        // Forward kinematics with the solution
        pinocchio::forwardKinematics(model_, data_, q_solution);
        pinocchio::updateFramePlacements(model_, data_);

        pinocchio::SE3 achieved_pose = data_.oMf[ee_frame_id_];

        // Compute position and orientation errors
        Eigen::Vector3d pos_error = achieved_pose.translation() - target_pose.translation();

        // Convert rotations to quaternions for easier comparison
        Eigen::Quaterniond target_quat(target_pose.rotation());
        Eigen::Quaterniond achieved_quat(achieved_pose.rotation());

        // Compute angular error
        Eigen::Quaterniond quat_error = target_quat.inverse() * achieved_quat;
        double angle_error = 2.0 * std::acos(std::abs(quat_error.w()));

        RCLCPP_INFO(this->get_logger(), "IK Solution Verification:");
        RCLCPP_INFO(this->get_logger(), "  Target pos: [%.6f, %.6f, %.6f]",
                   target_pose.translation().x(), target_pose.translation().y(), target_pose.translation().z());
        RCLCPP_INFO(this->get_logger(), "  Achieved pos: [%.6f, %.6f, %.6f]",
                   achieved_pose.translation().x(), achieved_pose.translation().y(), achieved_pose.translation().z());
        RCLCPP_INFO(this->get_logger(), "  Position error: [%.6f, %.6f, %.6f] (norm: %.6f)",
                   pos_error.x(), pos_error.y(), pos_error.z(), pos_error.norm());
        RCLCPP_INFO(this->get_logger(), "  Angular error: %.6f rad (%.2f deg)", angle_error, angle_error * 180.0 / M_PI);

        // Print joint values for active joints
        RCLCPP_INFO(this->get_logger(), "Joint solution:");
        for (auto idx : active_joint_indices_) {
            size_t q_idx = model_.joints[idx].idx_q();
            RCLCPP_INFO(this->get_logger(), "  %s: %.6f", model_.names[idx].c_str(), q_solution[q_idx]);
        }
    }

    void publishJointTrajectory(const Eigen::VectorXd& q_solution)
    {
        trajectory_msgs::msg::JointTrajectory traj_msg;
        // traj_msg.header.stamp = this->now();
        traj_msg.header.frame_id = "";

        // Add only the right arm joints that we actually control (excluding lift_joint)
        std::vector<std::string> controlled_joints = {
            "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
            "arm_r_joint5", "arm_r_joint6", "arm_r_joint7"
        };

        for (const auto& joint_name : controlled_joints) {
            auto it = joint_name_to_index_.find(joint_name);
            if (it != joint_name_to_index_.end()) {
                traj_msg.joint_names.push_back(joint_name);
            }
        }

        // Add gripper joint with default open position
        traj_msg.joint_names.push_back("gripper_r_joint1");

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.time_from_start.sec = 0;    // Execute immediately like realtime_ik_solver_jl
        point.time_from_start.nanosec = 0;

        // Add arm joint positions
        for (const auto& joint_name : controlled_joints) {
            auto it = joint_name_to_index_.find(joint_name);
            if (it != joint_name_to_index_.end()) {
                size_t q_idx = model_.joints[it->second].idx_q();
                point.positions.push_back(q_solution[q_idx]);
            }
        }

        // Add gripper position (slightly open)
        point.positions.push_back(0.1);

        // Leave velocities, accelerations, and effort empty like realtime_ik_solver_jl
        point.velocities = {};
        point.accelerations = {};
        point.effort = {};

        traj_msg.points.push_back(point);
        joint_trajectory_pub_->publish(traj_msg);

        RCLCPP_INFO(this->get_logger(), "📤 Joint trajectory sent to leader! Robot should move immediately.");
        RCLCPP_INFO(this->get_logger(), "Published trajectory with %ld arm joints + gripper", controlled_joints.size());
    }

private:
    // ROS2 interfaces
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

    // Pinocchio model and data
    pinocchio::Model model_;
    pinocchio::Data data_;
    pinocchio::FrameIndex ee_frame_id_;

    // Configuration
    bool use_robot_description_;
    std::string urdf_path_;
    std::string base_link_;
    std::string end_effector_link_;
    int max_iterations_;
    double tolerance_;
    double step_size_;
    double ik_damping_;
    double dt_;
    double max_joint_velocity_;
    double min_progress_;
    int max_stagnation_;
    double pos_weight_;
    double ori_weight_;

    // State
    Eigen::VectorXd q_;  // Current joint configuration
    sensor_msgs::msg::JointState current_joint_state_;
    bool has_joint_state_ = false;

    // Joint management
    std::map<std::string, size_t> joint_name_to_index_;
    std::map<std::string, size_t> joint_name_to_qidx_;
    std::vector<size_t> active_joint_indices_;  // Joints that participate in IK
    std::vector<std::string> excluded_joints_;  // Joints excluded from IK
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PinocchioIKSolver>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
