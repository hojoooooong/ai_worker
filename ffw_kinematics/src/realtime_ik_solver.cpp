#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolverpos_nr.hpp>
#include <kdl/frames.hpp>
#include <kdl_parser/kdl_parser.hpp>

#include <urdf/model.h>
#include <memory>
#include <vector>
#include <map>

class RealtimeIKSolver : public rclcpp::Node
{
public:
    RealtimeIKSolver() : Node("realtime_ik_solver"), 
                         lift_joint_index_(-1),
                         setup_complete_(false),
                         has_joint_states_(false)
    {
        // Parameters
        this->declare_parameter<std::string>("base_link", "base_link");
        this->declare_parameter<std::string>("end_effector_link", "arm_r_link7");
        this->declare_parameter<std::string>("target_pose_topic", "/target_pose");
        
        base_link_ = this->get_parameter("base_link").as_string();
        end_effector_link_ = this->get_parameter("end_effector_link").as_string();
        std::string target_pose_topic = this->get_parameter("target_pose_topic").as_string();
        
        RCLCPP_INFO(this->get_logger(), "🚀 Realtime IK Solver starting...");
        RCLCPP_INFO(this->get_logger(), "Base link: %s", base_link_.c_str());
        RCLCPP_INFO(this->get_logger(), "End effector link: %s", end_effector_link_.c_str());
        RCLCPP_INFO(this->get_logger(), "Target pose topic: %s", target_pose_topic.c_str());
        
        // Subscribers
        robot_description_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/robot_description", rclcpp::QoS(1).transient_local(),
            std::bind(&RealtimeIKSolver::robotDescriptionCallback, this, std::placeholders::_1));
            
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&RealtimeIKSolver::jointStateCallback, this, std::placeholders::_1));
            
        // Subscribe to target pose topic for real-time IK solving
        target_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            target_pose_topic, 10,
            std::bind(&RealtimeIKSolver::targetPoseCallback, this, std::placeholders::_1));
        
        // Publishers
        current_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/current_end_effector_pose", 10);
            
        joint_trajectory_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory", 10);
        
        // Timer for publishing current pose at 10Hz
        pose_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&RealtimeIKSolver::publishCurrentPose, this));
            
        // Try to get robot_description from parameter server
        auto param_client = std::make_shared<rclcpp::SyncParametersClient>(this, "/robot_state_publisher");
        
        if (param_client->wait_for_service(std::chrono::seconds(2))) {
            try {
                auto parameters = param_client->get_parameters({"robot_description"});
                if (!parameters.empty() && parameters[0].get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
                    std::string robot_description = parameters[0].as_string();
                    processRobotDescription(robot_description);
                }
            } catch (const std::exception& e) {
                RCLCPP_WARN(this->get_logger(), "Failed to get robot_description from parameter server: %s", e.what());
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "✅ Node initialized. Waiting for target poses on %s", target_pose_topic.c_str());
    }

private:
    void robotDescriptionCallback(const std_msgs::msg::String::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received robot_description via topic");
        processRobotDescription(msg->data);
    }
    
    void processRobotDescription(const std::string& robot_description)
    {
        RCLCPP_INFO(this->get_logger(), "Processing robot_description (%zu bytes)", robot_description.size());
        
        try {
            // Parse URDF
            urdf::Model model;
            if (!model.initString(robot_description)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to parse URDF");
                return;
            }
            
            RCLCPP_INFO(this->get_logger(), "URDF parsed successfully: %s", model.getName().c_str());
            
            // Create KDL tree
            KDL::Tree tree;
            if (!kdl_parser::treeFromUrdfModel(model, tree)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to create KDL tree from URDF");
                return;
            }
            
            RCLCPP_INFO(this->get_logger(), "KDL tree created with %d segments", tree.getNrOfSegments());
            
            // Extract chain from tree
            if (!tree.getChain(base_link_, end_effector_link_, chain_)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to extract chain from %s to %s", 
                           base_link_.c_str(), end_effector_link_.c_str());
                
                // Print available segments for debugging
                auto segments = tree.getSegments();
                RCLCPP_INFO(this->get_logger(), "Available segments:");
                for (const auto& seg : segments) {
                    RCLCPP_INFO(this->get_logger(), "  - %s", seg.first.c_str());
                }
                return;
            }
            
            RCLCPP_INFO(this->get_logger(), "KDL chain extracted with %d joints", chain_.getNrOfJoints());
            
            // Create solvers
            fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_);
            ik_vel_solver_ = std::make_unique<KDL::ChainIkSolverVel_pinv>(chain_);
            // Increase max iterations and relax tolerance for better convergence
            ik_solver_ = std::make_unique<KDL::ChainIkSolverPos_NR>(
                chain_, *fk_solver_, *ik_vel_solver_, 1000, 1e-3);
            
            // Extract joint names and identify lift_joint index
            joint_names_.clear();
            lift_joint_index_ = -1;
            
            for (unsigned int i = 0; i < chain_.getNrOfSegments(); i++) {
                KDL::Segment segment = chain_.getSegment(i);
                if (segment.getJoint().getType() != KDL::Joint::None) {
                    std::string joint_name = segment.getJoint().getName();
                    joint_names_.push_back(joint_name);
                    
                    // Check if this is the lift_joint
                    if (joint_name == "lift_joint") {
                        lift_joint_index_ = joint_names_.size() - 1;
                        RCLCPP_INFO(this->get_logger(), "Found lift_joint at index %d (will be fixed)", lift_joint_index_);
                    }
                }
            }
            
            RCLCPP_INFO(this->get_logger(), "Joint names extracted:");
            for (size_t i = 0; i < joint_names_.size(); i++) {
                RCLCPP_INFO(this->get_logger(), "  [%zu] %s%s", i, joint_names_[i].c_str(), 
                           (i == lift_joint_index_) ? " (LIFT - will be fixed)" : "");
            }
            
            setup_complete_ = true;
            RCLCPP_INFO(this->get_logger(), "✅ KDL setup completed successfully! Ready for real-time IK.");
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing robot description: %s", e.what());
        }
    }
    
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        if (!setup_complete_) {
            return;
        }
        
        // Extract joint positions for our chain
        current_joint_positions_.assign(joint_names_.size(), 0.0);
        bool all_joints_found = true;
        
        for (size_t i = 0; i < joint_names_.size(); i++) {
            bool found = false;
            for (size_t j = 0; j < msg->name.size(); j++) {
                if (msg->name[j] == joint_names_[i] && j < msg->position.size()) {
                    current_joint_positions_[i] = msg->position[j];
                    found = true;
                    break;
                }
            }
            if (!found) {
                all_joints_found = false;
                break;
            }
        }
        
        if (all_joints_found && !has_joint_states_) {
            has_joint_states_ = true;
            RCLCPP_INFO(this->get_logger(), "✅ Joint states received. System ready for target poses!");
        }
    }
    
    void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        if (!setup_complete_ || !has_joint_states_) {
            RCLCPP_WARN(this->get_logger(), "🚫 IK solver not ready yet, ignoring target pose");
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "🎯 Received target pose:");
        RCLCPP_INFO(this->get_logger(), "   Position: x=%.3f, y=%.3f, z=%.3f", 
                   msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        RCLCPP_INFO(this->get_logger(), "   Orientation: x=%.3f, y=%.3f, z=%.3f, w=%.3f",
                   msg->pose.orientation.x, msg->pose.orientation.y, 
                   msg->pose.orientation.z, msg->pose.orientation.w);
        
        // Solve IK immediately for the new target
        solveIKAndMove(*msg);
    }
    
    void solveIKAndMove(const geometry_msgs::msg::PoseStamped& target_pose)
    {
        RCLCPP_INFO(this->get_logger(), "🔧 Solving IK for target pose...");
        
        // Convert target pose to KDL Frame
        KDL::Frame target_frame;
        target_frame.p.x(target_pose.pose.position.x);
        target_frame.p.y(target_pose.pose.position.y);
        target_frame.p.z(target_pose.pose.position.z);
        
        // Convert quaternion to rotation matrix
        KDL::Rotation rot = KDL::Rotation::Quaternion(
            target_pose.pose.orientation.x,
            target_pose.pose.orientation.y,
            target_pose.pose.orientation.z,
            target_pose.pose.orientation.w
        );
        target_frame.M = rot;
        
        // Get current joint positions as initial guess
        KDL::JntArray q_init(chain_.getNrOfJoints());
        for (size_t i = 0; i < current_joint_positions_.size(); i++) {
            q_init(i) = current_joint_positions_[i];
        }
        
        KDL::JntArray q_result(chain_.getNrOfJoints());
        int ik_result = ik_solver_->CartToJnt(q_init, target_frame, q_result);
        
        // If IK failed, try with different initial guesses
        if (ik_result < 0) {
            RCLCPP_WARN(this->get_logger(), "IK failed with current guess (error: %d), trying alternatives...", ik_result);
            
            // Try with zero initial position
            KDL::JntArray q_zero(chain_.getNrOfJoints());
            for (unsigned int i = 0; i < chain_.getNrOfJoints(); i++) {
                q_zero(i) = 0.0;
            }
            ik_result = ik_solver_->CartToJnt(q_zero, target_frame, q_result);
            
            if (ik_result < 0) {
                // Try with small random perturbation
                KDL::JntArray q_perturb = q_init;
                for (unsigned int i = 0; i < chain_.getNrOfJoints(); i++) {
                    if ((int)i != lift_joint_index_) {  // Don't perturb lift joint
                        q_perturb(i) += (rand() % 1000 - 500) * 0.001;  // ±0.5 rad perturbation
                    }
                }
                ik_result = ik_solver_->CartToJnt(q_perturb, target_frame, q_result);
            }
        }
        
        // Fix lift_joint to its current position
        if (lift_joint_index_ >= 0 && lift_joint_index_ < (int)q_result.rows()) {
            q_result(lift_joint_index_) = current_joint_positions_[lift_joint_index_];
        }
        
        if (ik_result >= 0) {
            // Verify the solution
            KDL::Frame verify_frame;
            fk_solver_->JntToCart(q_result, verify_frame);
            
            KDL::Vector error_pos = verify_frame.p - target_frame.p;
            double error_magnitude = sqrt(
                error_pos.x() * error_pos.x() + 
                error_pos.y() * error_pos.y() + 
                error_pos.z() * error_pos.z());
            
            if (error_magnitude < 0.3) {  // 30cm tolerance
                RCLCPP_INFO(this->get_logger(), "✅ IK solution found (error: %.4fm). Moving robot...", error_magnitude);
                
                // Send joint trajectory command to move the robot
                sendJointTrajectory(q_result);
                
            } else {
                RCLCPP_WARN(this->get_logger(), "⚠️ Large verification error: %.4f m. Skipping movement.", error_magnitude);
            }
            
        } else {
            // Provide detailed error information
            std::string error_msg;
            switch(ik_result) {
                case -1: error_msg = "Failed to converge"; break;
                case -2: error_msg = "Undefined problem"; break;  
                case -3: error_msg = "Degraded gradient"; break;
                case -4: error_msg = "Singularity detected"; break;
                case -5: error_msg = "Maximum iterations exceeded"; break;
                default: error_msg = "Unknown error"; break;
            }
            
            RCLCPP_ERROR(this->get_logger(), "❌ IK failed: %d (%s)", ik_result, error_msg.c_str());
        }
    }
    
    void sendJointTrajectory(const KDL::JntArray& joint_positions)
    {
        // Create joint trajectory message (format matching real leader)
        auto traj_msg = trajectory_msgs::msg::JointTrajectory();
        // traj_msg.header.stamp = this->get_clock()->now();  // Use current time for real-time control
        traj_msg.header.frame_id = "";
        
        // Set joint names (arm joints + gripper, excluding lift_joint)
        std::vector<std::string> arm_joint_names;
        std::vector<double> target_arm_positions;
        
        for (size_t i = 0; i < joint_names_.size(); i++) {
            if (joint_names_[i] != "lift_joint") {  // Skip lift_joint
                arm_joint_names.push_back(joint_names_[i]);
                target_arm_positions.push_back(joint_positions(i));
            }
        }
        
        // Add gripper joint (gripper_r_joint1) with default open position
        arm_joint_names.push_back("gripper_r_joint1");
        target_arm_positions.push_back(0.1);  // Slightly open gripper
        
        traj_msg.joint_names = arm_joint_names;
        
        // Create single trajectory point with smooth timing
        auto point = trajectory_msgs::msg::JointTrajectoryPoint();
        point.positions = target_arm_positions;
        point.velocities = {};  // Empty velocities array
        point.accelerations = {};  // Empty accelerations array 
        point.effort = {};  // Empty effort array
        point.time_from_start.sec = 0;    // 3 second execution time for smoother movement
        point.time_from_start.nanosec = 0;
        
        traj_msg.points.push_back(point);
        
        // Publish the trajectory
        joint_trajectory_pub_->publish(traj_msg);
        
        RCLCPP_INFO(this->get_logger(), "📤 Joint trajectory sent! Robot should move in 1 second.");
    }
    
    void publishCurrentPose()
    {
        if (!setup_complete_ || !has_joint_states_) {
            return;
        }
        
        // Forward Kinematics to get current end-effector pose
        KDL::JntArray q(chain_.getNrOfJoints());
        for (size_t i = 0; i < current_joint_positions_.size(); i++) {
            q(i) = current_joint_positions_[i];
        }
        
        KDL::Frame end_effector_frame;
        int fk_result = fk_solver_->JntToCart(q, end_effector_frame);
        
        if (fk_result >= 0) {
            // Extract position and orientation
            KDL::Vector pos = end_effector_frame.p;
            double qx, qy, qz, qw;
            end_effector_frame.M.GetQuaternion(qx, qy, qz, qw);
            
            // Publish current pose
            auto pose_msg = geometry_msgs::msg::PoseStamped();
            pose_msg.header.stamp = this->get_clock()->now();
            pose_msg.header.frame_id = base_link_;
            pose_msg.pose.position.x = pos.x();
            pose_msg.pose.position.y = pos.y();
            pose_msg.pose.position.z = pos.z();
            pose_msg.pose.orientation.x = qx;
            pose_msg.pose.orientation.y = qy;
            pose_msg.pose.orientation.z = qz;
            pose_msg.pose.orientation.w = qw;
            current_pose_pub_->publish(pose_msg);
        }
    }

private:
    // Parameters
    std::string base_link_;
    std::string end_effector_link_;
    
    // ROS interfaces
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_description_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr current_pose_pub_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_pub_;
    rclcpp::TimerBase::SharedPtr pose_timer_;
    
    // KDL objects
    KDL::Chain chain_;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
    std::unique_ptr<KDL::ChainIkSolverVel_pinv> ik_vel_solver_;
    std::unique_ptr<KDL::ChainIkSolverPos_NR> ik_solver_;
    
    // Joint information
    std::vector<std::string> joint_names_;
    std::vector<double> current_joint_positions_;
    int lift_joint_index_;  // Index of lift_joint in the chain (-1 if not found)
    
    // Status flags
    bool setup_complete_;
    bool has_joint_states_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<RealtimeIKSolver>();
    
    RCLCPP_INFO(node->get_logger(), "🚀 Realtime IK Solver node started");
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}
