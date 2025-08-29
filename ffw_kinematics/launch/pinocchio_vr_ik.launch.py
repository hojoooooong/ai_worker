from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Args
    target_pose_topic_arg = DeclareLaunchArgument(
        'target_pose_topic',
        default_value='/right_target_pose',
        description='Topic to consume as IK target (e.g., /right_target_pose or /left_target_pose)'
    )

    vr_scale_arg = DeclareLaunchArgument(
        'vr_scale',
        default_value='1.2',
        description='VR scale factor applied to wrist positions'
    )

    ik_damping_arg = DeclareLaunchArgument(
        'ik_damping',
        default_value='1e-4',
        description='Damped least-squares lambda'
    )

    dt_arg = DeclareLaunchArgument(
        'dt',
        default_value='0.1',
        description='Integration timestep (s)'
    )

    step_size_arg = DeclareLaunchArgument(
        'step_size',
        default_value='0.01',
        description='Extra IK step gain'
    )

    # New IK tuning args
    tolerance_arg = DeclareLaunchArgument(
        'tolerance',
        default_value='0.3',
        description='IK convergence tolerance (error norm)'
    )

    max_joint_velocity_arg = DeclareLaunchArgument(
        'max_joint_velocity',
        default_value='0.5',
        description='Max per-iteration joint velocity (rad/s equivalent)'
    )

    min_progress_arg = DeclareLaunchArgument(
        'min_progress',
        default_value='1e-6',
        description='Minimum error reduction between iterations'
    )

    max_stagnation_arg = DeclareLaunchArgument(
        'max_stagnation',
        default_value='10',
        description='Max iterations allowed with minimal progress'
    )

    pos_weight_arg = DeclareLaunchArgument(
        'pos_weight',
        default_value='1.0',
        description='Weight for position error (first 3 components)'
    )

    ori_weight_arg = DeclareLaunchArgument(
        'ori_weight',
        default_value='1.0',
        description='Weight for orientation error (last 3 components)'
    )

    # Load URDF into robot_description param
    urdf_path = os.path.join(
        get_package_share_directory('ffw_description'),
        'urdf', 'ffw_bg2_rev4_follower', 'ffw_bg2_follower.urdf'
    )
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    vr_node = Node(
        package='ffw_kinematics',
        executable='vr_hand_pose_transformer',
        name='vr_hand_pose_transformer',
        output='screen',
        parameters=[{
            'vr_scale': LaunchConfiguration('vr_scale'),
        }]
    )

    ik_node = Node(
        package='ffw_kinematics',
        executable='pinocchio_ik_solver',
        name='pinocchio_ik_solver',
        output='screen',
        parameters=[
            {'robot_description': robot_description},
            {'use_robot_description': True},
            {'end_effector_link': 'arm_r_link7'},
            {'max_iterations': 20000},
            {'tolerance': LaunchConfiguration('tolerance')},
            {'step_size': LaunchConfiguration('step_size')},
            {'ik_damping': LaunchConfiguration('ik_damping')},
            {'dt': LaunchConfiguration('dt')},
            {'max_joint_velocity': LaunchConfiguration('max_joint_velocity')},
            {'min_progress': LaunchConfiguration('min_progress')},
            {'max_stagnation': LaunchConfiguration('max_stagnation')},
            {'pos_weight': LaunchConfiguration('pos_weight')},
            {'ori_weight': LaunchConfiguration('ori_weight')},
        ],
        remappings=[
            ('/target_pose', LaunchConfiguration('target_pose_topic')),
        ]
    )

    return LaunchDescription([
        target_pose_topic_arg,
        vr_scale_arg,
        ik_damping_arg,
        dt_arg,
        step_size_arg,
    tolerance_arg,
    max_joint_velocity_arg,
    min_progress_arg,
    max_stagnation_arg,
    pos_weight_arg,
    ori_weight_arg,
        vr_node,
        ik_node,
    ])
