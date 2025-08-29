from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Launch arguments
    base_link_arg = DeclareLaunchArgument(
        'base_link',
        default_value='base_link',
        description='Base link name for the kinematic chain'
    )

    end_effector_link_arg = DeclareLaunchArgument(
        'end_effector_link',
        default_value='arm_r_link7',
        description='End effector link name for the kinematic chain'
    )

    target_pose_topic_arg = DeclareLaunchArgument(
        'target_pose_topic',
        default_value='/target_pose',
        description='Topic name for receiving target poses'
    )

    # VR scale arguments
    vr_scale_arg = DeclareLaunchArgument(
        'vr_scale',
        default_value='1.2',
        description='VR scale factor'
    )

    # VR Hand Pose Transformer node
    vr_hand_pose_transformer_node = Node(
        package='ffw_kinematics',
        executable='vr_hand_pose_transformer',
        name='vr_hand_pose_transformer',
        output='screen',
        parameters=[{
            'vr_scale': LaunchConfiguration('vr_scale'),
        }]
    )

    # Realtime IK Solver with Joint Limits node
    realtime_ik_solver_jl_node = Node(
        package='ffw_kinematics',
        executable='realtime_ik_solver_jl',
        name='realtime_ik_solver_jl',
        output='screen',
        parameters=[{
            'base_link': LaunchConfiguration('base_link'),
            'end_effector_link': LaunchConfiguration('end_effector_link'),
            'target_pose_topic': LaunchConfiguration('target_pose_topic'),
        }]
    )

    return LaunchDescription([
        base_link_arg,
        end_effector_link_arg,
        target_pose_topic_arg,
        vr_scale_arg,
        vr_hand_pose_transformer_node,
        realtime_ik_solver_jl_node,
    ])
