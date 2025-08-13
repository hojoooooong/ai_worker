from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ffw_test',
            executable='kdl_ik_solver_cpp',
            name='kdl_ik_solver_cpp',
            output='screen',
            parameters=[
                {'base_link': 'base_link'},
                {'end_effector_link': 'arm_r_link7'}
            ]
        )
    ])
