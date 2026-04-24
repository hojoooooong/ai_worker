#!/usr/bin/env python3
"""Automate head sweep + finish call for cloud_accumulator.

Pre-waits for the accumulator, commands the head through a
center -> left -> center -> right -> center sweep (pitched down) via
/head_controller/follow_joint_trajectory, post-waits, then calls
/safety/finish_accumulation. Exits when done; the accumulator stays alive
holding the latched cloud.

While sweeping, the leader's joystick_controller is temporarily
deactivated via /leader/controller_manager/switch_controller so its
continuous head commands don't preempt our action goal. If the leader
isn't running (service unavailable), we silently skip and proceed — scan
still works because nothing is competing for head control anyway.
"""

import threading
import time

import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class HeadScanSweeper(Node):

    def __init__(self):
        super().__init__('head_scan_sweeper')

        self.declare_parameter(
            'head_action_topic',
            '/head_controller/follow_joint_trajectory')
        self.declare_parameter(
            'finish_service', '/safety/finish_accumulation')
        self.declare_parameter('head_pitch', 0.71)
        self.declare_parameter('yaw_amplitude', 0.8)
        self.declare_parameter('leg_duration', 2.0)
        self.declare_parameter('num_sweeps', 1)
        self.declare_parameter('pre_wait', 3.0)
        self.declare_parameter('post_wait', 1.0)
        self.declare_parameter(
            'leader_switch_service',
            '/leader/controller_manager/switch_controller')
        self.declare_parameter(
            'leader_joystick_controller', 'joystick_controller')

        self.head_action_topic = self.get_parameter('head_action_topic').value
        self.finish_service = self.get_parameter('finish_service').value
        self.head_pitch = float(self.get_parameter('head_pitch').value)
        self.yaw_amplitude = float(self.get_parameter('yaw_amplitude').value)
        self.leg_duration = float(self.get_parameter('leg_duration').value)
        self.num_sweeps = int(self.get_parameter('num_sweeps').value)
        self.pre_wait = float(self.get_parameter('pre_wait').value)
        self.post_wait = float(self.get_parameter('post_wait').value)
        self.leader_switch_service = self.get_parameter(
            'leader_switch_service').value
        self.leader_joystick_controller = self.get_parameter(
            'leader_joystick_controller').value

        self.action_client = ActionClient(
            self, FollowJointTrajectory, self.head_action_topic)
        self.finish_cli = self.create_client(Empty, self.finish_service)
        self.leader_switch_cli = self.create_client(
            SwitchController, self.leader_switch_service)
        self._joystick_paused = False

    def run_sequence(self):
        # Suspend leader's joystick_controller (best-effort) so it doesn't
        # preempt our head action goals. If leader isn't running the call
        # just logs and moves on.
        self._pause_leader_joystick()
        try:
            self._run_scan()
        finally:
            self._resume_leader_joystick()

    def _run_scan(self):
        self.get_logger().info(
            f'Pre-wait {self.pre_wait:.1f}s for accumulator...')
        time.sleep(self.pre_wait)

        self.get_logger().info(
            f'Waiting for action server {self.head_action_topic}...')
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                'Head action server not available; aborting.')
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = self._build_trajectory()
        total = goal.trajectory.points[-1].time_from_start
        total_s = total.sec + total.nanosec * 1e-9
        self.get_logger().info(
            f'Sweeping: pitch={self.head_pitch}, yaw=+/-{self.yaw_amplitude}, '
            f'{self.num_sweeps} cycle(s), total {total_s:.1f}s.')

        send_future = self.action_client.send_goal_async(goal)
        self._wait_future(send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Head goal rejected; aborting.')
            return

        result_future = goal_handle.get_result_async()
        self._wait_future(result_future, timeout=total_s + 10.0)
        result = result_future.result()
        status = result.status if result is not None else -1
        status_names = {4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}
        status_name = status_names.get(status, 'UNKNOWN/timeout')
        self.get_logger().info(
            f'Head sweep finished. Status: {status} ({status_name}).')
        if status != 4:
            self.get_logger().warn(
                'Head did not reach SUCCEEDED. If ABORTED, leader or another '
                'publisher likely preempted the goal on '
                f'{self.head_action_topic.replace("/follow_joint_trajectory", "")}.')

        self.get_logger().info(f'Post-wait {self.post_wait:.1f}s...')
        time.sleep(self.post_wait)

        self.get_logger().info(f'Calling {self.finish_service}...')
        if not self.finish_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Finish service not available.')
            return
        call_future = self.finish_cli.call_async(Empty.Request())
        self._wait_future(call_future)
        self.get_logger().info('Accumulation frozen. Sweeper exiting.')

    def _pause_leader_joystick(self):
        if not self.leader_switch_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                f'{self.leader_switch_service} not available; '
                'skipping leader joystick pause (leader may not be up).')
            return
        req = SwitchController.Request()
        req.deactivate_controllers = [self.leader_joystick_controller]
        req.strictness = SwitchController.Request.BEST_EFFORT
        future = self.leader_switch_cli.call_async(req)
        self._wait_future(future, timeout=2.0)
        self._joystick_paused = True
        self.get_logger().info(
            f'Leader {self.leader_joystick_controller} deactivated for '
            'scan duration.')

    def _resume_leader_joystick(self):
        if not self._joystick_paused:
            return
        if not self.leader_switch_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                f'{self.leader_switch_service} not available at '
                'resume; leader joystick may remain deactivated.')
            return
        req = SwitchController.Request()
        req.activate_controllers = [self.leader_joystick_controller]
        req.strictness = SwitchController.Request.BEST_EFFORT
        future = self.leader_switch_cli.call_async(req)
        self._wait_future(future, timeout=2.0)
        self._joystick_paused = False
        self.get_logger().info(
            f'Leader {self.leader_joystick_controller} reactivated.')

    def _wait_future(self, future, timeout=30.0):
        end = time.time() + timeout
        while rclpy.ok() and not future.done() and time.time() < end:
            time.sleep(0.05)

    def _build_trajectory(self):
        traj = JointTrajectory()
        traj.joint_names = ['head_joint1', 'head_joint2']
        t = 0.0

        def append(yaw):
            nonlocal t
            t += self.leg_duration
            p = JointTrajectoryPoint()
            p.positions = [self.head_pitch, yaw]
            sec = int(t)
            nsec = int(round((t - sec) * 1e9))
            p.time_from_start = DurationMsg(sec=sec, nanosec=nsec)
            traj.points.append(p)

        append(0.0)
        for _ in range(self.num_sweeps):
            append(self.yaw_amplitude)
            append(0.0)
            append(-self.yaw_amplitude)
            append(0.0)
        return traj


def main(args=None):
    rclpy.init(args=args)
    node = HeadScanSweeper()

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run_sequence()
    except KeyboardInterrupt:
        pass
    finally:
        # Best-effort resume in case run_sequence was cut short before
        # its own finally ran (very unlikely but cheap).
        try:
            node._resume_leader_joystick()
        except Exception:
            pass
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
