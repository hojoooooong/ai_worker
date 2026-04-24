#!/usr/bin/env python3
"""Predictive capsule-based safety gate for leader-teleop.

Observes (never publishes to) /leader/.../joint_trajectory. For each
incoming message, predicts where each arm capsule would land by running
forward kinematics on the target joint positions (parsed from the URDF
on startup). Each capsule (p1, p2, radius) is tested against the height
map derived from /safety/accumulated_cloud: if any xy cell covered by
the capsule would have its capsule_min_z below ceiling[cell] +
safety_margin, the node calls /controller_manager/switch_controller to
deactivate the corresponding arm controller. When the next leader
trajectory is fully safe again, the node reactivates it and the arm
resumes tracking the leader.

The filter is purely side-by-side: it does not intercept the leader
topic, it does not publish trajectories. Running it is opt-in; without
it the follower controllers subscribe directly to the leader (as in the
original follower launch), so teleop works normally.

Capsule visualization: MarkerArray is published on /safety/capsule_markers
at 10 Hz so the user can tune capsule dimensions visually in RViz. Green
means safe; red means the capsule that is currently in violation.

Permissive fallbacks:
    - URDF not yet parsed → no deactivation
    - No cloud → no deactivation
    - Capsule has no cell coverage in ceiling → does not constrain
    - switch_controller service unavailable → warn and skip

Pure numpy FK and capsule math. No PyKDL / FCL dependencies. Capsule
definitions loaded from YAML (see ffw_safety/config/capsules.yaml).
"""

import threading
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
import yaml
from controller_manager_msgs.srv import SwitchController
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Empty, String
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker, MarkerArray


def _rot_rpy(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rot_axis_angle(axis, angle):
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.eye(3)
    a = axis / n
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([
        [0.0, -a[2], a[1]],
        [a[2], 0.0, -a[0]],
        [-a[1], a[0], 0.0],
    ])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _parse_urdf_joints_by_parent(xml_str):
    root = ET.fromstring(xml_str)
    jbp = {}
    for j in root.findall('joint'):
        parent = j.find('parent').get('link')
        child = j.find('child').get('link')
        name = j.get('name')
        jtype = j.get('type')
        origin = j.find('origin')
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()]
            rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
        axis_el = j.find('axis')
        axis = [1.0, 0.0, 0.0]
        if axis_el is not None:
            axis = [float(v) for v in axis_el.get('xyz', '1 0 0').split()]
        info = {
            'name': name,
            'type': jtype,
            'child': child,
            'axis': np.array(axis, dtype=np.float64),
            'xyz': np.array(xyz, dtype=np.float64),
            'rpy': np.array(rpy, dtype=np.float64),
        }
        jbp.setdefault(parent, []).append(info)
    return jbp


def _build_chain(joints_by_parent, base_link, tip_link):
    def dfs(current):
        if current == tip_link:
            return []
        for j in joints_by_parent.get(current, []):
            sub = dfs(j['child'])
            if sub is not None:
                return [j] + sub
        return None

    chain = dfs(base_link)
    if chain is None:
        raise RuntimeError(
            f'No URDF chain from {base_link} to {tip_link}.')
    return chain


def _fk(chain, joint_positions):
    T = np.eye(4)
    for info in chain:
        T_fixed = np.eye(4)
        T_fixed[:3, :3] = _rot_rpy(info['rpy'])
        T_fixed[:3, 3] = info['xyz']
        T = T @ T_fixed
        if info['type'] in ('revolute', 'continuous'):
            q = joint_positions.get(info['name'], 0.0)
            T_joint = np.eye(4)
            T_joint[:3, :3] = _rot_axis_angle(info['axis'], q)
            T = T @ T_joint
        elif info['type'] == 'prismatic':
            q = joint_positions.get(info['name'], 0.0)
            T_joint = np.eye(4)
            T_joint[:3, 3] = info['axis'] * q
            T = T @ T_joint
    return T


def _quat_from_vectors(v_from, v_to):
    """Shortest-arc rotation quaternion (x,y,z,w) mapping v_from → v_to."""
    a = np.asarray(v_from, dtype=np.float64)
    b = np.asarray(v_to, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    a = a / na
    b = b / nb
    dot = float(a @ b)
    if dot > 0.999999:
        return [0.0, 0.0, 0.0, 1.0]
    if dot < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return [float(axis[0]), float(axis[1]), float(axis[2]), 0.0]
    cross = np.cross(a, b)
    w = 1.0 + dot
    q = np.array([cross[0], cross[1], cross[2], w])
    q = q / np.linalg.norm(q)
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


class LeaderSafetyFilter(Node):

    def __init__(self):
        super().__init__('leader_safety_filter')

        self.declare_parameter('safety_margin', 0.015)
        self.declare_parameter('voxel_size', 0.01)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('capsule_config_file', '')
        self.declare_parameter('left_controller', 'arm_l_controller')
        self.declare_parameter('right_controller', 'arm_r_controller')
        self.declare_parameter(
            'switch_controller_service',
            '/controller_manager/switch_controller')
        self.declare_parameter('cloud_topic', '/safety/accumulated_cloud')
        self.declare_parameter(
            'robot_description_topic', '/robot_description')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter(
            'left_input_topic',
            '/leader/joint_trajectory_command_broadcaster_left/'
            'joint_trajectory')
        self.declare_parameter(
            'right_input_topic',
            '/leader/joint_trajectory_command_broadcaster_right/'
            'joint_trajectory')
        self.declare_parameter('marker_topic', '/safety/capsule_markers')
        self.declare_parameter('marker_publish_rate', 10.0)
        self.declare_parameter(
            'left_resync_topic', '/leader/resync_left')
        self.declare_parameter(
            'right_resync_topic', '/leader/resync_right')

        p = self.get_parameter
        self.safety_margin = float(p('safety_margin').value)
        self.voxel_size = float(p('voxel_size').value)
        self.base_frame = p('base_frame').value
        self.left_controller = p('left_controller').value
        self.right_controller = p('right_controller').value
        capsule_file = p('capsule_config_file').value

        self.capsules = {'left': [], 'right': []}
        if capsule_file:
            try:
                self._load_capsules_from_file(capsule_file)
            except Exception as e:
                self.get_logger().error(
                    f'Failed to load capsule config {capsule_file}: {e}')
        if not self.capsules['left'] and not self.capsules['right']:
            self.get_logger().error(
                'No capsules configured. Set capsule_config_file '
                'parameter to a valid yaml path.')

        self.ceiling = {}
        self.ceiling_cells = None
        self.ceiling_z = None
        self.ceiling_lock = threading.Lock()
        self.last_joint_state = {}
        self.js_lock = threading.Lock()
        self.violation = {'left': False, 'right': False}
        self.violating_idx = {'left': None, 'right': None}
        self.chains_ready = {'left': False, 'right': False}

        cb = ReentrantCallbackGroup()

        latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        default_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(
            String, p('robot_description_topic').value,
            self.on_robot_description, latched_qos, callback_group=cb)
        self.create_subscription(
            PointCloud2, p('cloud_topic').value, self.on_cloud, latched_qos,
            callback_group=cb)
        self.create_subscription(
            JointState, p('joint_states_topic').value,
            self.on_joint_states, default_qos, callback_group=cb)
        self.create_subscription(
            JointTrajectory, p('left_input_topic').value,
            lambda m: self.on_traj('left', m), default_qos,
            callback_group=cb)
        self.create_subscription(
            JointTrajectory, p('right_input_topic').value,
            lambda m: self.on_traj('right', m), default_qos,
            callback_group=cb)

        self.switch_cli = self.create_client(
            SwitchController, p('switch_controller_service').value,
            callback_group=cb)

        self.pub_markers = self.create_publisher(
            MarkerArray, p('marker_topic').value, latched_qos)

        self.resync_pubs = {
            'left': self.create_publisher(
                Empty, p('left_resync_topic').value, default_qos),
            'right': self.create_publisher(
                Empty, p('right_resync_topic').value, default_qos),
        }

        marker_rate = float(p('marker_publish_rate').value)
        if marker_rate > 0.0:
            self.create_timer(
                1.0 / marker_rate, self._publish_capsule_markers,
                callback_group=cb)

        total = len(self.capsules['left']) + len(self.capsules['right'])
        self.get_logger().info(
            f'leader_safety_filter started. margin={self.safety_margin} m, '
            f'voxel={self.voxel_size} m. {total} capsules loaded '
            f"({len(self.capsules['left'])} left + "
            f"{len(self.capsules['right'])} right). "
            'Awaiting /robot_description and /safety/accumulated_cloud.')

    def _load_capsules_from_file(self, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        for side in ('left', 'right'):
            for entry in (data.get(side) or []):
                self.capsules[side].append({
                    'frame': entry['frame'],
                    'p1': np.array(entry['p1'], dtype=np.float64),
                    'p2': np.array(entry['p2'], dtype=np.float64),
                    'radius': float(entry['radius']),
                    'chain': None,     # filled on URDF receipt
                    'p1_world': None,
                    'p2_world': None,
                })
        self.get_logger().info(
            f"Loaded capsules: left={len(self.capsules['left'])}, "
            f"right={len(self.capsules['right'])}.")

    def on_robot_description(self, msg: String):
        if self.chains_ready['left'] and self.chains_ready['right']:
            return
        try:
            jbp = _parse_urdf_joints_by_parent(msg.data)
        except Exception as e:
            self.get_logger().error(f'URDF parse failed: {e}')
            return
        for side in ('left', 'right'):
            for cap in self.capsules[side]:
                try:
                    cap['chain'] = _build_chain(
                        jbp, self.base_frame, cap['frame'])
                except Exception as e:
                    self.get_logger().error(
                        f"Chain build failed for {side}/{cap['frame']}: {e}")
                    cap['chain'] = None
            self.chains_ready[side] = all(
                cap['chain'] is not None for cap in self.capsules[side])
        self.get_logger().info(
            f"URDF parsed. chains_ready={self.chains_ready}")

    def on_cloud(self, msg: PointCloud2):
        data = point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if hasattr(data, 'dtype') and data.dtype.names is not None:
            pts = np.column_stack(
                [data['x'], data['y'], data['z']]).astype(np.float64)
        else:
            pts = np.array(list(data), dtype=np.float64).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] == 0:
            return
        cells = np.floor(pts[:, :2] / self.voxel_size).astype(np.int64)
        ceiling = {}
        for i in range(pts.shape[0]):
            key = (int(cells[i, 0]), int(cells[i, 1]))
            z = float(pts[i, 2])
            cur = ceiling.get(key)
            if cur is None or z > cur:
                ceiling[key] = z
        ceiling_cells = np.array(list(ceiling.keys()), dtype=np.int64)
        ceiling_z = np.array(
            [ceiling[tuple(c)] for c in ceiling_cells], dtype=np.float64)
        with self.ceiling_lock:
            self.ceiling = ceiling
            self.ceiling_cells = ceiling_cells
            self.ceiling_z = ceiling_z
        self.get_logger().info(
            f'height map built: {len(ceiling)} cells from {pts.shape[0]} '
            'points.')

    def on_joint_states(self, msg: JointState):
        with self.js_lock:
            self.last_joint_state = {
                n: p for n, p in zip(msg.name, msg.position)
            }

    def _capsule_vs_ceiling(self, p1, p2, radius):
        """Return violation dict or None."""
        with self.ceiling_lock:
            cells = self.ceiling_cells
            z_arr = self.ceiling_z
        if cells is None or len(cells) == 0:
            return None

        xmin = min(p1[0], p2[0]) - radius
        xmax = max(p1[0], p2[0]) + radius
        ymin = min(p1[1], p2[1]) - radius
        ymax = max(p1[1], p2[1]) + radius

        cx_all = (cells[:, 0] + 0.5) * self.voxel_size
        cy_all = (cells[:, 1] + 0.5) * self.voxel_size

        bb_mask = (
            (cx_all >= xmin) & (cx_all <= xmax) &
            (cy_all >= ymin) & (cy_all <= ymax)
        )
        if not np.any(bb_mask):
            return None

        cx = cx_all[bb_mask]
        cy = cy_all[bb_mask]
        zc = z_arr[bb_mask]
        cbb = cells[bb_mask]

        dxy = p2[:2] - p1[:2]
        seg_sq = float(dxy @ dxy)
        if seg_sq < 1e-12:
            t = np.zeros(len(cx))
        else:
            t = ((cx - p1[0]) * dxy[0] + (cy - p1[1]) * dxy[1]) / seg_sq
            t = np.clip(t, 0.0, 1.0)

        closest_x = p1[0] + t * dxy[0]
        closest_y = p1[1] + t * dxy[1]
        d = np.sqrt((closest_x - cx) ** 2 + (closest_y - cy) ** 2)

        covered = d < radius
        if not np.any(covered):
            return None

        closest_z = p1[2] + t * (p2[2] - p1[2])
        z_offset = np.sqrt(np.maximum(radius ** 2 - d ** 2, 0.0))
        capsule_min_z = closest_z - z_offset

        threshold = zc + self.safety_margin
        violations = covered & (capsule_min_z < threshold)
        if not np.any(violations):
            return None

        idx = int(np.argmax(violations))
        return {
            'cell': (int(cbb[idx, 0]), int(cbb[idx, 1])),
            'capsule_min_z': float(capsule_min_z[idx]),
            'ceiling_z': float(zc[idx]),
        }

    def on_traj(self, side: str, msg: JointTrajectory):
        controller = (self.left_controller if side == 'left'
                      else self.right_controller)
        if not self.chains_ready[side]:
            self._ensure_active(side, controller)
            return
        with self.ceiling_lock:
            has_ceiling = self.ceiling_cells is not None and len(
                self.ceiling_cells) > 0
        if not has_ceiling:
            self._ensure_active(side, controller)
            return
        if not msg.points:
            return

        target = dict(
            zip(msg.joint_names, msg.points[0].positions))
        with self.js_lock:
            q = dict(self.last_joint_state)
        q.update(target)

        unsafe = False
        hit = None
        for i, cap in enumerate(self.capsules[side]):
            try:
                T = _fk(cap['chain'], q)
            except Exception as e:
                self.get_logger().warn(
                    f"FK failed on {side}/{cap['frame']}: {e}",
                    throttle_duration_sec=2.0)
                continue
            p1w = T[:3, :3] @ cap['p1'] + T[:3, 3]
            p2w = T[:3, :3] @ cap['p2'] + T[:3, 3]
            # cache for marker publishing (target pose, not current)
            cap['p1_world'] = p1w
            cap['p2_world'] = p2w
            violation = self._capsule_vs_ceiling(p1w, p2w, cap['radius'])
            if violation is not None:
                unsafe = True
                hit = {'capsule': cap, 'index': i, **violation}
                break

        if unsafe and not self.violation[side]:
            self.get_logger().warn(
                f"Violation {side}: deactivating {controller}. "
                f"capsule={hit['capsule']['frame']} (idx={hit['index']}) "
                f"min_z={hit['capsule_min_z']:.3f} < "
                f"ceiling+margin={(hit['ceiling_z'] + self.safety_margin):.3f} "
                f"at cell {hit['cell']}",
                throttle_duration_sec=1.0)
            self._set_controller_active(controller, False)
            self.violation[side] = True
            self.violating_idx[side] = hit['index']
        elif not unsafe and self.violation[side]:
            self.get_logger().info(
                f'Safe again {side}: reactivating {controller} + resync.')
            # Resync first so broadcaster's next publish uses adaptive
            # time_from_start; then re-activate the controller so it
            # actually executes.
            self.resync_pubs[side].publish(Empty())
            self._set_controller_active(controller, True)
            self.violation[side] = False
            self.violating_idx[side] = None
        elif unsafe and self.violation[side]:
            # still violating — update which capsule is currently offending
            self.violating_idx[side] = hit['index']

    def _ensure_active(self, side: str, controller: str):
        if self.violation[side]:
            self.get_logger().info(
                f'Permissive ({side}): reactivating {controller} + resync.')
            self.resync_pubs[side].publish(Empty())
            self._set_controller_active(controller, True)
            self.violation[side] = False
            self.violating_idx[side] = None

    def _set_controller_active(self, name: str, active: bool):
        if not self.switch_cli.service_is_ready():
            self.get_logger().warn(
                f'switch_controller service not ready; cannot set '
                f'{name} {"active" if active else "inactive"}.',
                throttle_duration_sec=5.0)
            return
        req = SwitchController.Request()
        if active:
            req.activate_controllers = [name]
        else:
            req.deactivate_controllers = [name]
        req.strictness = SwitchController.Request.BEST_EFFORT
        self.switch_cli.call_async(req)

    def _publish_capsule_markers(self):
        if not (self.chains_ready['left'] or self.chains_ready['right']):
            return
        with self.js_lock:
            q = dict(self.last_joint_state)
        if not q:
            return

        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for side in ('left', 'right'):
            if not self.chains_ready[side]:
                continue
            for i, cap in enumerate(self.capsules[side]):
                try:
                    T = _fk(cap['chain'], q)
                except Exception:
                    continue
                p1w = T[:3, :3] @ cap['p1'] + T[:3, 3]
                p2w = T[:3, :3] @ cap['p2'] + T[:3, 3]
                is_violating = (self.violating_idx[side] == i)
                marker_array.markers.extend(
                    self._build_capsule_markers(
                        side, i, cap, p1w, p2w, is_violating, stamp))
        self.pub_markers.publish(marker_array)

    def _build_capsule_markers(self, side, idx, cap, p1w, p2w,
                               is_violating, stamp):
        markers = []
        ns = f'capsule_{side}'
        if is_violating:
            rgba = (1.0, 0.15, 0.15, 0.55)
        else:
            rgba = (0.15, 0.85, 0.15, 0.45)

        direction = p2w - p1w
        length = float(np.linalg.norm(direction))
        if length > 1e-6:
            center = 0.5 * (p1w + p2w)
            quat = _quat_from_vectors([0.0, 0.0, 1.0], direction)
            m = Marker()
            m.header.frame_id = self.base_frame
            m.header.stamp = stamp
            m.ns = ns
            m.id = 3 * idx
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(center[0])
            m.pose.position.y = float(center[1])
            m.pose.position.z = float(center[2])
            m.pose.orientation.x = quat[0]
            m.pose.orientation.y = quat[1]
            m.pose.orientation.z = quat[2]
            m.pose.orientation.w = quat[3]
            m.scale.x = 2.0 * cap['radius']
            m.scale.y = 2.0 * cap['radius']
            m.scale.z = length
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            markers.append(m)

        for cap_idx, pw in enumerate((p1w, p2w)):
            m = Marker()
            m.header.frame_id = self.base_frame
            m.header.stamp = stamp
            m.ns = ns
            m.id = 3 * idx + 1 + cap_idx
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pw[0])
            m.pose.position.y = float(pw[1])
            m.pose.position.z = float(pw[2])
            m.pose.orientation.w = 1.0
            m.scale.x = 2.0 * cap['radius']
            m.scale.y = 2.0 * cap['radius']
            m.scale.z = 2.0 * cap['radius']
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            markers.append(m)

        return markers

    def shutdown_safely(self):
        """Best-effort reactivate any controllers we deactivated."""
        for side, viol in list(self.violation.items()):
            if viol:
                controller = (self.left_controller if side == 'left'
                              else self.right_controller)
                self.get_logger().info(
                    f'Shutdown: reactivating {controller} + resync.')
                try:
                    self.resync_pubs[side].publish(Empty())
                except Exception:
                    pass
                self._set_controller_active(controller, True)


def main(args=None):
    rclpy.init(args=args)
    node = LeaderSafetyFilter()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.shutdown_safely()
        except Exception:
            pass
        try:
            executor.shutdown(timeout_sec=2.0)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    main()
