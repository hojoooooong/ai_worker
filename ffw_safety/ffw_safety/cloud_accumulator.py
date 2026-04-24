#!/usr/bin/env python3
"""Accumulate ZED point clouds into base_link frame during a head sweep.

Each incoming cloud is transformed to base_link, ROI-filtered, voxel-quantized,
and merged into a counter-keyed voxel set (each voxel tracks how many frames
observed it). On /safety/finish_accumulation, voxels seen in fewer than
`min_observations` frames are dropped as flying pixels; the filtered cloud is
then continuously republished on /safety/accumulated_cloud with transient_local
QoS so downstream consumers (e.g. leader_safety_filter) can pick it up.
"""

import threading

import numpy as np
import rclpy
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.time import Time
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformException, TransformListener


def _quaternion_to_rotation_matrix(qx, qy, qz, qw):
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    xw, yw, zw = qx * qw, qy * qw, qz * qw
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - zw),     2 * (xz + yw)],
        [2 * (xy + zw),     1 - 2 * (xx + zz), 2 * (yz - xw)],
        [2 * (xz - yw),     2 * (yz + xw),     1 - 2 * (xx + yy)],
    ], dtype=np.float64)


def _transform_to_matrix(tf) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    tr = tf.transform.translation
    rot = tf.transform.rotation
    T[:3, :3] = _quaternion_to_rotation_matrix(rot.x, rot.y, rot.z, rot.w)
    T[:3, 3] = [tr.x, tr.y, tr.z]
    return T


class CloudAccumulator(Node):

    def __init__(self):
        super().__init__('cloud_accumulator')

        self.declare_parameter(
            'input_topic', '/zedm/zed_node/point_cloud/cloud_registered')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('x_min', 0.2)
        self.declare_parameter('x_max', 1.5)
        self.declare_parameter('y_min', -1.0)
        self.declare_parameter('y_max', 1.0)
        self.declare_parameter('z_min', -0.5)
        self.declare_parameter('z_max', 1.5)
        self.declare_parameter('voxel_size', 0.01)
        self.declare_parameter('publish_rate', 2.0)
        self.declare_parameter('min_observations', 1)

        self.input_topic = self.get_parameter('input_topic').value
        self.base_frame = self.get_parameter('base_frame').value
        self.x_min = float(self.get_parameter('x_min').value)
        self.x_max = float(self.get_parameter('x_max').value)
        self.y_min = float(self.get_parameter('y_min').value)
        self.y_max = float(self.get_parameter('y_max').value)
        self.z_min = float(self.get_parameter('z_min').value)
        self.z_max = float(self.get_parameter('z_max').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.min_observations = int(
            self.get_parameter('min_observations').value)

        self.accum_keys = None
        self.accum_counts = None
        self._accum_lock = threading.Lock()

        self._cloud_cb_group = MutuallyExclusiveCallbackGroup()
        self._service_cb_group = ReentrantCallbackGroup()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(
            PointCloud2, self.input_topic, self.on_cloud, sub_qos,
            callback_group=self._cloud_cb_group)

        pub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(
            PointCloud2, '/safety/accumulated_cloud', pub_qos)

        self.timer = self.create_timer(
            1.0 / self.publish_rate, self.on_publish_tick,
            callback_group=self._service_cb_group)

        self.srv = self.create_service(
            Empty, '/safety/finish_accumulation', self.on_finish,
            callback_group=self._service_cb_group)

        self._shutting_down = False

        self.get_logger().info(
            f'cloud_accumulator started. input={self.input_topic}, '
            f'base={self.base_frame}, voxel={self.voxel_size} m, '
            f'ROI x=[{self.x_min},{self.x_max}] '
            f'y=[{self.y_min},{self.y_max}] '
            f'z=[{self.z_min},{self.z_max}], '
            f'min_observations={self.min_observations}. '
            'Call /safety/finish_accumulation to stop.')

    def on_cloud(self, msg: PointCloud2):
        if self._shutting_down:
            return

        data = point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if hasattr(data, 'dtype') and data.dtype.names is not None:
            pts = np.column_stack(
                [data['x'], data['y'], data['z']]).astype(np.float32)
        else:
            pts = np.array(list(data), dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 0:
            return

        finite_mask = np.isfinite(pts).all(axis=1)
        pts = pts[finite_mask]
        if pts.shape[0] == 0:
            return

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame, msg.header.frame_id,
                Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=0.5))
        except TransformException as e:
            self.get_logger().warn(
                f'TF lookup {self.base_frame} <- {msg.header.frame_id} '
                f'failed: {e}',
                throttle_duration_sec=2.0)
            return

        T = _transform_to_matrix(tf_msg)
        homog = np.hstack(
            [pts.astype(np.float64), np.ones((pts.shape[0], 1))])
        pts_base = (homog @ T.T)[:, :3]

        mask = (
            (pts_base[:, 0] >= self.x_min) & (pts_base[:, 0] <= self.x_max) &
            (pts_base[:, 1] >= self.y_min) & (pts_base[:, 1] <= self.y_max) &
            (pts_base[:, 2] >= self.z_min) & (pts_base[:, 2] <= self.z_max)
        )
        pts_roi = pts_base[mask]
        if pts_roi.shape[0] == 0:
            return

        keys = np.floor(pts_roi / self.voxel_size).astype(np.int64)
        new_keys = np.unique(keys, axis=0)
        new_counts = np.ones(len(new_keys), dtype=np.int64)

        with self._accum_lock:
            if self._shutting_down:
                return
            if self.accum_keys is None:
                self.accum_keys = new_keys
                self.accum_counts = new_counts
            else:
                combined_keys = np.vstack([self.accum_keys, new_keys])
                combined_counts = np.concatenate(
                    [self.accum_counts, new_counts])
                uniq_keys, inverse = np.unique(
                    combined_keys, axis=0, return_inverse=True)
                uniq_counts = np.bincount(
                    inverse, weights=combined_counts).astype(np.int64)
                self.accum_keys = uniq_keys
                self.accum_counts = uniq_counts

    def _active_keys(self):
        with self._accum_lock:
            if self.accum_keys is None:
                return None
            if self._shutting_down:
                mask = self.accum_counts >= self.min_observations
                return self.accum_keys[mask]
            return self.accum_keys

    def _build_cloud_msg(self, keys) -> PointCloud2:
        centers = (keys.astype(np.float32) + 0.5) * self.voxel_size
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.base_frame
        return point_cloud2.create_cloud_xyz32(header, centers.tolist())

    def on_publish_tick(self):
        keys = self._active_keys()
        if keys is None or len(keys) == 0:
            return
        self.pub.publish(self._build_cloud_msg(keys))

    def on_finish(self, request, response):
        with self._accum_lock:
            if self._shutting_down:
                return response
            self._shutting_down = True
            accum_keys = self.accum_keys
            accum_counts = self.accum_counts

        n_total = 0 if accum_keys is None else int(len(accum_keys))
        if n_total == 0:
            self.get_logger().warn(
                'No voxels accumulated; nothing to publish.')
            return response
        mask = accum_counts >= self.min_observations
        n_kept = int(mask.sum())
        self.get_logger().info(
            f'finish: {n_total} raw voxels; after consensus '
            f'(min_obs>={self.min_observations}) -> {n_kept} voxels.')
        if n_kept == 0:
            self.get_logger().warn(
                'No voxels met the consensus threshold. Lower '
                'min_observations or sweep more frames.')
            return response
        self.pub.publish(self._build_cloud_msg(accum_keys[mask]))
        self.get_logger().info(
            'Accumulation frozen. Filtered cloud will keep publishing at '
            f'{self.publish_rate} Hz. Ctrl+C to exit.')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CloudAccumulator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
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
