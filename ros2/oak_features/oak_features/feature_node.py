"""ROS2 node publishing keypoints with binary descriptors from an OAK pipeline.

Thin wrapper over the ROS-agnostic core in <example>/core/.
"""

import os
import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from oak_features_msgs.msg import Keypoint, KeypointArray
from sensor_msgs.msg import Image


def default_core_path():
    """Example root: two levels up from this package's source directory.

    Falls back to the OAK_FEATURES_CORE environment variable.
    """
    env = os.environ.get("OAK_FEATURES_CORE")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


class FeatureNode(Node):
    def __init__(self):
        super().__init__("oak_features")

        self.declare_parameter("core_path", default_core_path())
        self.declare_parameter("archive", "depthai_models/descriptor64_itq_strip32_slim.tar.xz")
        self.declare_parameter("max_keypoints", 64)
        self.declare_parameter("nn_nodes", 2)
        self.declare_parameter("tracker_decimate", 2)
        self.declare_parameter("motion_estimator", "sw")
        self.declare_parameter("tracking", True)
        self.declare_parameter("describe_gate", False)
        self.declare_parameter("change_thresh", 6.0)
        self.declare_parameter("refresh_every", 30)
        self.declare_parameter("fps_limit", 0)
        self.declare_parameter("compression", "itq")
        self.declare_parameter("publish_debug_image", False)
        self.declare_parameter("frame_id", "oak_camera")

        core_path = self.get_parameter("core_path").get_parameter_value().string_value
        archive = self.get_parameter("archive").get_parameter_value().string_value
        max_keypoints = (
            self.get_parameter("max_keypoints").get_parameter_value().integer_value
        )
        nn_nodes = self.get_parameter("nn_nodes").get_parameter_value().integer_value
        tracker_decimate = (
            self.get_parameter("tracker_decimate").get_parameter_value().integer_value
        )
        motion_estimator = (
            self.get_parameter("motion_estimator").get_parameter_value().string_value
        )
        tracking = self.get_parameter("tracking").get_parameter_value().bool_value
        describe_gate = (
            self.get_parameter("describe_gate").get_parameter_value().bool_value
        )
        change_thresh = (
            self.get_parameter("change_thresh").get_parameter_value().double_value
        )
        refresh_every = (
            self.get_parameter("refresh_every").get_parameter_value().integer_value
        )
        fps_limit = self.get_parameter("fps_limit").get_parameter_value().integer_value
        self.compression_name = (
            self.get_parameter("compression").get_parameter_value().string_value.lower()
        )
        self.publish_debug_image = (
            self.get_parameter("publish_debug_image")
            .get_parameter_value()
            .bool_value
        )
        self.frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )

        compression_map = {
            "subsample": KeypointArray.COMPRESSION_SUBSAMPLE,
            "xorfold": KeypointArray.COMPRESSION_XORFOLD,
            "lsh": KeypointArray.COMPRESSION_LSH,
            "itq": KeypointArray.COMPRESSION_ITQ,
        }
        if self.compression_name not in compression_map:
            self.get_logger().fatal(
                "Unknown 'compression' value '%s'; expected one of %s"
                % (self.compression_name, sorted(compression_map.keys()))
            )
            raise ValueError("invalid compression parameter")
        self.compression_const = compression_map[self.compression_name]

        if not os.path.isabs(archive):
            archive = os.path.join(core_path, archive)

        if core_path not in sys.path:
            sys.path.insert(0, core_path)
        from core.pipeline import FeaturePipeline, synthetic_texture_source
        from core.packer import pack_frame

        self._pack_frame = pack_frame
        # Scene-independent mode for automated/containerized verification:
        # replaces the camera with a synthetic moving texture fed host-side.
        frame_source = (
            synthetic_texture_source()
            if os.environ.get("OAK_FEATURES_SYNTHETIC_FRAMES") == "1"
            else None
        )
        self._pipeline = FeaturePipeline(
            archive_path=archive,
            max_keypoints=max_keypoints,
            fps_limit=fps_limit if fps_limit > 0 else None,
            frame_source=frame_source,
            nn_nodes=nn_nodes,
            tracker_decimate=tracker_decimate,
            motion_estimator=motion_estimator,
            tracking=tracking,
            describe_gate=describe_gate,
            change_thresh=change_thresh,
            refresh_every=refresh_every,
        )

        self.features_pub = self.create_publisher(
            KeypointArray, "/oak/features", qos_profile_sensor_data
        )
        self.debug_pub = None
        if self.publish_debug_image:
            self.debug_pub = self.create_publisher(
                Image,
                "/oak/features/debug_image",
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            )

        self._last_summary = self.get_clock().now()
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._spin_pipeline, daemon=True)

    def start(self):
        self._thread.start()

    def request_shutdown(self):
        self._shutdown.set()

    def _spin_pipeline(self):
        with self._pipeline as pipeline:
            while rclpy.ok() and not self._shutdown.is_set():
                try:
                    frame = pipeline.next_frame()
                except Exception as exc:  # noqa: BLE001 - log and keep node alive
                    self.get_logger().error("Pipeline error: %s" % exc)
                    break
                if frame is None:
                    self.get_logger().info(
                        "Pipeline returned no frame (EOF); stopping publisher loop."
                    )
                    break
                gray, keypoints, codes, full_descs = frame
                stamp = self.get_clock().now().to_msg()
                records = self._pack_frame(keypoints, codes, full_descs)
                self._publish(stamp, records)
                if self.debug_pub is not None and self.debug_pub.get_subscription_count():
                    self._publish_debug(stamp, gray, records)
                self._log_summary(records)

    def _publish(self, stamp, records):
        msg = KeypointArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.compression = self.compression_const
        desc_len = 0
        for rec in records:
            if rec["desc"]:
                desc_len = len(rec["desc"])
                break
        msg.desc_bits = desc_len * 8
        for rec in records:
            kp = Keypoint()
            kp.x = rec["x"]
            kp.y = rec["y"]
            kp.track_id = rec["track_id"]
            kp.age = rec["age"]
            kp.desc = bytes(rec["desc"])
            kp.has_zorder = rec["has_zorder"]
            kp.zorder = rec["zorder"]
            msg.keypoints.append(kp)
        self.features_pub.publish(msg)

    def _publish_debug(self, stamp, gray, records):
        import cv2

        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for rec in records:
            color_seed = (rec["track_id"] * 2654435761) & 0xFFFFFF
            color = (
                color_seed & 0xFF,
                (color_seed >> 8) & 0xFF,
                (color_seed >> 16) & 0xFF,
            )
            cv2.circle(
                vis, (int(round(rec["x"])), int(round(rec["y"]))), 3, color, -1
            )
        vis_gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height, msg.width = vis_gray.shape[:2]
        msg.encoding = "mono8"
        msg.is_bigendian = 0
        msg.step = msg.width
        msg.data = vis_gray.tobytes()
        self.debug_pub.publish(msg)

    def _log_summary(self, records):
        now = self.get_clock().now()
        if (now - self._last_summary).nanoseconds >= 1_000_000_000:
            with_code = sum(1 for r in records if r["has_zorder"])
            self.get_logger().info(
                "keypoints=%d with_code=%d" % (len(records), with_code),
                throttle_duration_sec=1.0,
            )
            self._last_summary = now


def main(args=None):
    rclpy.init(args=args)
    node = FeatureNode()
    node.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.request_shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
