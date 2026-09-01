#!/usr/bin/env python3
"""ROS1 node publishing keypoints with binary descriptors from an OAK pipeline.

Thin wrapper over the ROS-agnostic core in <example>/core/.
"""

import os
import sys
import time

import rospy

from oak_features_msgs.msg import Keypoint, KeypointArray
from sensor_msgs.msg import Image


COMPRESSION_MAP = {
    "subsample": KeypointArray.COMPRESSION_SUBSAMPLE,
    "xorfold": KeypointArray.COMPRESSION_XORFOLD,
    "lsh": KeypointArray.COMPRESSION_LSH,
    "itq": KeypointArray.COMPRESSION_ITQ,
}


def default_core_path():
    """Example root: two levels up from this script's package directory.

    Falls back to the OAK_FEATURES_CORE environment variable.
    """
    env = os.environ.get("OAK_FEATURES_CORE")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


class FeatureNode:
    def __init__(self):
        rospy.init_node("oak_features")

        core_path = rospy.get_param("~core_path", default_core_path())
        archive = rospy.get_param("~archive", "depthai_models/descriptor64_itq_strip32_slim.tar.xz")
        self.max_keypoints = rospy.get_param("~max_keypoints", 64)
        nn_nodes = rospy.get_param("~nn_nodes", 2)
        tracker_decimate = rospy.get_param("~tracker_decimate", 2)
        motion_estimator = rospy.get_param("~motion_estimator", "sw")
        tracking = rospy.get_param("~tracking", True)
        describe_gate = rospy.get_param("~describe_gate", False)
        change_thresh = rospy.get_param("~change_thresh", 6.0)
        refresh_every = rospy.get_param("~refresh_every", 30)
        fps_limit = rospy.get_param("~fps_limit", 0)
        self.compression = rospy.get_param("~compression", "itq").lower()
        self.publish_debug_image = rospy.get_param("~publish_debug_image", False)
        self.frame_id = rospy.get_param("~frame_id", "oak_camera")

        if self.compression not in COMPRESSION_MAP:
            rospy.logfatal(
                "Unknown ~compression '%s'; expected one of %s",
                self.compression,
                sorted(COMPRESSION_MAP.keys()),
            )
            sys.exit(1)

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
            max_keypoints=self.max_keypoints,
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

        self.features_pub = rospy.Publisher(
            "/oak/features", KeypointArray, queue_size=10
        )
        self.debug_pub = None
        if self.publish_debug_image:
            self.debug_pub = rospy.Publisher(
                "/oak/features/debug_image", Image, queue_size=1
            )

        self._last_summary = time.monotonic()

    def run(self):
        with self._pipeline as pipeline:
            while not rospy.is_shutdown():
                frame = pipeline.next_frame()
                if frame is None:
                    rospy.loginfo("Pipeline returned no frame (EOF); shutting down.")
                    break
                gray, keypoints, codes, full_descs = frame
                stamp = rospy.Time.now()
                records = self._pack_frame(keypoints, codes, full_descs)
                self._publish(stamp, records)
                if self.debug_pub is not None and self.debug_pub.get_num_connections():
                    self._publish_debug(stamp, gray, records)
                self._log_summary(records)

    def _publish(self, stamp, records):
        msg = KeypointArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.compression = COMPRESSION_MAP[self.compression]
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
            kp.desc = rec["desc"]
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
        now = time.monotonic()
        if now - self._last_summary >= 1.0:
            with_code = sum(1 for r in records if r["has_zorder"])
            rospy.loginfo(
                "keypoints=%d with_code=%d", len(records), with_code
            )
            self._last_summary = now


def main():
    node = FeatureNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
