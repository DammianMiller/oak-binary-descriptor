#!/usr/bin/env python3
"""ROS2 subscriber verifying /oak/features messages. Exits 0 on success."""

import sys

import rclpy
from rclpy.qos import qos_profile_sensor_data

from oak_features_msgs.msg import KeypointArray

N_MSGS = 20


def main():
    rclpy.init()
    node = rclpy.create_node("oak_features_verify")
    msgs = []
    node.create_subscription(
        KeypointArray, "/oak/features", msgs.append, qos_profile_sensor_data
    )

    node.get_logger().info(f"waiting for {N_MSGS} messages on /oak/features...")
    deadline = node.get_clock().now().nanoseconds + 45e9
    while len(msgs) < N_MSGS and node.get_clock().now().nanoseconds < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()

    if len(msgs) < N_MSGS:
        print(f"FAIL: received {len(msgs)}/{N_MSGS} messages")
        return 1

    counts = [len(m.keypoints) for m in msgs]
    with_code = [sum(1 for k in m.keypoints if k.has_zorder) for m in msgs]
    desc_ok = all(
        len(k.desc) == m.desc_bits // 8 for m in msgs for k in m.keypoints if k.desc
    )
    sorted_ok = all(
        [k.zorder for k in m.keypoints if k.has_zorder]
        == sorted(k.zorder for k in m.keypoints if k.has_zorder)
        for m in msgs
    )
    frame_ids = {m.header.frame_id for m in msgs}
    stamp_tuples = [(m.header.stamp.sec, m.header.stamp.nanosec) for m in msgs]
    stamps_advance = all(
        stamp_tuples[i + 1] >= stamp_tuples[i] for i in range(len(stamp_tuples) - 1)
    )
    comp = {m.compression for m in msgs}

    print(f"messages: {len(msgs)}")
    print(f"keypoints/msg: min={min(counts)} max={max(counts)}")
    print(f"keypoints with zorder: min={min(with_code)} max={max(with_code)}")
    print(f"desc sizes match desc_bits: {desc_ok}")
    print(f"sorted by zorder: {sorted_ok}")
    print(f"frame_id(s): {frame_ids}, compression: {comp}, stamps monotonic: {stamps_advance}")

    ok = (
        min(counts) > 0
        and min(with_code) > 0
        and desc_ok
        and sorted_ok
        and stamps_advance
    )
    print("VERIFY ROS2: PASS" if ok else "VERIFY ROS2: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
