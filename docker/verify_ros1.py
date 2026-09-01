#!/usr/bin/env python3
"""ROS1 subscriber verifying /oak/features messages (runs on system Python 3.8,
no depthai needed). Exits 0 on success, 1 on failure."""

import sys

import rospy
from oak_features_msgs.msg import KeypointArray

N_MSGS = 20


def main():
    rospy.init_node("oak_features_verify", anonymous=True)
    msgs = []

    def cb(msg):
        msgs.append(msg)

    sub = rospy.Subscriber("/oak/features", KeypointArray, cb, queue_size=50)
    rospy.loginfo("waiting for %d messages on /oak/features...", N_MSGS)

    deadline = rospy.Time.now() + rospy.Duration(45.0)
    while len(msgs) < N_MSGS and rospy.Time.now() < deadline and not rospy.is_shutdown():
        rospy.sleep(0.1)
    sub.unregister()

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
    stamps_advance = all(
        msgs[i + 1].header.stamp >= msgs[i].header.stamp for i in range(len(msgs) - 1)
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
    print("VERIFY ROS1: PASS" if ok else "VERIFY ROS1: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
