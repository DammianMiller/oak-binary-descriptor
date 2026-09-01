#!/bin/bash
# End-to-end ROS2 proof: feature node + verifier.
# Note: no `set -u` — ROS setup.bash references unbound variables.
source /opt/ros/humble/setup.bash
source /opt/ws/install/setup.bash

OAK_FEATURES_CORE=/opt/binary-descriptor \
OAK_FEATURES_SYNTHETIC_FRAMES="${OAK_FEATURES_SYNTHETIC_FRAMES:-1}" \
    ros2 run oak_features feature_node > /tmp/node.log 2>&1 &
NODE_PID=$!
sleep 8

python3 /opt/binary-descriptor/docker/verify_ros2.py
RC=$?

echo "--- node log (tail) ---"
tail -5 /tmp/node.log
kill $NODE_PID 2>/dev/null
exit $RC
