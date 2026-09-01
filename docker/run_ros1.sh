#!/bin/bash
# End-to-end ROS1 proof: roscore + feature node (python3.10/depthai) + verifier.
# Note: no `set -u` — ROS setup.bash references unbound variables.
source /opt/ros/noetic/setup.bash
source /opt/ws/devel/setup.bash

roscore > /tmp/roscore.log 2>&1 &
ROSCORE_PID=$!
sleep 3

# The node needs the standalone Python 3.10 (depthai). rospy/genpy and the
# generated messages are pure Python: import the apt ROS packages (built for
# 3.8) and the catkin devel space via PYTHONPATH.
PYTHONPATH="/opt/ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages" \
OAK_FEATURES_CORE=/opt/binary-descriptor \
OAK_FEATURES_SYNTHETIC_FRAMES="${OAK_FEATURES_SYNTHETIC_FRAMES:-1}" \
    /opt/python310/bin/python3 /opt/binary-descriptor/ros1/oak_features/scripts/feature_node.py \
    > /tmp/node.log 2>&1 &
NODE_PID=$!
sleep 8

python3 /opt/binary-descriptor/docker/verify_ros1.py
RC=$?

echo "--- node log (tail) ---"
tail -5 /tmp/node.log
kill $NODE_PID $ROSCORE_PID 2>/dev/null
exit $RC
