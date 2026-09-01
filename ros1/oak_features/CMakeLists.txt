cmake_minimum_required(VERSION 3.0.2)
project(oak_features)

find_package(catkin REQUIRED COMPONENTS
  rospy
  oak_features_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy oak_features_msgs sensor_msgs std_msgs
)

catkin_install_python(
  PROGRAMS scripts/feature_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
