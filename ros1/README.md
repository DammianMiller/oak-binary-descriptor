# ROS1 (Noetic) packages for binary-descriptor features

Thin rospy wrapper over the ROS-agnostic core in `../core/`. Publishes
`oak_features_msgs/KeypointArray` on `/oak/features` and, optionally, a mono8
debug image with drawn keypoints on `/oak/features/debug_image`.

## Prerequisites

- ROS Noetic installed.
- Python3 dependencies available to the Noetic Python3 interpreter:

  ```bash
  python3 -m pip install depthai numpy opencv-python-headless
  ```

- The NNArchive must be built first via `../tools/` (produces
  `depthai_models/descriptor64.tar.xz`).

## Build

```bash
mkdir -p ~/catkin_ws/src
ln -s "$(pwd)/oak_features_msgs" ~/catkin_ws/src/
ln -s "$(pwd)/oak_features" ~/catkin_ws/src/
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws
catkin build          # or: catkin_make
source devel/setup.bash
```

## Run

```bash
roslaunch oak_features feature_node.launch
# or
rosrun oak_features feature_node.py
```

Useful launch args (all mirror node params):

| arg | default | meaning |
| --- | --- | --- |
| `core_path` | auto-detected | Path to the example root containing `core/`; falls back to the `OAK_FEATURES_CORE` env var |
| `archive` | `depthai_models/descriptor64_itq_strip32_slim.tar.xz` | NNArchive path, relative to `core_path` unless absolute (trained + ITQ head) |
| `max_keypoints` | `64` | Per-frame descriptor budget (64 @ ~32 fps with the default archive; 684 = FeatureTracker hardware ceiling @ ~1.6 fps with the strip-64 archive, see README perf table) |
| `nn_nodes` | `2` | Parallel NN replicas sharing the NCE (2 = ~1.4x over 1; 3 adds nothing) |
| `fps_limit` | `0` | 0 = unlimited |
| `compression` | `itq` | One of `subsample`, `xorfold`, `lsh`, `itq` |
| `publish_debug_image` | `false` | Publish annotated mono8 image |
| `frame_id` | `oak_camera` | Header frame_id |
