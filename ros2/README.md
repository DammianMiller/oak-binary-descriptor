# ROS2 (Humble) packages for binary-descriptor features

Thin rclpy wrapper over the ROS-agnostic core in `../core/`. Publishes
`oak_features_msgs/msg/KeypointArray` on `/oak/features` (sensor-data QoS) and,
optionally, a mono8 debug image with drawn keypoints on
`/oak/features/debug_image`.

## Prerequisites

- ROS2 Humble installed.
- Python3 dependencies available to the Humble Python3 interpreter:

  ```bash
  python3 -m pip install depthai numpy opencv-python-headless
  ```

- The NNArchive must be built first via `../tools/` (produces
  `depthai_models/descriptor64.tar.xz`).

## Build

```bash
mkdir -p ~/ros2_ws/src
ln -s "$(pwd)/oak_features_msgs" ~/ros2_ws/src/
ln -s "$(pwd)/oak_features" ~/ros2_ws/src/
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
colcon build --packages-select oak_features_msgs oak_features
source install/setup.bash
```

## Run

```bash
ros2 launch oak_features feature_node.launch.py
```

Parameters (same names as the ROS1 node):

| parameter | default | meaning |
| --- | --- | --- |
| `core_path` | auto-detected | Path to the example root containing `core/`; falls back to the `OAK_FEATURES_CORE` env var |
| `archive` | `depthai_models/descriptor64_itq_strip32_slim.tar.xz` | NNArchive path, relative to `core_path` unless absolute (trained + ITQ head) |
| `max_keypoints` | `64` | Per-frame descriptor budget (64 @ ~32 fps with the default archive; 684 = FeatureTracker hardware ceiling @ ~1.6 fps with the strip-64 archive, see README perf table) |
| `nn_nodes` | `2` | Parallel NN replicas sharing the NCE (2 = ~1.4x over 1; 3 adds nothing) |
| `fps_limit` | `0` | 0 = unlimited |
| `compression` | `itq` | One of `subsample`, `xorfold`, `lsh`, `itq` |
| `publish_debug_image` | `false` | Publish annotated mono8 image |
| `tracking` | `true` | `false` = detection-only: no track IDs (`track_id=-1`), every detected keypoint described, re-identification purely by `zorder` code (measured ~100% frame-to-frame re-id at ~100 keypoints/frame, color archive) |
| `frame_id` | `oak_camera` | Header frame_id |
