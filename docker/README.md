# Containerized ROS1/ROS2 runs

Docker images that build the message packages, run the feature node against a
host-attached OAK-1, and verify `/oak/features` end to end.

Prerequisites: Docker, and an OAK-1 attached to the host (USB). The descriptor
NNArchive must exist at `depthai_models/descriptor64.tar.xz` (see
[../tools/README.md](../tools/README.md)).

## ROS2 (Humble)

```bash
docker build -f docker/Dockerfile.ros2 -t oak-binary-desc:ros2 .
docker run --rm --privileged -v /dev/bus/usb:/dev/bus/usb --net=host oak-binary-desc:ros2
```

## ROS1 (Noetic)

```bash
docker build -f docker/Dockerfile.ros1 -t oak-binary-desc:ros1 .
docker run --rm --privileged -v /dev/bus/usb:/dev/bus/usb --net=host oak-binary-desc:ros1
```

Note: Noetic ships Python 3.8 but depthai needs >= 3.9, so the image installs
a standalone CPython 3.10 build (astral-sh/python-build-standalone) for the
node. `rospy`/`genpy` are not on PyPI, so the node imports the apt-installed
ROS Python packages (pure Python, built for 3.8) from
`/opt/ros/noetic/lib/python3/dist-packages` via PYTHONPATH, while
catkin/roscore stay on the system Python (generated message code is
version-agnostic).

Each run script starts the node, waits, then runs a verifier subscriber that
checks for 20 messages: keypoints present, `has_zorder` set, `desc` length
matches `desc_bits`, keypoints sorted by `zorder`, monotonic header stamps.
Exit code 0 = pass (`VERIFY ROS1: PASS` / `VERIFY ROS2: PASS`).

The scripts set `OAK_FEATURES_SYNTHETIC_FRAMES=1` by default: the node then
replaces the camera with a host-fed synthetic moving texture
(`core.pipeline.synthetic_texture_source`), so verification is independent of
what the (possibly covered) camera sees. Export
`OAK_FEATURES_SYNTHETIC_FRAMES=0` to verify against the live camera instead.

Without a device, drop the USB/privileged flags and use replay media instead:
set `OAK_FEATURES_CORE` as usual and pass `--media_path` equivalents via the
node params (`media_path` is a constructor arg of `FeaturePipeline`; wire it
through launch args if needed).
