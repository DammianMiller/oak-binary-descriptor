from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    core_path = LaunchConfiguration("core_path")
    archive = LaunchConfiguration("archive")
    max_keypoints = LaunchConfiguration("max_keypoints")
    fps_limit = LaunchConfiguration("fps_limit")
    compression = LaunchConfiguration("compression")
    publish_debug_image = LaunchConfiguration("publish_debug_image")
    tracking = LaunchConfiguration("tracking")
    describe_gate = LaunchConfiguration("describe_gate")
    change_thresh = LaunchConfiguration("change_thresh")
    refresh_every = LaunchConfiguration("refresh_every")
    frame_id = LaunchConfiguration("frame_id")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "core_path",
                default_value="",
                description="Path to the example root containing core/ "
                "(empty = auto-detect / OAK_FEATURES_CORE).",
            ),
            DeclareLaunchArgument(
                "archive",
                default_value="depthai_models/descriptor64_itq_strip32_slim.tar.xz",
                description="NNArchive path, relative to core_path unless absolute.",
            ),
            DeclareLaunchArgument("max_keypoints", default_value="64"),
            DeclareLaunchArgument(
                "fps_limit", default_value="0", description="0 = unlimited."
            ),
            DeclareLaunchArgument(
                "compression",
                default_value="itq",
                description="One of: subsample, xorfold, lsh, itq.",
            ),
            DeclareLaunchArgument("publish_debug_image", default_value="false"),
            DeclareLaunchArgument(
                "tracking",
                default_value="true",
                description="false = detection-only (no track IDs; keypoints "
                "arrive with track_id -1 and are re-identified purely by "
                "descriptor code).",
            ),
            DeclareLaunchArgument(
                "describe_gate",
                default_value="false",
                description="true = describe only new/changed/stale keypoints "
                "(host fingerprint cache; codes stay valid via cache, "
                "has_zorder=False only for never-described keypoints).",
            ),
            DeclareLaunchArgument(
                "change_thresh",
                default_value="6.0",
                description="Gate fingerprint change threshold (0-255 scale).",
            ),
            DeclareLaunchArgument(
                "refresh_every",
                default_value="30",
                description="Gate staleness sweep period (frames).",
            ),
            DeclareLaunchArgument("frame_id", default_value="oak_camera"),
            Node(
                package="oak_features",
                executable="feature_node",
                name="oak_features",
                output="screen",
                parameters=[
                    {
                        "archive": archive,
                        "max_keypoints": max_keypoints,
                        "fps_limit": fps_limit,
                        "compression": compression,
                        "publish_debug_image": publish_debug_image,
                        "tracking": tracking,
                        "describe_gate": describe_gate,
                        "change_thresh": change_thresh,
                        "refresh_every": refresh_every,
                        "frame_id": frame_id,
                    },
                    # Pass core_path only when non-empty so the node's
                    # auto-detected default is preserved otherwise.
                    # (Empty string would be a valid explicit value anyway.)
                ],
                additional_env={"OAK_FEATURES_CORE": core_path},
            ),
        ]
    )
