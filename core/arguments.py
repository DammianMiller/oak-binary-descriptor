import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.description = (
        "Binary feature descriptor example for OAK-1 (RVC2). On-device FeatureTracker keypoints are cropped "
        "as 32x32 grayscale patches and described by a custom NNArchive running on the Myriad X NCE, producing "
        "a 512-bit raw descriptor and a 64-bit compressed code per keypoint. The archive must be built with "
        "the scripts in tools/ first (see tools/README.md)."
    )

    parser.add_argument(
        "-a",
        "--archive",
        help="Path to the descriptor NNArchive (.tar.xz) built with tools/. "
        "The slim strip-32 default targets ~30 Hz; use the strip-64 build "
        "(tools: --strip 64, no --slim) for maximum keypoints/frame.",
        default="depthai_models/descriptor64_itq_strip32_slim.tar.xz",
        type=str,
    )

    parser.add_argument(
        "--max_keypoints",
        help="Maximum number of keypoints cropped and described per frame. The "
        "FeatureTracker hardware ceiling is ~684 (2 memory slices x 342). "
        "Measured with the default slim strip-32 archive, 2 NN nodes, async: "
        "32 kps @ ~28 fps (640x400) up to 64 kps @ ~32 fps (320x200 tracker); "
        "with the strip-64 archive: 684 kps @ ~1.6 fps.",
        required=False,
        default=64,
        type=int,
    )

    parser.add_argument(
        "--describe_budget",
        help="Keypoints described per frame (default: = max_keypoints). Set "
        "below max_keypoints to track/show many features while describing "
        "only the oldest N each frame (codes align with the first N "
        "keypoints). The NCE aggregates ~3000 described keypoints/s, so "
        "describing everything caps fps at ~3000/max_keypoints.",
        required=False,
        default=None,
        type=int,
    )

    parser.add_argument(
        "--describe_gate",
        help="Change-gated describing: describe only new/changed/stale "
        "keypoints each frame (host-side fingerprint cache), capped at "
        "describe_budget per frame, most stale first. Static scenes drop to "
        "~zero strip traffic at full frame rate; codes come back parallel to "
        "keypoints (cached codes included, per-keypoint has_code/code_age).",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--change_thresh",
        help="Describe-gate fingerprint change threshold: mean abs diff of "
        "the 8x8 block-mean patch fingerprint (0-255 scale) above which a "
        "keypoint is re-described.",
        required=False,
        default=6.0,
        type=float,
    )

    parser.add_argument(
        "--refresh_every",
        help="Describe-gate staleness sweep: force re-describe of every "
        "keypoint at least this often (frames), bounding fingerprint false "
        "negatives; also the dead-entry eviction horizon.",
        required=False,
        default=30,
        type=int,
    )

    parser.add_argument(
        "--no_tracking",
        help="Detection-only: disable the FeatureTracker motion estimator "
        "(no track IDs; features arrive with id -1) and describe each "
        "frame's detections. Cross-frame re-identification is then purely "
        "descriptor-based (best-Hamming code matching).",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--tracker_size",
        help="WxH of the grayscale stream the FeatureTracker runs on "
        "(default: 640x400). 320x200 roughly doubles tracker throughput and "
        "is the measured >=30 Hz configuration at 64 keypoints/frame.",
        required=False,
        default="640x400",
        type=str,
    )

    parser.add_argument(
        "--tracker_decimate",
        help="Run the FeatureTracker on every Nth frame and propagate tracks "
        "host-side (per-track velocity, median fallback) on skipped frames. "
        "Descriptors are still computed every frame. N=2 roughly doubles "
        "tracker headroom at 640x400; N=1 disables (default: 2).",
        required=False,
        default=2,
        type=int,
    )

    parser.add_argument(
        "--motion_estimator",
        help="FeatureTracker motion estimation: 'sw' = optical flow (default; "
        "most stable descriptor codes). 'hw' = on-device block matching: no "
        "fps gain in the descriptor pipeline (NN-bound) and slightly less "
        "stable codes, but for extraction-only workloads it holds ~60 fps up "
        "to 1000 target features vs SW dropping to ~50 at 684.",
        required=False,
        default="sw",
        choices=["sw", "hw"],
        type=str,
    )

    parser.add_argument(
        "--no_autotune",
        help="Disable lighting autotune (host feedback loop that nudges the "
        "Harris threshold at runtime to hold the detected feature count near "
        "the target across lighting changes). Autotune is on by default.",
        action="store_true",
    )

    parser.add_argument(
        "--nn_nodes",
        help="Number of parallel NN replicas sharing the NCE (default: 2; "
        "measured ~1.4x over 1, 3 adds nothing).",
        required=False,
        default=2,
        type=int,
    )

    parser.add_argument(
        "-fps",
        "--fps_limit",
        help="FPS limit for the pipeline runtime.",
        required=False,
        default=None,
        type=int,
    )

    parser.add_argument(
        "-d",
        "--device",
        help="Optional name, DeviceID or IP of the camera to connect to.",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "-media",
        "--media_path",
        help="Path to the media file you aim to run the pipeline on. If not set, the pipeline will run on the camera input.",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "--compression",
        help="Informational only: 512->64 bit compression strategy. Must match the strategy the archive was compiled with in tools/.",
        required=False,
        default="itq",
        choices=["subsample", "xorfold", "lsh", "itq"],
        type=str,
    )

    parser.add_argument(
        "--no_full_desc",
        help="If passed, skips decoding the full 512-bit descriptors and only returns the 64-bit codes.",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--color",
        help="Opponent-color descriptor mode (Opponent-LATCH style: the NN "
        "consumes o1/o2/o3 opponent planes cropped from a second BGR stream). "
        "Default: auto-detect from the archive filename (contains 'color'). "
        "Use --color / --no_color to override.",
        required=False,
        default=None,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--detector",
        help="Corner detector used by the on-device FeatureTracker.",
        required=False,
        default="harris",
        choices=["harris", "shi_tomasi"],
        type=str,
    )

    args = parser.parse_args()

    return parser, args
