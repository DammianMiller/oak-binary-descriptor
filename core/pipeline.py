"""DepthAI pipeline for on-device keypoint tracking with binary descriptors.

Wiring (RVC2 / OAK-1 target):

    camera / ReplayVideo -> GRAY8 frame -> FeatureTracker.inputImage
    FeatureTracker.outputFeatures        -> host queue (tracked keypoints)
    FeatureTracker.passthroughInputImage -> host queue (aligned display frame)
    host: numpy-crop 32x32 patches at keypoints, stack B patches into one
          32 x 32*B GRAY8 "strip" frame -> NeuralNetwork.input (input queue)
    NeuralNetwork.out -> host queue      (one NNData per strip, batched)

Why host-side stacking instead of on-device ImageManip crops: per-crop
ImageManip jobs plus per-crop inferences cap the pipeline at ~5 fps regardless
of keypoint budget. Stacking B patches into one frame turns N keypoints into
ceil(N/B) inferences; the strip blob is compiled with input shape
(1,1,32*B,32) (Myriad X collapses a true batch dim > 1, so the strip axis is
folded out inside the graph instead), and a GRAY8 strip frame of 32 x 32*B
pixels has exactly the contiguous byte layout the blob expects.

Overlap: strips are round-robined over ``nn_nodes`` parallel NN replicas
(measured: 2 nodes = ~1.4x aggregate throughput, 3 = NCE-saturated), and
next_frame() drains the PREVIOUS frame's strip outputs after submitting the
current frame's, so tracker/crop/NN stages run concurrently at one frame of
descriptor latency.

Opponent-color variant (archives with "color" in the name): the camera's
NV12 color stream at the tracker size doubles as the host frame carrier
(decoded to BGR by ImgFrame.getCvFrame); no grayscale frame is streamed to
the host at all. Two measured USB2 (~30 MB/s practical) constraints shape
this: BGR888i would need ~23 MB/s at 30 fps by itself, and adding a host
gray stream on top of NV12 caps 64 kps/frame color at ~20 fps (NV12-only
sustains ~30). Host crops are converted to (o1, o2, o3) opponent
planes (core/color.py) and packed channel-major into a 96*B-high GRAY8 strip
(the graph re-exposes the channel axis with a Reshape). The FeatureTracker
always stays on grayscale.

Correlation: one features message, one passthrough frame (arrival-order
aligned; FeatureTracker delays its features output by ~4 frames, so
passthrough pairing is by order, not sequence number), then per-strip NN
outputs matched by (node, arrival order). No cross-queue sequence matching.
"""

from __future__ import annotations

import datetime
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import depthai as dai
import numpy as np

from core.color import bgr_to_opponent_u8, stack_color_strip

from core.gate import DescribeGate
from core.packer import FULL_DESC_BYTES, KeypointRecord
from core.projection import pack_bits

FRAME_WIDTH = 640
FRAME_HEIGHT = 400
PATCH_SIZE = 32

_CODE_TENSOR = "code"  # uint8 (B,64) compressed descriptor bits
_DESC_TENSOR = "desc"  # float32 (B,512) raw descriptor, sign() is the 512-bit code

# FeatureTracker search target. The node clamps to ~342 features per memory
# slice and supports at most 2 slices (setHardwareResources(2, 2)), so ~684
# tracked features is the hardware ceiling on RVC2; the per-frame descriptor
# budget (max_keypoints) is enforced host-side after selection.
_TRACKER_TARGET_FEATURES = 684


class ThresholdAutotuner:
    """Host-side feedback loop holding the detected feature count steady
    across lighting changes.

    Why host-side: the firmware's own adaptive-threshold band
    (thresholds.min/max) re-detects iteratively within a frame, and a wide
    band collapsed throughput to ~1 fps (measured). This controller instead
    nudges thresholds.initialValue via the tracker's inputConfig queue at
    most once per interval — zero per-frame cost.

    Measured (640x400, 60 fps, detection-only, forced manual exposure):
    moderate dark (brightness ~25) recovered 212 -> 732 features/frame as the
    threshold stepped 200 -> 140; near-black (brightness ~6) cannot recover —
    corners are below the noise floor, no threshold helps. Counts above
    ~450-500 start costing frame rate, so callers should cap the target.
    """

    def __init__(self, config_queue, target: int, initial: float = 200.0,
                 low: float = 15.0, high: float = 2000.0, interval_s: float = 0.7,
                 min_fps: float | None = None):
        self._queue = config_queue
        self._target = target
        self._threshold = initial
        self._low = low
        self._high = high
        self._interval = interval_s
        self._min_fps = min_fps
        self._recent: list[int] = []
        self._recent_fps: list[float] = []
        self._last = 0.0
        self._fps_ok_seen = False  # a window at/above min_fps was observed
        self._last_dir = 0  # +1 last move raised the threshold, -1 lowered it
        self._guard_fires = 0  # consecutive guard fires without fps recovery

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def target(self) -> int:
        return self._target

    @target.setter
    def target(self, value: int) -> None:
        self._target = int(value)
        self._recent.clear()

    def update(self, n_features: int, fps: float | None = None) -> None:
        """Feed the latest detected (pre-cap) feature count (and measured fps).

        Converges the count onto the target: each update multiplies the
        threshold by sqrt(avg/target), clamped to [0.6, 1.6] per step, so
        the loop closes geometrically (typically 3-6 updates, ~2-4 s) without
        overshoot. A +/-5% deadband suppresses dithering on frame noise.
        Note the response is inverse: avg > target -> raise the threshold.

        If min_fps was set, the fps guard watches for the 60 fps cliff:
        below a threshold of ~200 the detector overruns the 16 ms frame
        budget and output collapses to ~46 fps / a truncated ~320-feature
        grid (measured). The guard fires only when fps was healthy before,
        then dropped right after the loop itself lowered the threshold —
        so an environmentally slow stream (dark scene, long auto-exposure)
        never triggers it. On firing it steps the threshold back up and
        ratchets the low bound above the offending value for this session.
        If three consecutive raises fail to restore fps, the fps loss is
        environmental and the guard disables itself for the session.
        """
        self._recent.append(n_features)
        if fps is not None:
            self._recent_fps.append(fps)
        now = time.time()
        if len(self._recent) < 20 or now - self._last < self._interval:
            return
        avg = sum(self._recent) / len(self._recent)
        avg_fps = (
            sum(self._recent_fps) / len(self._recent_fps) if self._recent_fps else None
        )
        self._recent.clear()
        self._recent_fps.clear()
        self._last = now
        if self._min_fps is not None and avg_fps is not None:
            if avg_fps >= self._min_fps:
                self._fps_ok_seen = True
                self._guard_fires = 0
            elif self._fps_ok_seen and self._last_dir < 0:
                if self._guard_fires >= 2:
                    # Raising the threshold is not restoring fps: the loss is
                    # environmental, not the detector budget. Stop blaming the
                    # threshold for the rest of this session.
                    self._min_fps = None
                else:
                    self._guard_fires += 1
                    new = min(self._high, self._threshold * 1.3)
                    self._low = min(self._high, max(self._low, self._threshold * 1.1))
                    if new != self._threshold:
                        self._threshold = new
                        self._last_dir = +1
                        self._send()
                return
        if self._target <= 0 or abs(avg - self._target) <= 0.05 * self._target:
            return
        if avg <= 0:
            factor = 0.5  # scene went dark: drop fast toward the floor
        else:
            factor = (avg / self._target) ** 0.5
            factor = min(1.6, max(0.6, factor))
        new = min(self._high, max(self._low, self._threshold * factor))
        if new == self._threshold:
            return  # clamped at a bound; nothing more to do
        self._last_dir = +1 if new > self._threshold else -1
        self._threshold = new
        self._send()

    def _send(self) -> None:
        cfg = dai.FeatureTrackerConfig()
        cd = dai.FeatureTrackerConfig.CornerDetector()
        cd.type = dai.FeatureTrackerConfig.CornerDetector.Type.HARRIS
        cd.thresholds.initialValue = self._threshold
        cfg.setCornerDetector(cd)
        cfg.setNumTargetFeatures(self._target)
        self._queue.send(cfg)


def synthetic_texture_source(seed: int = 7, size=(FRAME_WIDTH, FRAME_HEIGHT),
                             color: bool = False):
    """Return a callable producing moving GRAY8 frames host-side (or BGR
    (H, W, 3) frames with ``color=True``: the gray texture with mild
    per-channel offsets so the opponent chroma channels are non-degenerate).

    Used for deterministic, scene-independent pipeline verification (e.g. in
    containers): a slightly smoothed random texture viewed through a window
    on a slow Lissajous path, so the tracker sees hundreds of stable corners.
    """
    w, h = size
    rng = np.random.default_rng(seed)
    big = rng.integers(0, 255, (2 * h + 400, 2 * w + 560), dtype=np.uint8)
    big = (
        np.asarray(big, np.uint16)
        + np.roll(big, 1, 0)
        + np.roll(big, 1, 1)
        + np.roll(big, -1, 0)
        + np.roll(big, -1, 1)
    ) // 5
    big = big.astype(np.uint8)
    t = [0.0]

    if color:
        # Mild fixed chroma offsets so o1/o2 are non-degenerate.
        offsets = np.array([-10, 0, 10], dtype=np.int16)
        big_c = np.clip(np.asarray(big, np.int16)[..., None] + offsets, 0, 255).astype(np.uint8)
    else:
        big_c = None

    def source() -> np.ndarray:
        t[0] += 1.0
        cx = int(w // 2 + 280 + (w // 2 + 120) * np.sin(t[0] * 0.05))
        cy = int(h // 2 + 200 + (h // 2 + 50) * np.cos(t[0] * 0.037))
        src = big_c if color else big
        return src[cy : cy + h, cx : cx + w]

    return source


def _select_features(features, max_keypoints):
    """Cap at max_keypoints, FIFO by track ID (oldest tracks first)."""
    selected = sorted(features, key=lambda f: f.id)
    return selected[:max_keypoints]


def crop_patches(gray: np.ndarray, features) -> np.ndarray:
    """Crop 32x32 patches (clamped to frame bounds) around each feature.

    Returns an (n, 32, 32) uint8 array.
    """
    h, w = gray.shape[:2]
    half = PATCH_SIZE // 2
    patches = np.empty((len(features), PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    for i, f in enumerate(features):
        x0 = min(max(int(f.position.x) - half, 0), w - PATCH_SIZE)
        y0 = min(max(int(f.position.y) - half, 0), h - PATCH_SIZE)
        patches[i] = gray[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]
    return patches


def _nv12_to_bgr(msg, scale: int = 1) -> np.ndarray:
    """NV12 device frame -> (H, W, 3) uint8 BGR, upscaled x``scale``.

    The color stream is requested as NV12 to halve its USB bandwidth (USB2
    link; BGR888i at 30 fps starved the whole pipeline to ~14 fps), and in
    no-track mode at half resolution (scale=2) to quarter it — the measured
    way to free strip bandwidth at a full 30 fps cadence. Chroma sits at
    half resolution natively (quarter at scale=2), which the 3x3-window
    opponent transform tolerates; the intensity plane pays a mild upscale
    blur. ImgFrame.getCvFrame() already decodes NV12 to (H, W, 3) BGR
    (verified on depthai 3.8.0).
    """
    bgr = msg.getCvFrame()
    if scale > 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return bgr


def crop_opponent_planes(bgr: np.ndarray, features) -> np.ndarray:
    """Crop 32x32 color patches and convert to opponent planes.

    ``bgr`` is an (H, W, 3) uint8 frame. Returns (n, 3, 32, 32) uint8
    (o1, o2, o3) planes with zero at 127.5, ready for stack_color_strip.
    """
    h, w = bgr.shape[:2]
    half = PATCH_SIZE // 2
    patches = np.empty((len(features), PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    for i, f in enumerate(features):
        x0 = min(max(int(f.position.x) - half, 0), w - PATCH_SIZE)
        y0 = min(max(int(f.position.y) - half, 0), h - PATCH_SIZE)
        patches[i] = bgr[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]
    return bgr_to_opponent_u8(patches)


def bgr_to_gray_u8(bgr: np.ndarray) -> np.ndarray:
    """BT.601 luma for feeding the (grayscale) FeatureTracker from a color
    host frame source."""
    x = np.asarray(bgr, dtype=np.float32)
    return np.round(x[..., 0] * 0.114 + x[..., 1] * 0.587 + x[..., 2] * 0.299).astype(np.uint8)


# Script node text for tracker decimation: forwards every Nth frame to the
# FeatureTracker. NOTE: frame.getData() on full-size frames hard-crashes the
# device, so the decimator must pass the frame through untouched (measured:
# plain passthrough scripts sustain ~29 msg/s at 640x400, enough to read at
# 30 fps and forward at 15 fps).
_DECIMATOR_SCRIPT = """
i = 0
while True:
    f = node.io["in"].get()
    i += 1
    if i % DECIMATE == 0:
        node.io["out"].send(f)
"""


def build_pipeline(
    pipeline,
    archive_path,
    max_keypoints,
    fps_limit=None,
    media_path=None,
    platform="RVC2",
    detector="harris",
    host_input=False,
    nn_nodes=1,
    tracker_size=(FRAME_WIDTH, FRAME_HEIGHT),
    tracker_decimate=1,
    motion_estimator="sw",
    color=False,
    tracking=True,
    color_scale=1,
):
    """Create and wire the device-side nodes of the pipeline.

    Returns a ``SimpleNamespace(source, to_gray, gray_out, feature_tracker, nns)``.
    The NN inputs are intentionally left unlinked; the host feeds stacked strip
    frames through ``nn.input.createInputQueue()``. With ``host_input=True`` no
    camera/relay node is created either; the host pushes GRAY8 frames through
    ``feature_tracker.inputImage.createInputQueue()`` (used for deterministic
    benchmarks when the camera view is unsuitable). ``nn_nodes`` creates that
    many parallel NN replicas sharing the single NCE; measured on OAK-1, 2
    nodes give ~1.4x aggregate descriptor throughput (DMA/setup overlap),
    3 nodes add nothing (NCE saturated).

    With ``color=True`` (opponent-color descriptor archive), an NV12 color
    stream at the tracker size is exposed as ``color_out``; it doubles as the
    host frame carrier (no grayscale frame is streamed to the host in color
    mode) and the host crops opponent planes from it. The tracker itself
    stays grayscale.

    With ``tracking=False`` the FeatureTracker's motion estimator is
    disabled (detection-only, ids are -1); tracker_decimate is meaningless
    there and the caller forces it to 1.
    """
    color_out = None
    if host_input:
        source = None
        to_gray = None
        gray_out = None
    elif media_path:
        source = pipeline.create(dai.node.ReplayVideo)
        source.setReplayVideoFile(Path(media_path))
        if platform == "RVC2":
            source.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        elif platform == "RVC4":
            source.setOutFrameType(dai.ImgFrame.Type.BGR888i)
        else:
            raise ValueError(f"ReplayVideo node not supported for {platform}.")
        source.setLoop(True)
        if fps_limit:
            source.setFps(float(fps_limit))
        to_gray = pipeline.create(dai.node.ImageManip)
        to_gray.initialConfig.setOutputSize(*tracker_size)
        to_gray.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
        to_gray.setMaxOutputFrameSize(FRAME_WIDTH * FRAME_HEIGHT)
        source.out.link(to_gray.inputImage)
        gray_out = to_gray.out
        if color:
            to_color = pipeline.create(dai.node.ImageManip)
            to_color.initialConfig.setOutputSize(*tracker_size)
            # NV12: half the bytes of BGR888i. On a USB2 (HIGH speed) link a
            # 640x400x3 BGR stream alone is ~23 MB/s at 30 fps and the whole
            # pipeline throttles to ~14 fps; NV12 keeps the total under the
            # ~30 MB/s practical USB2 ceiling. Chroma is half-res, which the
            # opponent transform tolerates (LATCH windows are 3x3 anyway).
            to_color.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
            to_color.setMaxOutputFrameSize(FRAME_WIDTH * FRAME_HEIGHT * 3)
            source.out.link(to_color.inputImage)
            color_out = to_color.out
    else:
        to_gray = None
        source = pipeline.create(dai.node.Camera).build()
        gray_out = source.requestOutput(
            tracker_size,
            type=dai.ImgFrame.Type.GRAY8,
            fps=fps_limit,
        )
        if color:
            # NV12 instead of BGR888i: halves the color stream's USB
            # bandwidth (see the media branch above). color_scale=2 requests
            # it at half resolution (96 KB vs 384 KB/frame) and the host
            # upscales x2 before cropping — the measured way to free USB2
            # budget for descriptor strips at a full 30 fps cadence. (Do NOT
            # use requestOutput(fps=...) to thin the stream: it throttles the
            # WHOLE sensor, measured gray 30 -> 16 fps.)
            cs = max(1, int(color_scale))
            color_out = source.requestOutput(
                (tracker_size[0] // cs, tracker_size[1] // cs),
                type=dai.ImgFrame.Type.NV12,
                fps=fps_limit,
            )

    feature_tracker = pipeline.create(dai.node.FeatureTracker)
    corner_type = (
        dai.FeatureTrackerConfig.CornerDetector.Type.HARRIS
        if detector == "harris"
        else dai.FeatureTrackerConfig.CornerDetector.Type.SHI_THOMASI
    )
    feature_tracker.initialConfig.setCornerDetector(corner_type)
    if not tracking:
        # Detection-only: no optical flow, no track IDs (features arrive
        # with id -1). Re-identification across frames is then purely the
        # descriptor's job (best-Hamming code matching). This is the cheap
        # detector configuration the extraction-only fast path uses.
        feature_tracker.initialConfig.setMotionEstimator(False)
    elif motion_estimator == "hw":
        # Hardware block-matching motion estimation. Extraction-only it is a
        # big win (flat ~60 fps up to 1000 target features vs SW dropping to
        # ~50 at 684), but in the descriptor pipeline throughput is NN-bound
        # either way and SW optical flow gives measurably more stable codes
        # (median 0 / p90 2 vs median 1 / p90 3-10 bits), so "sw" is the
        # default here.
        feature_tracker.initialConfig.setHwMotionEstimation()
    elif motion_estimator != "sw":
        raise ValueError(f"unknown motion_estimator {motion_estimator!r}; expected 'sw' or 'hw'")
    # Track only as many features as the descriptor budget can use: tracker
    # throughput scales inversely with feature count, so a small budget
    # should not pay for tracking 684.
    feature_tracker.initialConfig.setNumTargetFeatures(
        min(max_keypoints, _TRACKER_TARGET_FEATURES)
    )
    # 2 SHAVEs + 2 memory slices: optical flow tracking at the maximum
    # supported feature count (~342 per slice).
    feature_tracker.setHardwareResources(2, 2)
    decimator = None
    if gray_out is not None:
        if tracker_decimate > 1:
            # Feed the tracker every Nth frame; the host propagates tracks on
            # the skipped frames. The raw stream still goes to the host for
            # per-frame crops/display.
            decimator = pipeline.create(dai.node.Script)
            decimator.setScript(_DECIMATOR_SCRIPT.replace("DECIMATE", str(tracker_decimate)))
            gray_out.link(decimator.inputs["in"])
            decimator.outputs["out"].link(feature_tracker.inputImage)
        else:
            gray_out.link(feature_tracker.inputImage)

    nns = []
    for _ in range(nn_nodes):
        nn = pipeline.create(dai.node.NeuralNetwork)
        nn.setNNArchive(dai.NNArchive(str(archive_path)))
        nn.setNumInferenceThreads(1)  # strip order must match send order
        nns.append(nn)

    return SimpleNamespace(
        source=source,
        nns=nns,
        to_gray=to_gray,
        gray_out=gray_out,
        color_out=color_out,
        feature_tracker=feature_tracker,
        decimator=decimator,
        nn=nn,
    )


class FeaturePipeline:
    """Context-managed wrapper running the binary-descriptor pipeline.

    Usage:
        with FeaturePipeline("depthai_models/descriptor64.tar.xz") as fp:
            frame, keypoints, codes, full_descs = fp.next_frame()
    """

    def __init__(
        self,
        archive_path,
        max_keypoints=684,
        fps_limit=None,
        device=None,
        media_path=None,
        publish_full_desc=True,
        describe_budget=None,
        detector="harris",
        frame_source=None,
        nn_nodes=2,
        tracker_size=None,
        tracker_decimate=2,
        motion_estimator="sw",
        autotune=True,
        color=None,
        tracking=True,
        color_scale=None,
        describe_gate=False,
        change_thresh=6.0,
        refresh_every=30,
    ):
        archive = Path(archive_path)
        if not archive.is_file():
            raise FileNotFoundError(
                f"NNArchive not found: {archive}. Build the descriptor model "
                "first with the scripts in tools/ (see tools/README.md)."
            )
        # Opponent-color variant: the strip folds 3 planes per patch into the
        # height axis (input (1,1,96*B,32)). Auto-detected from the archive
        # filename; override with color=True/False.
        self._color = ("color" in archive.name) if color is None else bool(color)
        # Patches per inference: either a true batch dim (B,1,32,32) or a
        # strip input (1,1,32*B,32). Myriad X collapses real batch > 1, so the
        # strip form is what the tools actually produce for B > 1.
        cfg = dai.NNArchive(str(archive)).getConfigV1().model
        shape = cfg.inputs[0].shape  # [N, 1, H, W]
        per_patch = PATCH_SIZE * (3 if self._color else 1)
        self._batch = max(int(shape[0]), int(shape[2]) // per_patch)
        # Code-only archives (make_projection.py --code_only) don't stream the
        # float desc tensor at all; skip decoding it instead of failing.
        output_names = {o.name for o in cfg.outputs}
        if _DESC_TENSOR not in output_names:
            publish_full_desc = False

        self._archive_path = archive
        self._max_keypoints = max_keypoints
        # Descriptor budget per frame, decoupled from the tracked/displayed
        # count: the NCE aggregates ~3000 described keypoints/s (color blob:
        # 96 strips/s), so describing every tracked keypoint caps the frame
        # rate at ~3000/max_keypoints fps. With a budget, the tracker and the
        # output show up to max_keypoints tracks while only the oldest
        # describe_budget (FIFO by track ID, a prefix of the keypoints list)
        # get codes each frame. None = describe everything (legacy behavior).
        if describe_budget is None:
            self._describe_budget = max_keypoints
        else:
            self._describe_budget = max(1, min(int(describe_budget), max_keypoints))
        # Change-gated describing ("describe only what moved"): a host-side
        # cache of code+desc+patch-fingerprint per keypoint identity (track
        # ID, or spatial nearest-neighbor in no-track mode) selects only new/
        # changed/stale keypoints for strip extraction each frame, capped at
        # describe_budget/frame (most stale first = round-robin fairness).
        # Static scenes drop to ~zero strip traffic at full frame rate; a
        # fully-moving scene degrades gracefully to describe-everything. The
        # output contract changes: codes/full_descs come back PARALLEL to
        # keypoints (cached codes included), with KeypointRecord.has_code
        # False for never-described keypoints and code_age = frames since the
        # code was actually described.
        self._gate = (
            DescribeGate(self._describe_budget, change_thresh, refresh_every)
            if describe_gate
            else None
        )
        self._fps_limit = fps_limit
        self._device_id = device
        self._media_path = media_path
        self._publish_full_desc = publish_full_desc
        self._detector = detector
        # Optional callable returning the next host-fed GRAY8 frame
        # (FRAME_HEIGHT x FRAME_WIDTH uint8); replaces the camera entirely.
        self._frame_source = frame_source
        self._nn_nodes = nn_nodes
        # Tracker/crop frame size. 320x200 roughly doubles FeatureTracker
        # throughput (fewer pixels, fewer corners), at the cost of keypoint
        # coordinate resolution and 32x32 patches covering 2x2 the area.
        self._tracker_size = tracker_size or (FRAME_WIDTH, FRAME_HEIGHT)
        # Tracker decimation: run the FeatureTracker on every Nth frame only
        # (its cost scales with pixels x features and is the 30 Hz bottleneck
        # at 640x400) and propagate tracks host-side on the skipped frames
        # (per-track velocity, median-velocity fallback). Descriptors are
        # still computed EVERY frame from fresh crops at the propagated
        # positions, so keypoint output rate stays at the frame rate.
        self._decimate = max(1, int(tracker_decimate))
        self._motion_estimator = motion_estimator
        self._autotune = autotune
        self._autotuner = None
        self._tracking = bool(tracking)
        if not self._tracking:
            # Detection-only: no track IDs exist, so there is nothing to
            # propagate on skipped frames; the detector is cheap enough
            # (motion estimator off) to run every frame.
            self._decimate = 1
        # Color stream downscale factor (camera color mode). Default: 2 in
        # no-track color mode (96 KB vs 384 KB/frame frees USB2 budget for
        # descriptor strips at a full 30 fps cadence; crops come from the
        # x2-upscaled frame, so patch content is slightly blurred), else 1.
        if color_scale is None:
            color_scale = 2 if (not self._tracking and self._color) else 1
        self._color_scale = max(1, int(color_scale))
        if self._decimate > 1 and frame_source is not None:
            # Host-fed frames + decimation deadlocks the XLink input path
            # (cam queue and NN strip queues contend when the loop free-runs
            # without blocking device reads). Camera mode uses an on-device
            # Script decimator instead and is the supported decimation path.
            print(
                "Warning: tracker_decimate is ignored with a host frame_source "
                "(host-fed + decimate deadlocks); using decimate=1."
            )
            self._decimate = 1

        # Host-side track state for propagation (decimate > 1).
        self._tracks: dict[int, tuple[float, float, int]] = {}  # id -> (x, y, age)
        self._vel: dict[int, tuple[float, float]] = {}  # id -> per-frame velocity
        self._frames_since_update = 0
        self._update_spacing = float(self._decimate)
        self._feed_idx = 0

        self._device = None
        self._pipeline = None
        self.platform = None
        self.nodes = None
        self._q_features = None
        self._q_frames = None
        self._q_nns = None
        self._q_ins = None
        # In-flight frame job (one-frame descriptor latency for overlap).
        self._pending = None
        # Liveness tracking (see next_frame): a session that goes silent for
        # >8 s is dead (firmware crash/hang). _dead also guards __exit__:
        # depthai's close() RPCs a dead device and can SEGFAULT the host
        # (measured: PipelineImpl::stop -> DeviceBase::closeImpl ->
        # hasCrashDump -> nanorpc on a dead XLink connection), so a dead
        # device is never closed — the OS reclaims the USB session at
        # process exit.
        self._started_ts = None
        self._last_output_ts = None
        self._dead = False
        self._nn_stall_streak = 0

    @property
    def pipeline(self):
        return self._pipeline

    def __enter__(self):
        self._device = (
            dai.Device(dai.DeviceInfo(self._device_id)) if self._device_id else dai.Device()
        )
        try:
            if self._device.hasCrashDump():
                print(
                    "Warning: device carries a crash dump from a previous "
                    "session (firmware watchdog reset, errorId 9001). Dumps: "
                    "~/.cache/depthai/crashdumps/. If sessions keep dying at "
                    "teardown/reopen, power-cycle the device."
                )
        except Exception:
            pass
        self.platform = self._device.getPlatformAsString()
        self._pipeline = dai.Pipeline(self._device)
        self.nodes = build_pipeline(
            self._pipeline,
            self._archive_path,
            self._max_keypoints,
            fps_limit=self._fps_limit,
            media_path=self._media_path,
            platform=self.platform,
            detector=self._detector,
            host_input=self._frame_source is not None,
            nn_nodes=self._nn_nodes,
            tracker_size=self._tracker_size,
            tracker_decimate=self._decimate,
            motion_estimator=self._motion_estimator,
            color=self._color,
            tracking=self._tracking,
            color_scale=self._color_scale,
        )
        self._q_features = self.nodes.feature_tracker.outputFeatures.createOutputQueue(
            maxSize=16, blocking=False
        )
        color_stream = self._color and self.nodes.color_out is not None
        if color_stream:
            # Color mode: the NV12 color stream IS the frame carrier (crops
            # and display), and no grayscale frame is streamed to the host at
            # all — the tracker keeps its on-device gray link, but a host
            # gray queue would cost ~8 MB/s on the USB2 link and cap 64
            # kps/frame color at ~20 fps instead of ~30. Features pair with
            # the raw color stream by arrival order, exactly like the
            # decimated mode's raw-gray pairing.
            self._q_frames = self.nodes.color_out.createOutputQueue(maxSize=16, blocking=False)
        elif self._decimate > 1:
            # Crops/display come from the raw stream (camera mode) or the host
            # source frames; the tracker passthrough only carries tracker-fed
            # frames now. It must NOT be queued unless it is drained every
            # loop: an unread queue backpressures the tracker, which then
            # stops consuming its input and wedges the pipeline (observed).
            if self._frame_source is None:
                self._q_frames = self.nodes.gray_out.createOutputQueue(maxSize=8, blocking=False)
            else:
                self._q_frames = None
        else:
            self._q_frames = self.nodes.feature_tracker.passthroughInputImage.createOutputQueue(
                maxSize=16, blocking=False
            )
        self._q_nns = [nn.out.createOutputQueue(maxSize=64, blocking=False) for nn in self.nodes.nns]
        self._q_ins = [
            nn.input.createInputQueue(maxSize=8, blocking=True) for nn in self.nodes.nns
        ]
        self._q_cam = (
            self.nodes.feature_tracker.inputImage.createInputQueue(maxSize=4, blocking=True)
            if self._frame_source is not None
            else None
        )
        if self._autotune:
            # Lighting feedback: hold the detected feature count near the
            # tracker target by nudging the Harris threshold at runtime.
            q_cfg = self.nodes.feature_tracker.inputConfig.createInputQueue(
                maxSize=4, blocking=False
            )
            self._autotuner = ThresholdAutotuner(
                q_cfg, min(self._max_keypoints, _TRACKER_TARGET_FEATURES)
            )
        self._pipeline.start()
        self._started_ts = time.time()
        self._last_output_ts = None
        return self

    # Teardown/reopen contract, all measured on-device (bisected):
    # 1. Close promptly: immediate stop+close reopens cleanly. Draining
    #    queues or settling between stop and close leaves the NEXT session's
    #    queues permanently empty (0 frames, no crash).
    # 2. Do NOT idle between close and reopen: a ~2 s gap reproducibly
    #    yields dead sessions (opens fine, every queue stays empty until the
    #    process fully exits) — the firmware does not restart cleanly from
    #    its between-session idle state for this pipeline shape.
    # 3. Rarely (~1 in several heavy transitions) teardown hangs the firmware
    #    outright: watchdog errorId 9001, 1.5 s stall, auto-reset (dumps in
    #    ~/.cache/depthai/crashdumps/). Not host-preventable; the wedge
    #    detector in next_frame turns the fallout into a loud error.
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._dead:
            # Never close a device we know is dead: closeImpl RPCs it and can
            # segfault the host (measured). The USB session is reclaimed at
            # process exit; the device itself reboots via watchdog.
            print("Device session was already dead; skipping close.")
            self._pending = None
            self._pipeline = None
            self._device = None
            return
        try:
            if self._pipeline is not None and self._pipeline.isRunning():
                self._pipeline.stop()
        except Exception:
            pass
        try:
            if self._device is not None:
                self._device.close()
        except Exception:
            pass
        self._pending = None
        self._pipeline = None
        self._device = None

    @staticmethod
    def _get(queue, timeout):
        """Blocking get with an optional timeout in seconds; None on timeout."""
        if timeout is None:
            return queue.get()
        return queue.get(datetime.timedelta(seconds=timeout))

    def _get_live(self, queue, timeout):
        """_get that converts a lost XLink connection into a loud error.

        depthai logs "Communication exception ... X_LINK_ERROR" and raises
        QueueException when the device vanishes mid-session (firmware crash/
        hang). Mark the session dead so __exit__ skips the segfault-prone
        close path, and raise immediately instead of timing out silently.
        """
        try:
            return self._get(queue, timeout)
        except dai.MessageQueue.QueueException as e:
            self._dead = True
            raise RuntimeError(
                "Lost the XLink connection to the device mid-session "
                "(firmware crash/hang; check ~/.cache/depthai/crashdumps/). "
                "The device reboots itself via watchdog; exit this process "
                "fully before reopening (closing a dead device can segfault "
                "the depthai host library)."
            ) from e

    def _send_strips(self, patches: np.ndarray) -> list[int]:
        """Send ceil(n/B) stacked strips round-robin over the NN nodes.

        ``patches`` is (n,32,32) grayscale or (n,3,32,32) opponent planes in
        color mode (packed channel-major into a 96*B-high GRAY8 strip).
        Returns the node index each strip was sent to (strip order is
        preserved per node, so outputs can be matched by (node, order)).
        """
        n = len(patches)
        if n == 0:
            return []
        b = self._batch
        n_strips = (n + b - 1) // b
        strip_nodes = []
        for s in range(n_strips):
            chunk = patches[s * b : (s + 1) * b]
            if self._color:
                strip = stack_color_strip(chunk, b)  # zero-pads internally
            else:
                if len(chunk) < b:  # zero-pad the final strip
                    pad = np.zeros((b - len(chunk), PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                    chunk = np.concatenate([chunk, pad], axis=0)
                strip = chunk.reshape(b * PATCH_SIZE, PATCH_SIZE)  # rows = patches
            frame = dai.ImgFrame()
            frame.setCvFrame(strip, dai.ImgFrame.Type.GRAY8)
            node = s % len(self._q_ins)
            self._q_ins[node].send(frame)
            strip_nodes.append(node)
        return strip_nodes

    def _drain_job(self, job, timeout):
        """Read a frame's strip outputs and correlate them to its keypoints.

        Legacy mode: codes/full_descs align with the FIRST n_described
        entries of the returned keypoints (a prefix; tracks beyond the
        describe budget have no code this frame). Gated mode: the gate merges
        fresh codes with its cache and returns arrays PARALLEL to keypoints
        (has_code/code_age set per record).
        """
        n = job["n_described"]
        strip_nodes = job["strip_nodes"]
        codes = np.zeros((0,), dtype=np.uint64)
        full_descs = (
            np.zeros((0, FULL_DESC_BYTES), dtype=np.uint8)
            if self._publish_full_desc
            else None
        )
        if n and strip_nodes:
            code_rows = [None] * len(strip_nodes)
            desc_rows = [None] * len(strip_nodes) if self._publish_full_desc else None
            by_node: dict[int, list[int]] = {}
            for s, nd in enumerate(strip_nodes):
                by_node.setdefault(nd, []).append(s)
            missing = False
            for nd, strip_ids in by_node.items():
                for s in strip_ids:
                    out = self._get_live(self._q_nns[nd], timeout)
                    if out is None:
                        # Keep the keypoints (they are valid tracking output);
                        # only the codes are lost. But repeated misses mean
                        # the NN nodes have wedged (observed live: camera and
                        # tracker streams keep flowing while the NCE stops
                        # producing; the 8 s silence detector can't see it
                        # because frames still arrive). Fail loudly after 3
                        # consecutive stalled frames instead of warning
                        # forever.
                        self._nn_stall_streak += 1
                        if self._nn_stall_streak >= 3:
                            self._dead = True
                            raise RuntimeError(
                                "NN nodes stopped producing outputs while "
                                "camera/tracker streams kept flowing (NCE "
                                "wedge). The device needs a session reset: "
                                "exit this process fully and restart it."
                            )
                        print("Warning: NN output missing for a strip; dropping frame's descriptors.")
                        missing = True
                        break
                    code_rows[s] = np.asarray(out.getTensor(_CODE_TENSOR)).reshape(self._batch, 64)
                    if self._publish_full_desc:
                        desc_rows[s] = np.asarray(out.getTensor(_DESC_TENSOR)).reshape(
                            self._batch, 512
                        )
                if missing:
                    break
            if not missing:
                self._nn_stall_streak = 0
                codes_all = np.concatenate(code_rows, axis=0)[:n]
                codes = np.atleast_1d(pack_bits((codes_all > 0).astype(np.uint8)))
                if self._publish_full_desc:
                    raw = np.concatenate(desc_rows, axis=0)[:n]
                    full_descs = np.stack(
                        [np.packbits((row > 0).astype(np.uint8)) for row in raw]
                    ).astype(np.uint8)

        if job.get("gate_plan") is not None:
            codes, full_descs = self._gate.apply(
                job["gate_plan"], job["features"], job["keypoints"],
                codes, full_descs, self._publish_full_desc,
            )
        return job["frame"], job["keypoints"], codes, full_descs

    def _update_tracks(self, kp_msg):
        """Adopt a tracker observation and estimate per-frame velocities."""
        spacing = max(1, self._frames_since_update + 1)
        new_tracks = {}
        for f in kp_msg.trackedFeatures:
            new_tracks[int(f.id)] = (float(f.position.x), float(f.position.y), int(f.age))
        vels = {}
        dxs, dys = [], []
        for tid, (x, y, _age) in new_tracks.items():
            if tid in self._tracks:
                ox, oy, _ = self._tracks[tid]
                vx, vy = (x - ox) / spacing, (y - oy) / spacing
                vels[tid] = (vx, vy)
                dxs.append(vx)
                dys.append(vy)
        # New/reacquired tracks get the global median velocity (robust to a
        # few bad correspondences) until they have their own estimate.
        mx = float(np.median(dxs)) if dxs else 0.0
        my = float(np.median(dys)) if dys else 0.0
        self._vel = {tid: vels.get(tid, (mx, my)) for tid in new_tracks}
        self._tracks = new_tracks
        self._frames_since_update = 0

    def _propagated_features(self):
        """Current-frame feature list: last observation + velocity * lag."""
        feats = []
        for tid, (x, y, age) in self._tracks.items():
            vx, vy = self._vel.get(tid, (0.0, 0.0))
            feats.append(
                SimpleNamespace(
                    id=tid,
                    position=SimpleNamespace(
                        x=x + vx * self._frames_since_update,
                        y=y + vy * self._frames_since_update,
                    ),
                    age=age + self._frames_since_update,
                )
            )
        return feats

    def next_frame(self, timeout=None):
        """Get the next processed frame.

        Returns:
            ``(frame, keypoints, codes, full_descs)`` where ``frame`` is a
            grayscale ndarray (BGR in color mode), ``keypoints`` a list of
            ``core.packer.KeypointRecord``, ``codes`` an (m,) uint64 array of
            compressed descriptor codes, and ``full_descs`` an
            (m, FULL_DESC_BYTES) uint8 array (or None when
            ``publish_full_desc`` is False). Returns None on timeout/EOF.

            With ``describe_budget`` < ``max_keypoints``, m can be smaller
            than len(keypoints): codes/descs align with the FIRST m keypoints
            (oldest tracks first); the rest are tracking-only that frame.

            With ``describe_gate`` enabled, codes/descs are PARALLEL to
            keypoints (m == len(keypoints)) and include cached codes for
            unchanged keypoints; KeypointRecord.has_code is False for
            never-described keypoints and code_age counts frames since each
            code was actually (re-)described.

            Note the one-frame overlap: the returned frame/keypoints/codes
            belong to the previously submitted frame (constant one-frame
            descriptor latency; the very first call is synchronous).
        """
        # Silence detector: 8 s with no output means the session is dead —
        # either a pipeline that never produced (firmware wedged after a
        # watchdog crash that did not reboot: sessions open fine but every
        # queue stays empty) or a mid-session hang. Fail loudly instead of
        # returning None forever.
        ref = self._last_output_ts or self._started_ts
        if ref is not None and time.time() - ref > 8.0:
            self._dead = True
            raise RuntimeError(
                "Device produced no frames for 8+ s — the firmware is "
                "wedged or crashed (watchdog errorId 9001; dumps in "
                "~/.cache/depthai/crashdumps/). Exit ALL processes holding "
                "the device (a full USB session drop is what recovers it) "
                "and retry; power-cycle if it persists."
            )
        decimated = self._decimate > 1
        color_src = None
        if self._frame_source is not None:
            frame = self._frame_source()
            if self._color:
                color_src = frame  # host source supplies BGR (H, W, 3)
                feed = bgr_to_gray_u8(frame)
            else:
                feed = frame
            if not decimated or self._feed_idx % self._decimate == 0:
                img = dai.ImgFrame()
                img.setCvFrame(feed, dai.ImgFrame.Type.GRAY8)
                self._q_cam.send(img)
            self._feed_idx += 1
        else:
            frame_msg = self._get_live(self._q_frames, timeout)
            if frame_msg is None:
                return None
            if self._color:
                # The color stream is the frame carrier (NV12 -> BGR,
                # upscaled if the stream runs at reduced resolution).
                color_src = _nv12_to_bgr(frame_msg, self._color_scale)
                frame = color_src
            else:
                frame = frame_msg.getCvFrame()

        if decimated:
            # Tracker runs on every Nth frame; propagate tracks in between so
            # descriptors are still computed at the full frame rate.
            try:
                kp_msg = self._q_features.tryGet()
            except dai.MessageQueue.QueueException:
                # Same dead-device path as _get_live.
                self._dead = True
                raise RuntimeError(
                    "Lost the XLink connection to the device mid-session "
                    "(firmware crash/hang; check ~/.cache/depthai/crashdumps/)."
                ) from None
            if kp_msg is not None:
                if self._autotuner is not None:
                    self._autotuner.update(len(kp_msg.trackedFeatures))
                self._update_tracks(kp_msg)
            else:
                self._frames_since_update += 1
            features = _select_features(self._propagated_features(), self._max_keypoints)
        else:
            kp_msg = self._get_live(self._q_features, timeout)
            if kp_msg is None:
                return None
            if self._autotuner is not None:
                self._autotuner.update(len(kp_msg.trackedFeatures))
            if self._frame_source is not None:
                frame_msg = self._get_live(self._q_frames, timeout)
                if frame_msg is None:
                    return None
                if not self._color:
                    frame = frame_msg.getCvFrame()
                # color mode keeps the BGR source frame for display/crops;
                # the passthrough read above only drains the queue.
            features = _select_features(list(kp_msg.trackedFeatures), self._max_keypoints)
        keypoints = [
            KeypointRecord(
                x=float(f.position.x),
                y=float(f.position.y),
                # Untracked mode: the firmware still numbers detections with
                # per-frame indices — normalize to -1 so consumers never
                # mistake them for persistent tracks.
                track_id=int(f.id) if self._tracking else -1,
                age=int(f.age) if self._tracking else 0,
            )
            for f in features
        ]
        # Describe selection: legacy mode describes the oldest
        # describe_budget tracks (a prefix of the selected features, sorted
        # by track ID); gated mode asks the DescribeGate for the new/changed/
        # stale subset. The rest are display/tracking-only this frame (gated
        # mode still publishes their cached codes).
        gate_plan = None
        if self._gate is not None:
            gate_plan = self._gate.select(
                features, color_src if self._color else frame
            )
            described = [features[i] for i in gate_plan.sel]
        else:
            described = features[: self._describe_budget]
        if not described:
            zshape = (0, 3, PATCH_SIZE, PATCH_SIZE) if self._color else (0, PATCH_SIZE, PATCH_SIZE)
            patches = np.zeros(zshape, dtype=np.uint8)
        elif self._color:
            # color_src is the BGR frame (host source, or the NV12 carrier);
            # a dropped color frame is a dropped frame, not a mismatch.
            patches = crop_opponent_planes(color_src, described)
        else:
            patches = crop_patches(frame, described)
        strip_nodes = self._send_strips(patches)
        job = {
            "frame": frame,
            "keypoints": keypoints,
            "features": features,
            "n_described": len(described),
            "strip_nodes": strip_nodes,
            "gate_plan": gate_plan,
        }

        # One-frame overlap: submit this frame's strips, then drain the
        # PREVIOUS frame's strips while the device works on the new ones.
        # Throughput becomes max(stage rates) instead of their sum.
        prev = self._pending
        self._pending = job
        self._last_output_ts = time.time()
        if prev is None:
            # First frame: nothing to overlap with, drain synchronously.
            self._pending = None
            return self._drain_job(job, timeout)
        return self._drain_job(prev, timeout)
