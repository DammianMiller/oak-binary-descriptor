# Binary Feature Descriptors on OAK-1 (RVC2)

This example runs a complete binary local-feature pipeline on a single OAK-1:

- **Keypoints on SHAVEs** — the on-device `FeatureTracker` node detects and tracks Harris (or Shi-Tomasi) corners on grayscale frames, giving each keypoint a stable track ID and age.
- **Descriptors on the NCE** — a small custom neural network describes 32x32 grayscale patches around keypoints. The host stacks **B patches into one 32x(32*B) GRAY8 "strip" frame** so a single inference describes all B: the graph convolves over the strip, then a Transpose/Reshape/Transpose inside the graph re-exposes the per-patch row blocks (Myriad X blobs silently collapse a true batch dim > 1, so the strip axis replaces it). The default archive is the slim strip-32 build (~13 ms/strip); a full strip-64 build is included for the max-keypoints mode. The network has two heads: a raw **512-bit** descriptor (`desc`, float32; its sign *is* the bit) and a **64-bit compressed code** (`code`) produced by a projection head baked into the model.
- **512 -> 64 bit compression** — the projection head implements one of four strategies selected at model-build time: Morton/Z-order-style `subsample`, `xorfold` chunk folding, SimHash `lsh`, or learned `itq` rotation (see [`core/projection.py`](core/projection.py)).
- **Host correlation** — crops are taken host-side (numpy) from the tracker's passthrough frame in a deterministic selection order. Strips are round-robined over **2 parallel NN nodes** and outputs drain one frame behind the sends (async overlap), so each descriptor row maps back to its keypoint by (node, arrival order) without extra synchronization messages.

The 64-bit codes are designed to stay stable while a keypoint moves, so temporal re-identification is a cheap XOR + popcount (Hamming distance) — see `hamming64` in [`core/projection.py`](core/projection.py) and `match_codes` in [`core/packer.py`](core/packer.py).

## Architecture

```
                 OAK-1 (RVC2)                                   Host
 ┌──────────────────────────────────────────────┐  ┌─────────────────────────────┐
 │                                              │  │                             │
 │  Camera ──► GRAY8 640x400 ──┬───────────────►│  │  display/crop frames        │
 │    (or ReplayVideo+grayscale)│  passthrough   │  │         │                   │
 │    (via Script decimator,   ▼                │  │         ▼ numpy: 32x32      │
 │     every Nth frame)   FeatureTracker        │  │     crops at keypoints,     │
 │            (Harris/Shi-Tomasi, 2 SHAVEs,     │  │     cap @ max_keypoints     │
 │             2 mem slices, ~684 max feats)    │  │     (oldest tracks first)   │
 │                             │ TrackedFeatures│  │    (skipped frames: host    │
 │                             └───────────────►│──┤     velocity propagation)   │
 │                                              │  │     into 32x(32*B) strips   │
 │              2x NeuralNetwork (NNArchive)    │  │         │ (round-robin      │
 │              strip in: (1,1,32*B,32) u8      │◄─┼─────────┘  input queues)    │
 │              code: uint8 (B,64)              │──┼──► strip outputs, ordered   │
 │              desc: float32 (B,512)           │  │         │                   │
 └──────────────────────────────────────────────┘  │         ▼ correlate by      │
                                                   │     (node, arrival order),  │
                                                   │     drained 1 frame behind  │
                                                   │  KeypointRecord + uint64    │
                                                   │  codes + 512-bit descs      │
                                                   │  -> "Keypoints" visualizer  │
                                                   └─────────────────────────────┘
```

Measured on OAK-1, live camera (640x400 GRAY8, 30 fps sensor) with the
default async pipeline (2 parallel NN nodes, one frame of descriptor
latency):

| tracker size | `max_keypoints` | decimate | frame rate | notes |
|---|---|---|---|---|
| **640x400 (default)** | **64 (default)** | **2 (default)** | **~31 fps** | 30 Hz at full resolution |
| 640x400 | 64 | 1 | ~31 fps | tracker keeps up device-fed |
| 640x400 | 684 (tracker max) | 1 | ~1.6 fps | strip-64 archive, max kps/frame |

Host-fed frames (benchmarks/CI) pay a ~15 ms/frame USB round trip: the same
640x400/64 config runs ~23 fps host-fed vs ~31 fps camera-fed. Host-fed +
`tracker_decimate > 1` deadlocks (XLink input-path contention between the cam
and strip queues), so the pipeline forces decimate=1 with a host frame
source.

`--tracker_decimate N` feeds the FeatureTracker every Nth frame (on-device
Script decimator) and propagates tracks host-side on skipped frames
(per-track velocity, median-velocity fallback); descriptors are still
computed every frame. N=2 halves tracker load for free at 30 Hz. If you
raise `--fps_limit` above 30, use `--tracker_decimate 1`: the decimator
Script caps at ~29 msg/s passthrough.

The FeatureTracker hardware ceiling is **~684 keypoints/frame** (2 memory
slices x 342). Keypoints/frame in the table's live-camera rows is
scene-limited (~32 corners on a plain desk view; the synthetic-texture
benchmark saturates the 64 budget at ~2000 described keypoints/s).

Intra-track temporal stability at all operating points: median 0 bits
differ (out of 64) between consecutive frames, p90 <= 2.

### Extraction-only mode (no tracking, no descriptors)

Detection-only benchmarks (`tools/sweep_autotune.py`, motion estimator off,
Harris threshold driven by the autotuner):

| sensor rate | max features/frame | limiting factor |
|---|---|---|
| 60 fps | **~390** | detector frame budget: threshold < ~200 overruns 16 ms and output collapses to ~46 fps / a truncated ~320 grid |
| 30 fps | **~750** | XLink metadata serialization: TrackedFeatures messages > 51200 B (~68 B/feature) are dropped on device |

At 30 fps the autotuner converges onto targets it can reach (measured:
300->282, 400->399, 500->496, 800->702 features, all at 30.0 fps). Scene
supply matters: a corner-poor scene caps out lower regardless of target.

### Opponent-color variant

An Opponent-LATCH-inspired (IEEE 9824924) color build: the CNN consumes the
three opponent planes (o1 red-green, o2 blue-yellow, o3 intensity; see
[`core/color.py`](core/color.py)) instead of raw grayscale, packed
channel-major into a 96*B-high GRAY8 strip (the graph re-exposes the channel
axis with a Reshape). Build it with the `--color` flag of the same tools
(see [tools/README.md](tools/README.md)); archives named `*_color*` are
auto-detected at runtime (`--color/--no_color` overrides). The tracker
itself always runs on grayscale.

Training-time quality (same self-supervised warp-pair protocol, this build
vs the grayscale build): 512-way warp retrieval **0.545 vs 0.35**, and
**0.621 on isoluminant pairs** (flat intensity channel — a grayscale
descriptor is at chance there). ITQ 64-bit separation 0.962, preserved on
isoluminant pairs (iso_sep 0.962).

On-device (live camera, USB2 link), grayscale for comparison:

| `max_keypoints` | color fps | grayscale fps | notes |
|---|---|---|---|
| 32 (1 strip/frame) | **29.3** | ~31 | 30 Hz color met; stab med 1 bit, p90 2 |
| 64 (2 strips/frame) | 23.3 | 29.6 | USB2 payload limit (below) |

The color throughput cap is USB bandwidth, not NN compute (the color blob is
only ~10% slower per strip: 96 vs 106 strips/s). Two measured mitigations
are built in: the color stream is requested as **NV12** (half the bytes of
BGR888i, which alone starved the pipeline to ~14 fps), and **no grayscale
frame is streamed to the host in color mode** (the NV12 stream doubles as
the frame carrier; with both streams, 64 kps/frame capped at ~20 fps). A
USB3 host lifts the 64 kps/frame cap.

#### Dense mode: decoupled tracking vs describing

Describing *every* tracked keypoint is bounded by NCE throughput: the blob
aggregates ~3000 described keypoints/s (96 strips/s), so describe-everything
caps at 128 kps @ ~14 fps, 256 @ ~7.5, 384 @ ~5 (measured). To show many
features without paying that, pass `describe_budget < max_keypoints`
(`--describe_budget` on the CLI / serve_visual): the tracker targets
`max_keypoints` (up to the ~684 hardware ceiling) and every track is
returned, but only the oldest `describe_budget` tracks get codes each frame
(codes align with the FIRST entries of the keypoints list; `pack_frame`
marks the rest `has_zorder=False`). Measured on-device, live camera:

| tracked (target 684) | described/frame | fps |
|---|---|---|
| ~610 | 64 | **20.9** |
| ~610 | 128 | 12.2 |

A `*_codeonly*` archive variant (`make_projection.py --code_only` +
`compile_blob.py --code_only`) drops the float `desc` output from the graph
(2 KB instead of 66 KB streamed per strip). Measured gain is small (+1-2
fps), because the ceiling is NCE compute and per-message overhead rather
than the desc payload; use it when every MB/s counts.
`FeaturePipeline` auto-detects code-only archives and skips desc decoding.

#### Change-gated describing ("describe only what moved")

`--describe_gate` (CLI/serve_visual; `describe_gate=true` in the ROS nodes)
keeps a host-side cache of code + desc + an 8x8 block-mean patch
fingerprint per keypoint identity (track ID, or spatial nearest-neighbor
matching in no-track mode) and describes only keypoints that are NEW, whose
patch content changed above `--change_thresh`, or whose code is
`--refresh_every` frames stale — capped at `describe_budget` per frame,
most stale first (round-robin fairness under budget pressure). Unchanged
keypoints publish their cached codes, so the output is dense every frame
while strip traffic tracks scene motion: static scenes approach zero
describes/frame at full frame rate; a fully-moving scene degrades
gracefully to describe-everything. Measured on-device (color archive, live
camera, moderately active scene):

| config | fps | coded/frame | fresh describes/frame |
|---|---|---|---|
| ungated 256 described | 7.5 | 256 | 256 |
| gated tracked 256, budget 64 | 21.5 | 256 | ~47 |
| gated no-track 256, budget 64 | **22.8** | 256 | ~40 |
| gated no-track 384, budget 96 | 17.2 | 384 | ~55 |

Re-identification stays at 100% (Hamming mean 0.5-0.8/64) — cached codes
are exactly the codes the NN would reproduce for unchanged patches. The
output contract changes in this mode: codes/descs come back **parallel** to
the keypoints list (not a prefix), with per-keypoint `has_code` (False only
for never-described keypoints) and `code_age` (frames since the code was
actually described) on `KeypointRecord`; `pack_frame` maps these onto
`has_zorder`. The ROS message is unchanged.

#### No-tracking mode (pure descriptor re-identification)

`--no_tracking` (CLI/serve_visual; `tracking=false` in the ROS nodes and
`FeaturePipeline`) disables the FeatureTracker's motion estimator:
detection-only, every frame, no track IDs (keypoints report `track_id=-1`).
Every detected keypoint is described, and cross-frame re-identification is
purely descriptor-based (best-Hamming code matching) — which is what the
64-bit codes are for. Measured on-device with the color archive, live
camera: **~100% of keypoints re-identified between consecutive frames**
(match threshold 12/64 bits) at Hamming mean ~1.0-1.4.

Density vs fps is a USB2-bandwidth trade (the ~3000 described-keypoints/s
NCE ceiling is never reached here). In no-track color mode the color stream
defaults to **half resolution** (`color_scale=2`: 96 KB instead of 384
KB/frame; the host upscales x2 before cropping — re-id quality is
unaffected, and going to quarter resolution gains ~0.5 fps, so scale 2 is
kept). Measured with half-res color, matching disabled:

| described/frame | fps |
|---|---|
| 64 | **23.8** (2x the old 32/frame @ 30 Hz density) |
| 96 | 16.8 |
| 128 | 13.1 |

Do **not** try to thin the color stream with `requestOutput(fps=...)` to
win the bandwidth back at full resolution: the fps parameter throttles the
*whole sensor* (measured: gray stream 30 -> 16 fps when the color output
was capped at 15).

In this mode the ROS stream carries exactly timestamp (array header), x, y,
`desc`, and the 64-bit `zorder` code per keypoint — no tracking fields —
and the visual preview disables host-side matching (dots only); consumers
such as ROS subscribers do their own code matching.

## Quickstart

### 1. Build the model

The descriptor NNArchive is **not** committed — build it first:

```bash
# see tools/README.md for the full flow and options
python3 tools/train_descriptor.py           # optional but recommended: trained weights + ITQ calibration
python3 tools/export_descriptor_onnx.py --strip 32 --slim \
    --weights depthai_models/descriptor_weights_slim32.pth \
    --out depthai_models/descriptor_base_trained_slim32.onnx
python3 tools/make_projection.py --strategy itq \
    --calibration depthai_models/descriptor_calib.npy --inject \
    depthai_models/descriptor_base_trained_slim32.onnx depthai_models/descriptor64_itq_slim32.onnx
python3 tools/compile_blob.py --onnx depthai_models/descriptor64_itq_slim32.onnx \
    --strategy itq --strip 32 --shaves 6 --out depthai_models/descriptor64_itq_strip32_slim.tar.xz
```

Trained vs random-init weights, measured on-device: intra-track temporal
stability is excellent either way (median 0 bits), but inter-keypoint
discrimination goes from ~4.5 bits (unusable: different keypoints get
near-identical codes) to **~25 bits** mean pairwise Hamming after training.
Host-side warp-pair separation of the 64-bit codes: ITQ 0.910, LSH 0.893
(`tools/eval_descriptor.py`).

### 2. Install dependencies and run

Running this example requires a **Luxonis OAK-1** (RVC2) connected to your computer. Refer to the [documentation](https://docs.luxonis.com/software-v3/) to set up your device if you haven't done it already.

```bash
pip install -r requirements.txt
python3 main.py
```

Open the DepthAI Visualizer (default `http://localhost:8082`) and watch the `Keypoints` topic. The console prints the keypoint count and the mean Hamming distance between consecutive-frame code matches (low values = temporally stable codes).

Useful flags:

```
-a, --archive       path to the NNArchive (default: depthai_models/descriptor64_itq_strip32_slim.tar.xz)
--max_keypoints     per-frame descriptor budget (default: 64)
--detector          harris | shi_tomasi (default: harris)
--compression       subsample | xorfold | lsh | itq (informational; must match the built archive; default: itq)
--tracker_decimate  feed the tracker every Nth frame, propagate tracks host-side in between (default: 2)
--motion_estimator  sw | hw (default: sw; hw block matching wins for extraction-only, not here)
--tracker_size      tracker stream WxH (default: 640x400)
--nn_nodes          parallel NN replicas (default: 2)
--color/--no_color  force the opponent-color path on/off (default: auto-detect from the archive name)
--no_full_desc      only decode the 64-bit codes, skip the 512-bit descriptors
-media, --media_path  run on a video file instead of the camera
-fps, --fps_limit   cap the pipeline FPS
-d, --device        connect to a specific device
```

## Using it as a library

```python
from core.pipeline import FeaturePipeline

with FeaturePipeline("depthai_models/descriptor64.tar.xz", max_keypoints=684) as fp:
    frame, keypoints, codes, full_descs = fp.next_frame()
    # frame:      (H, W) uint8 grayscale ndarray
    # keypoints:  list[KeypointRecord]  (x, y, track_id, age)
    # codes:      (n,) uint64 compressed descriptor codes
    # full_descs: (n, 64) uint8 full 512-bit descriptors (None with publish_full_desc=False)
```

## ROS

The keypoint + code stream is packaged for ROS in the [`ros1/`](ros1/) and [`ros2/`](ros2/) subfolders (message packing lives in [`core/packer.py`](core/packer.py), mirroring `oak_features_msgs/Keypoint`). See [apps/ros/ros-driver-custom-workspace](https://github.com/luxonis/oak-examples/tree/main/apps/ros/ros-driver-custom-workspace) for the workspace-level integration pattern.

## Links

- [DepthAI documentation](https://docs.luxonis.com/software-v3/)
- [XFeat example](https://github.com/luxonis/oak-examples/tree/main/neural-networks/feature-detection/xfeat) — learned float descriptors with reference-frame and stereo matching
- [generic-example](https://github.com/luxonis/oak-examples/tree/main/neural-networks/generic-example) — single-model scaffold this example's CLI style follows
