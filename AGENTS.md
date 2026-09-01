# AGENTS.md

## Summary

On-device FeatureTracker keypoints (Harris/Shi-Tomasi) described by a learned binary descriptor network on the Myriad X NCE, compressed 512 -> 64 bits (Morton/LSH/ITQ-style head), for OAK-1 (RVC2). Use it when you need trackable keypoints with cheap Hamming-matchable binary codes rather than float descriptors.

## Use This Example When

- You need on-device keypoint detection/tracking plus a compact binary descriptor per keypoint.
- You want temporal re-identification of keypoints with XOR+popcount Hamming matching on 64-bit codes.
- You need a reference for strip-batched inference: N image patches stacked into one tall GRAY8 frame, with the batch axis folded out inside the ONNX graph (Myriad X blobs collapse a true batch dim > 1).

## Do Not Use This Example When

- You need float descriptors or learned matching quality comparable to XFeat/LightGlue.
- You need stereo depth, spatial coordinates, or multi-camera setups.
- You need RVC4; this example targets RVC2 (OAK-1) only.
- You have not built the descriptor NNArchive with [tools/](tools/) yet; there is no committed or Model Zoo model.

## Quick Facts

- `Category:` `neural-networks/feature-detection/binary-descriptor`
- `Shape:` `script`
- `Primary task:` tracked keypoints with 64-bit binary descriptor codes (plus optional full 512-bit descriptors)
- `Entrypoint:` [main.py](main.py)
- `Frontend:` none
- `Runs on:` RVC2 peripheral (OAK-1)
- `Requires:` OAK-1 device and a descriptor NNArchive built with [tools/](tools/) first (see [tools/README.md](tools/README.md)); the default archive path is [depthai_models/descriptor64_itq_strip32_slim.tar.xz](depthai_models/) (trained slim strip-32 + ITQ head, 6 shaves); [depthai_models/descriptor64.tar.xz](depthai_models/) (full strip-64) is the max-keypoints-per-frame variant; [depthai_models/descriptor64_itq_strip32_slim_color.tar.xz](depthai_models/) is the opponent-color variant (auto-detected by the "color" name, `--color/--no_color` overrides)
- `Input:` live camera by default or `ReplayVideo` via `--media_path` (note: ReplayVideo on RVC2 is codec-sensitive; it hard-crashed the device on 1080p h264 during development)
- `Output:` Visualizer topic `Keypoints` (annotated grayscale frame); host-side keypoints + codes via `FeaturePipeline.next_frame()`
- `Models:` custom strip descriptor NNArchive: input (1,1,32*B,32) uint8 (B stacked 32x32 patches), outputs `code` (uint8, (B,64)) and `desc` (float32, (B,512))
- `Throughput:` async pipeline (2 parallel NN nodes, one frame of descriptor latency): **~31 fps at the default 640x400 / 64 kps/frame, live camera** (30 Hz target met; camera-fed removes the ~15 ms host-feed USB round trip that caps host-fed at ~23 fps). ~684 (tracker hardware max) @ 1.6 fps with the strip-64 archive. `--tracker_decimate N` (default 2) runs the tracker on every Nth frame via an on-device Script decimator and propagates tracks host-side in between; the tracker target follows `--max_keypoints`
- `Visualizer / UI:` DepthAI Visualizer via `dai.RemoteConnection`

## Read First

- [README.md](README.md)
- [main.py](main.py): entrypoint, visualizer loop, per-frame Hamming stats
- [core/pipeline.py](core/pipeline.py): pipeline construction, host-side numpy cropping + strip stacking, host-side correlation
- [core/arguments.py](core/arguments.py): CLI options
- [core/packer.py](core/packer.py): `KeypointRecord`, message-shaped packing, `match_codes`
- [core/projection.py](core/projection.py): 512->64 bit compression strategies, `pack_bits`, `hamming64`
- [tools/](tools/): model training/compile flow that produces the NNArchive

## Architecture

- The camera (or `ReplayVideo` plus a grayscale `ImageManip` when `--media_path` is set) produces a GRAY8 stream at `--tracker_size` (default 640x400).
- With `--tracker_decimate N` > 1 (default 2), a `Script` decimator forwards every Nth frame to the tracker; the host propagates tracks on skipped frames (per-track velocity from the last two observations, global-median fallback for new tracks) and still crops/describes at the full frame rate from the raw stream queue. The decimator Script must pass frames untouched: `frame.getData()` on a full-size frame hard-crashes the device, and plain passthrough caps at ~29 msg/s (so use decimate=1 when `--fps_limit` > 30).
- `FeatureTracker` runs Harris or Shi-Tomasi keypoint detection/tracking on the Myriad X SHAVEs (`setHardwareResources(2, 2)` -> ~684 tracked features max, 342 per memory slice) and emits `TrackedFeatures` plus the aligned passthrough frame. Its `outputFeatures` lag the passthrough frames by ~4 frames; `FeaturePipeline.next_frame()` correlates them by arrival order (works because both queues are drained once per loop) and drops frames on drift.
- The host (`FeaturePipeline`) caps keypoints at `--max_keypoints` (FIFO by ascending track ID), numpy-crops 32x32 patches from the passthrough frame, and stacks them into 32x(32*B) GRAY8 "strip" frames round-robined over `nn_nodes` parallel `NeuralNetwork` replicas (default 2; measured ~1.4x over 1 node via DMA/compute overlap, 3 nodes saturate the NCE).
- The pipeline is async: `next_frame()` sends the current frame's strips, then drains the PREVIOUS frame's strip outputs from each NN node in order, so tracker/crop/NN stages overlap and results lag input by one frame (the first call is synchronous).
- The NN input shape is (1,1,32*B,32) batch-1 on purpose: Myriad X blobs collapse a true batch dim > 1 to a single patch (verified: changing patch 5 of a (64,1,32,32) blob's input changed no output). Inside the graph a Transpose/Reshape/Transpose after the conv stack folds the strip axis out, so GAP+Gemm emit one descriptor per patch in ONE inference (slim strip-32 ~13 ms, full strip-64 ~54 ms, vs ~5.3 ms/patch individually).
- The NN runs with `setNumInferenceThreads(1)`: strip outputs must match strip send order, and a saturated host input queue with 2 threads wedged the node in testing.
- `FeaturePipeline.next_frame()` returns (frame, keypoints, uint64 codes, full 512-bit descriptors) for the PREVIOUS frame in async mode.
- Opponent-color mode (archive name contains "color", `--color/--no_color` overrides): the camera's NV12 stream at the tracker size is the ONLY frame streamed to the host (ImgFrame.getCvFrame decodes it to BGR; no grayscale frame crosses USB in color mode) and features pair with it by arrival order, same as the decimated raw-stream contract. Host crops become (o1, o2, o3) opponent planes ([core/color.py](core/color.py)) packed channel-major into a 96*B-high GRAY8 strip; a Reshape inside the graph re-exposes the channel axis. Both choices are measured USB2 (~30 MB/s practical) mitigations: BGR888i alone (~23 MB/s at 30 fps) starved the pipeline to ~14 fps, and NV12 + a host gray stream capped 64 kps/frame at ~20 fps. Final on USB2: 29.3 fps at <=32 kps/frame (1 strip), 23.3 fps at 64 (2 strips) vs 29.6 grayscale; the color blob itself is only ~10% slower per strip (96 vs 106 strips/s), so a USB3 host lifts the cap. In no-track color mode the color stream additionally defaults to half resolution (`color_scale=2`, 96 KB/frame, host x2 upscale; re-id quality unaffected): 64 described keypoints/frame at 23.8 fps, 96 @ 16.8, 128 @ 13.1. Never thin a camera stream with `requestOutput(fps=...)` — the fps parameter throttles the WHOLE sensor (measured: capping the color output at 15 fps dropped the gray stream 30 -> 16 fps).
- [main.py](main.py) draws keypoints colored by track ID, prints the count plus mean intra-frame-pair Hamming distance between consecutive frames, and publishes the annotated frame on the `Keypoints` topic.

## Data Flow

- `camera or media file -> GRAY8 -> FeatureTracker -> TrackedFeatures + passthrough frames -> host queues`
- `host: crop 32x32 patches (numpy) -> stack B into a 32x(32*B) strip -> round-robin over nn_nodes NN input queues -> NeuralNetwork x2 -> NNData ("code" (B,64), "desc" (B,512)) -> host (drained one frame behind)`
- `host: (features, strip rows) correlated by (node, arrival order) -> KeypointRecord / uint64 codes / 64-byte descriptors`

## Modification Guide

- `Safe to change:` `--max_keypoints`, `--no_tracking` (detection-only mode: motion estimator off, decimate forced to 1, keypoints report track_id -1, cross-frame re-identification purely by descriptor code — measured ~100% frame-to-frame re-id at Hamming mean ~1/64; ROS nodes expose it as `tracking=false`; `color_scale` defaults to 2 in this mode), `color_scale` (color-stream downscale; see the color bullet for the measured fps-throttle dead end), `--describe_budget` (decouples tracked/displayed count from described count: codes align with the FIRST describe_budget keypoints, oldest tracks first; the NCE aggregates ~3000 described keypoints/s so describe-everything caps fps at ~3000/max_keypoints — measured color: 610 tracked @ 20.9 fps with 64 described, 12.2 fps with 128), `--describe_gate`/`--change_thresh`/`--refresh_every` ([core/gate.py](core/gate.py): describe only new/changed/stale keypoints, cached codes published for the rest — 256 coded @ 22.8 fps with ~40 fresh describes/frame vs 7.5 fps ungated; output contract changes to codes PARALLEL to keypoints with per-record has_code/code_age), `--fps_limit`, `--detector`, `--tracker_size`, `--nn_nodes`, visualizer topic names, host-side drawing in [main.py](main.py)
- `Autotune:` `ThresholdAutotuner` in [core/pipeline.py](core/pipeline.py) converges the feature count onto a target by scaling `thresholds.initialValue` via the tracker's `inputConfig` queue (20-frame windows, <=1 update per 0.7 s, sqrt-damped steps clamped to [0.6, 1.6], +/-5% deadband, band 15-2000). On by default in `FeaturePipeline`; disable with `--no_autotune`. Hard-won constraints: (1) do NOT use the firmware's adaptive `thresholds.min/max` band — it re-detects iteratively within a frame and collapsed throughput to ~1 fps; (2) at a 60 fps sensor rate, thresholds below ~200 overrun the detector's frame budget and output collapses to ~46 fps on a truncated ~320-feature grid — the optional `min_fps` guard watches for exactly this (fires only when fps fell right after the loop's own lowering; disables itself after 3 unproductive raises so a dark, exposure-limited stream never ping-pongs); (3) TrackedFeatures metadata over 51200 B (~750 features) is dropped on the XLink path, capping 30 fps extraction at ~750 features/frame; (4) build `numTargetFeatures` equal to the tuner target — a runtime-only change does not resize the detection grid, and a higher build-time cap lets the count overshoot and walk the threshold to its ceiling.
- `Requires care:` the arrival-order correlation contract (one features+frame queue drain per loop; per-NN-node outputs consumed in strip send order; the one-frame async drain delay), the codes-as-prefix contract when `describe_budget` is set (consumers must tolerate `len(codes) < len(keypoints)`; `core/packer.py` `pack_frame` and the ROS `has_zorder` field already do), the gate-mode parallel-codes contract (codes len == keypoints len, validity via `KeypointRecord.has_code`; the gate sees codes through frame k-2 because apply() runs at drain time during the next next_frame() call; gate fingerprints must stay on the integral-image path — the naive per-keypoint gather costs ~12 ms/frame at 256 color keypoints and halves throughput), `setNumInferenceThreads(1)`, the archive's strip size (`max(shape[0], shape[2] // 32)` in the config) matching the runtime stacking, the `--compression` flag staying in sync with the compiled archive, `--motion_estimator hw` (block matching is a win for extraction-only workloads — flat ~60 fps at 1000 targets — but the descriptor pipeline is NN-bound either way and SW optical flow gives more stable codes: median 0/p90 2 vs median 1/p90 3-10), the color mode's o3 normalization in [core/color.py](core/color.py) (o3 = 2(R+G+B)/765 - 1; the training generator samples o3 across the full [-1, 1], so any narrower/saturating runtime mapping is a train/inference skew)
- `Likely to break if changed blindly:` 32x32 patch size or strip size versus the compiled model input, tensor names `code`/`desc`, switching to multiple NN inference threads (ordering + a wedge observed with saturated queues), rebuilding the archive without `--strip` (a batch-1 non-strip blob fed from a host queue deadlocks when a FeatureTracker node is in the same pipeline), draining NN outputs synchronously instead of the one-frame-behind async pattern, using `tracker_decimate > 1` with a host frame source (free-running host-fed loop + strip sends wedge the XLink input path; the constructor forces decimate=1 there)

## Constraints

- RVC2 (OAK-1) only; the descriptor NNArchive must be built with [tools/](tools/) before running.
- Device teardown/reopen (measured, bisected): close PROMPTLY (immediate `pipeline.stop()` + `device.close()`; draining queues or settling in between leaves the next session's queues permanently empty) and reopen IMMEDIATELY (a ~2 s idle gap reproducibly yields dead sessions that open fine but produce no frames until the process fully exits). Rarely (~1 per several heavy transitions) teardown hangs the firmware outright (watchdog errorId 9001, auto-reset; dumps in `~/.cache/depthai/crashdumps/`). Mid-session death handling: `next_frame` raises a loud RuntimeError on X_LINK_ERROR QueueExceptions or 8 s of total silence; a session flagged dead is NEVER closed (`__exit__` skips stop/close) because depthai's closeImpl RPCs the dead device and segfaults the host (measured via `DEPTHAI_CRASH_DEVICE=1 device.crashDevice()`: `PipelineImpl::stop -> DeviceBase::closeImpl -> hasCrashDump -> nanorpc` on a dead connection); the process should exit instead, which releases the USB session. In-process recovery after a real crash is not possible (reopen hits ALREADY_IN_USE / segfault paths) — restart the process. A third failure mode: the NCE can wedge mid-run while camera/tracker streams keep flowing (observed live at 256 described/frame: NN output queues go silent, frames keep arriving, so the 8 s silence detector cannot see it) — `_drain_job` raises a loud RuntimeError after 3 consecutive frames with missing NN outputs. `serve_visual.py` handles SIGTERM so `pkill` closes the device cleanly, and exits with status 1 when the pipeline dies so a supervisor can restart it.
- Cropping and strip stacking happen host-side (numpy), so throughput is host/USB dependent; keypoints beyond `--max_keypoints` get no descriptor that frame. Moving strip assembly into a `Script` node was tried and is infeasible: `frame.getData()` on a 256 KB frame inside the Script runtime hard-crashes the device (even with a 1 KB output).
- Adjacent patches in a strip bleed slightly through conv receptive fields at block boundaries (measured: a patch change perturbs its own and +/-1 output rows); acceptable for bring-up, retrain in strip mode for production.
- `--compression` is informational; the actual projection is baked into the blob at build time.
- Frames with zero keypoints or missing NN outputs are dropped gracefully (zero-keypoint frame plus a warning).

## Related Examples

- [neural-networks/feature-detection/xfeat](https://github.com/luxonis/oak-examples/tree/main/neural-networks/feature-detection/xfeat): use this when you need higher-quality learned float descriptors and reference-frame/stereo matching
- [neural-networks/generic-example](https://github.com/luxonis/oak-examples/tree/main/neural-networks/generic-example): use this when you need the generic single-model scaffold
- [apps/ros/ros-driver-custom-workspace](https://github.com/luxonis/oak-examples/tree/main/apps/ros/ros-driver-custom-workspace): use this as the ROS integration baseline; this example's [ros1/](ros1/) and [ros2/](ros2/) subfolders package the keypoint stream

## Validation

- `Build the model first:` see [tools/README.md](tools/README.md), producing `depthai_models/descriptor64_itq_strip32_slim.tar.xz` (default, trained + ITQ head) or the strip-64 variant
- `Run:` `python3 main.py` (requires a connected OAK-1)
- `Media run:` `python3 main.py --media_path <FILE>`
- `Success looks like:` the Visualizer shows the `Keypoints` topic with colored keypoint circles, and the console prints keypoint count plus mean Hamming distance between consecutive frames
- `Common failure meaning:` the NNArchive is missing (build it with tools/), no device is connected, or the `--compression` flag does not match the compiled archive
