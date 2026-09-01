# Descriptor model tooling

Builds the on-device model for this example: a small HardNet-style CNN that
maps a 32x32 grayscale patch to a 512-d raw descriptor (`desc`), plus a baked-in
512 -> 64 bit compression head that emits a uint8 `code` output from the same
graph. One inference yields both outputs.

## Pipeline

```
export_descriptor_onnx.py          descriptor_base.onnx       (CNN: patch -> desc)
        |
make_projection.py --inject        descriptor64_<strategy>.onnx  (+ MatMul/Greater/Cast head)
        |
compile_blob.py                    descriptor64_<strategy>.tar.xz  (RVC2 NNArchive)
```

Other tools: `capture_visual.py` (headless annotated PNG captures),
`serve_visual.py` (MJPEG live feed, `--extraction_only` detection-only mode),
`sweep_autotune.py` (on-device sweep that converges the autotuner on ascending
feature targets at a given sensor rate and reports the max sustained
features/frame — e.g. `--fps 60 --targets 300 400 500`; `--no_tune` runs a
fixed-threshold control).

### 1. Export the base CNN

```bash
python3 tools/export_descriptor_onnx.py --strip 32 --slim [--weights hardnet.pth] [--out depthai_models/descriptor_base_slim32.onnx]
```

Use `--strip 64` (without `--slim`) for the maximum-keypoints-per-frame
variant.

- Architecture: 7x conv3x3 (32, 32, 64/s2, 64, 128/s2, 128, 128/s2) with ReLU
  except after the last conv, then GlobalAveragePool -> Flatten -> Gemm to 512.
- `--strip N` (the mode the example ships with): the graph input is one
  (1,1,32*N,32) strip of N vertically stacked patches; after the conv stack a
  Transpose/Reshape/Transpose folds the strip axis back out so GAP+Gemm emit N
  descriptors in ONE inference. This matters because Myriad X blobs collapse a
  true batch dim > 1 (a (64,1,32,32) blob computes only the first patch), and
  per-patch inferences are overhead-bound at ~5.3 ms each (~190 desc/s).
  Measured per-strip inference times (6 shaves, threads=1):

  | build | ms/strip | desc/s (1 NN node) | desc/s (2 NN nodes) |
  |---|---|---|---|
  | full strip-64 | ~54 | ~1200 | ~2400 |
  | full strip-32 | ~24 | ~1350 | ~2700 |
  | **slim strip-32 (default)** | **~13** | **~2400** | **~3400** |
  | slim strip-16 | ~7.3 | ~2190 | ~3300 |

  (3 nodes adds nothing over 2 — the NCE saturates.) Caveats:
  - Adjacent patches in a strip bleed through conv receptive fields at block
    boundaries (measured: a patch change perturbs its own and +/-1 output
    rows). Fine for bring-up; retrain in strip mode for production quality.
  - Keep `nn.setNumInferenceThreads(1)`: with 2 threads a saturated input
    queue can wedge the node (observed empirically; request-response works).
- `--slim` halves the conv widths (~4x fewer MACs). Unlike batch-1 per-patch
  inference (overhead-bound, slim adds nothing), strip mode is compute-bound,
  so slim is a real ~1.8x speedup there and is the shipped default.
- BatchNorm is omitted on purpose (plain Conv+ReLU maximizes OpenVINO myriad
  compatibility; with trained weights, BN is expected to be folded into conv
  weights beforehand).
- Without `--weights` (or if torch is not installed / the checkpoint is not
  recognized), a seeded random init is used and a loud warning is printed:
  the descriptor is UNTRAINED, only suitable for pipeline bring-up.
- Real descriptor quality requires trained HardNet-compatible weights via
  `--weights` (torch needed only for this step; it is not a runtime
  dependency and not in requirements.txt).

### 1b. (Optional) Train the weights — train_descriptor.py

```bash
pip install torch  # CPU build is fine; tool-only dependency
python3 tools/train_descriptor.py [--steps 12000] [--batch 256] \
    [--out depthai_models/descriptor_weights_slim32.pth] \
    [--calibration_out depthai_models/descriptor_calib.npy]
```

- Self-supervised, no dataset download: random synthetic textures (smoothed
  noise + shapes + gradients), anchor/positive pairs related by random affine
  warps (curriculum: mild warps, then up to +/-60 deg / 0.65-1.35x scale) plus
  brightness/contrast/noise jitter, trained with in-batch InfoNCE. This
  optimizes exactly the pipeline's requirement: stable codes under
  per-frame motion.
- The graph has no BatchNorm, so intensity invariance comes from the
  training-time intensity jitter (BN-free invariance).
- +/-180 deg rotation augmentation was tried and collapses training (rotation
  invariance at that range destroys discrimination); keep the strong phase
  at or below ~60 deg.
- Saves a checkpoint the export tool accepts via `--weights`, plus a
  (5120, 512) calibration dump for the ITQ head in step 2.
- Data generation is vectorized (torch `affine_grid`/`grid_sample`); a
  per-patch numpy warp generator was ~8 steps/s, the vectorized one is
  CPU-bound on the forward/backward instead.

#### Color variant (Opponent-LATCH style) — `--color`

```bash
python3 tools/train_descriptor.py --color [--steps 10000] [--batch 128]
python3 tools/export_descriptor_onnx.py --color --strip 32 --slim \
    --weights depthai_models/descriptor_weights_slim32_color.pth \
    --out depthai_models/descriptor_base_color.onnx
python3 tools/make_projection.py --strategy itq \
    --calibration depthai_models/descriptor_calib_color.npy --inject \
    depthai_models/descriptor_base_color.onnx depthai_models/descriptor64_itq_color.onnx
python3 tools/compile_blob.py --color --onnx depthai_models/descriptor64_itq_color.onnx \
    --strategy itq --strip 32 --shaves 6 \
    --out depthai_models/descriptor64_itq_strip32_slim_color.tar.xz
```

- The network consumes the three opponent planes (o1 red-green, o2
  blue-yellow, o3 intensity; see `core/color.py`) instead of raw grayscale,
  following Opponent-LATCH (IEEE 9824924). Textures are generated directly in
  normalized opponent space so chromatic edges are guaranteed (the net cannot
  ignore color): ~30% isoluminant patches (flat intensity), ~15% achromatic
  (zero chroma, so grayscale scenes still work), plus hue-rotation and
  saturation jitter in the (o1, o2) plane.
- Training also prints an ISOLUMINANT retrieval accuracy: intensity-flat
  pairs that a grayscale descriptor provably cannot separate.
- The strip packs the 3 planes channel-major into a (1,1,96*B,32) GRAY8
  frame; the graph re-exposes the channel axis with a Reshape and maps the
  uint8 storage (zero at 127.5) back to [-1,1] with Sub/Div.
- `eval_descriptor.py --color` compares the compression strategies and adds
  `iso_pos`/`iso_sep` columns for the isoluminant pairs.
- Optional: `make_projection.py --code_only` + `compile_blob.py --code_only`
  drop the float `desc` graph output (the blob streams only the 64-bit
  codes, ~2 KB instead of ~66 KB per strip). Measured gain is small (the
  high-keypoint ceiling is NCE compute, ~3000 described keypoints/s, not the
  desc payload); `FeaturePipeline` auto-detects such archives.

### 2. Bake in the compression head

```bash
python3 tools/make_projection.py --strategy {subsample,xorfold,lsh,itq} \
    [--seed 0] [--calibration calib.npy] [--matrix_out W.npy] \
    --inject depthai_models/descriptor_base.onnx depthai_models/descriptor64_<strategy>.onnx
```

- Appends `MatMul(desc, W^T) -> Greater(0) -> Cast(uint8)` so the graph has
  two outputs: `desc` (N,512) float32 and `code` (N,64) uint8 (the packed
  64-bit compressed descriptor / zorder); N is the strip size (or batch).
- `--strategy itq` additionally requires `--calibration`: an `(n, 512)` float32
  `.npy` of raw descriptor outputs.
- Calibration collection: run the pipeline with an uninjected (or any) blob
  and dump the raw `desc` tensors for a few thousand representative patches,
  save them stacked as `(n, 512)` `.npy`, then re-run this step with
  `--strategy itq --calibration ...` and recompile. See
  `tools/benchmark_compression.py` for a host-side preview of ITQ quality on
  BRISK descriptors.

### 3. Compile for Myriad X (RVC2) and package the NNArchive

```bash
python3 tools/compile_blob.py --onnx depthai_models/descriptor64_<strategy>.onnx \
    --strategy <strategy> --strip 32 [--shaves 6] [--version 2022.1] [--data_type FP16] \
    [--out depthai_models/descriptor64_<strategy>.tar.xz]
```

- `--strip N` must match the ONNX's strip size: it sets the NNArchive input
  shape to (1,1,32*N,32) and the output shapes to (N,512)/(N,64). `--batch` is
  legacy: Myriad X collapses batch > 1 to the first patch at runtime.
- Known device quirk: a batch-1 (non-strip) blob whose NN input is fed from a
  HOST queue deadlocks when a FeatureTracker node is also in the pipeline
  (the NN never consumes). The strip blob (and device-linked inputs in
  general) do not hit this; it is one reason the strip architecture exists.
- Moving strip assembly onto the device via a `Script` node was tried and is
  infeasible: `frame.getData()` on a 256 KB frame inside the Script runtime
  hard-crashes the device, even with a 1 KB output frame (passthrough without
  `getData()` works). Strip assembly stays host-side (numpy).

- Uses the Luxonis blobconverter cloud service (`pip install blobconverter`,
  a tool-only dependency); network access is required and the tool fails with
  a clear error if the service is unreachable.
- Flags: `--no_cache` forces a fresh server-side compile (the client cache is
  keyed by model content AND compile params, but stale blobs have been served
  before — use this if the device reports unexpected input sizes), and
  `--insecure` disables TLS verification (needed while
  blobconverter.luxonis.com serves an expired certificate).
- The blob is compiled with a U8 input (`-ip U8`): GRAY8 crops are 1 byte/px
  (1024 B) and the NN node rejects frames smaller than the blob's input
  tensor; the graph's first op (Div by 255) normalizes 0-255 -> [0,1]
  in-graph, so no host-side normalization is needed.
- Packages the blob as a DepthAI v3 NNArchive `.tar.xz` (`config.json` +
  `.blob` + `buildinfo.json`); the config is verified by loading the archive
  with `depthai` itself.
- `input_type` is `raw`: the pipeline feeds raw 0-255 GRAY8 strip frames
  (32 x 32*N); no on-device color conversion or scaling, and the in-graph Div
  by 255 handles normalization.
- `heads` is null (no parser): `ParsingNeuralNetwork` exposes the raw `desc`
  and `code` tensors.

## Host-side evidence before compiling

```bash
python3 tools/benchmark_compression.py
```

Matches BRISK descriptors across synthetic warps, full 512 bits vs each
64-bit compression strategy, and prints precision/recall tables. Note that it
feeds packed 0/1 bits in place of the raw float descriptor, which
systematically penalizes `xorfold` (sum of 0/1 bits with a `>0` threshold
degenerates to an OR-fold; with signed raw descriptor outputs it acts as a
majority/fold vote as designed). `subsample` and `itq` show the expected
behavior on this proxy.
