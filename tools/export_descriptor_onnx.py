"""Export the base descriptor CNN to ONNX, without any compression head.

Builds a HardNet-style patch descriptor network in pure ``onnx`` (no torch or
onnxruntime needed)::

    input (1,1,32,32) float32, grayscale patch normalized to [0, 1]
      -> conv3x3( 32) / ReLU          32x32
      -> conv3x3( 32) / ReLU          32x32
      -> conv3x3( 64, stride 2) / ReLU 16x16
      -> conv3x3( 64) / ReLU          16x16
      -> conv3x3(128, stride 2) / ReLU  8x8
      -> conv3x3(128) / ReLU           8x8
      -> conv3x3(128, stride 2)        4x4   (no ReLU, as in HardNet)
      -> GlobalAveragePool -> Flatten  (1,128)
      -> Gemm(128 -> 512)
    desc (1,512) float32

BatchNorm is deliberately omitted: with untrained random weights BN adds
nothing, and keeping the graph to Conv/ReLU/Gemm maximizes compatibility with
the OpenVINO myriad compiler used for RVC2. When trained HardNet-compatible
weights are provided via ``--weights``, BN is already folded into the conv
weights by the training-side export (this script only accepts plain conv
weights whose shapes match the stack above).

``--weights PATH`` loads a PyTorch HardNet ``.pth`` checkpoint, but only when
``torch`` is importable; torch is an optional extra for this tool and is NOT a
runtime requirement of the example. Without usable weights the network is
initialized with a seeded He/random init and a loud warning is printed: the
resulting descriptors are meaningless for matching and only suitable for
pipeline bring-up (shapes, bandwidth, compression-head validation).

The raw 512-d float output ``desc`` doubles as the 512-bit binary descriptor:
its sign pattern is the bit vector consumed by ``core.projection``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = EXAMPLE_ROOT / "depthai_models" / "descriptor_base.onnx"

# (out_channels, in_channels, stride, relu) per conv, in graph order.
CONV_STACK = [
    (32, 1, 1, True),
    (32, 32, 1, True),
    (64, 32, 2, True),
    (64, 64, 1, True),
    (128, 64, 2, True),
    (128, 128, 1, True),
    (128, 128, 2, False),
]
GEMM_IN = 128
DESC_DIM = 512

# Half-width stack: ~4x fewer MACs for throughput-oriented bring-up.
SLIM_CONV_STACK = [(max(cout // 2, 16), max(cin // 2, 1), s, r) for cout, cin, s, r in CONV_STACK]
SLIM_GEMM_IN = SLIM_CONV_STACK[-1][0]

# Opset 11 / IR v7 keeps the graph loadable by the older OpenVINO myriad
# compiler (2021.4/2022.1) used by blobconverter for RVC2 targets.
OPSET = 11
IR_VERSION = 7


def _he_init(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int) -> np.ndarray:
    """He normal init for conv weights feeding ReLUs."""
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def _random_weights(seed: int) -> dict[str, np.ndarray]:
    """Seeded random init for the whole stack (bring-up only)."""
    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {}
    for i, (cout, cin, _, _) in enumerate(CONV_STACK):
        weights[f"conv{i}_w"] = _he_init(rng, (cout, cin, 3, 3), cin * 3 * 3)
        weights[f"conv{i}_b"] = np.zeros(cout, dtype=np.float32)
    # Final linear layer has no ReLU after it; use a Xavier-ish scale.
    weights["gemm_w"] = (rng.standard_normal((DESC_DIM, GEMM_IN)) / np.sqrt(GEMM_IN)).astype(np.float32)
    weights["gemm_b"] = np.zeros(DESC_DIM, dtype=np.float32)
    return weights


def _load_torch_weights(path: str) -> dict[str, np.ndarray] | None:
    """Try to load a HardNet .pth checkpoint; return None if not possible.

    torch is an optional dependency of this tool. Any failure (torch missing,
    bad file, unexpected keys/shapes) returns None so the caller can fall back
    to random init with a warning.
    """
    try:
        import torch  # noqa: F401  (guarded optional import)
    except ImportError:
        print(
            "WARNING: --weights was given but torch is not installed; "
            "cannot load the checkpoint. Falling back to random init.",
            file=sys.stderr,
        )
        return None

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - report any load failure loudly
        print(f"WARNING: failed to load checkpoint {path}: {exc}", file=sys.stderr)
        return None
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    state = {k: v.cpu().numpy() for k, v in state.items() if hasattr(v, "cpu")}

    # Original HardNet checkpoints store convs under features.{0,3,6,9,12,15,18}
    # (Conv2d at every third index, BatchNorm in between) and the final linear
    # under "fc". Accept both that layout and plain conv0..6/gemm naming.
    conv_key_sets = [
        [f"features.{3 * i}.weight" for i in range(len(CONV_STACK))],
        [f"conv{i}.weight" for i in range(len(CONV_STACK))],
        [f"conv{i}_w" for i in range(len(CONV_STACK))],
    ]
    bias_key_sets = [
        [f"features.{3 * i}.bias" for i in range(len(CONV_STACK))],
        [f"conv{i}.bias" for i in range(len(CONV_STACK))],
        [f"conv{i}_b" for i in range(len(CONV_STACK))],
    ]
    fc_sets = [("fc.weight", "fc.bias"), ("gemm.weight", "gemm.bias"), ("gemm_w", "gemm_b")]

    for conv_keys, bias_keys in zip(conv_key_sets, bias_key_sets):
        if all(k in state for k in conv_keys):
            break
    else:
        print(
            f"WARNING: checkpoint {path} has no recognizable HardNet conv keys; "
            "falling back to random init.",
            file=sys.stderr,
        )
        return None

    weights: dict[str, np.ndarray] = {}
    for i, (cout, cin, _, _) in enumerate(CONV_STACK):
        w = state[conv_keys[i]]
        if w.shape != (cout, cin, 3, 3):
            print(
                f"WARNING: checkpoint conv shape {tuple(w.shape)} != {(cout, cin, 3, 3)}; "
                "falling back to random init.",
                file=sys.stderr,
            )
            return None
        weights[f"conv{i}_w"] = w.astype(np.float32)
        weights[f"conv{i}_b"] = (
            state[bias_keys[i]].astype(np.float32) if bias_keys[i] in state else np.zeros(cout, np.float32)
        )

    for fc_w_key, fc_b_key in fc_sets:
        if fc_w_key in state:
            fc_w = state[fc_w_key].astype(np.float32)
            if fc_w.shape != (DESC_DIM, GEMM_IN):
                print(
                    f"WARNING: checkpoint fc shape {tuple(fc_w.shape)} != {(DESC_DIM, GEMM_IN)}; "
                    "falling back to random init.",
                    file=sys.stderr,
                )
                return None
            weights["gemm_w"] = fc_w
            weights["gemm_b"] = (
                state[fc_b_key].astype(np.float32) if fc_b_key in state else np.zeros(DESC_DIM, np.float32)
            )
            return weights
    print(f"WARNING: checkpoint {path} has no fc/gemm weight; falling back to random init.", file=sys.stderr)
    return None


def build_model(weights: dict[str, np.ndarray], batch: int = 1, strip: int = 1,
                color: bool = False) -> onnx.ModelProto:
    """Assemble the ONNX graph described in the module docstring.

    With ``strip > 1`` the graph input is a single (1, 1, 32*strip, 32) frame:
    ``strip`` patches stacked vertically. The conv stack slides over the whole
    strip (one inference), then a Transpose/Reshape/Transpose re-exposes the
    per-patch row blocks as a leading dim before global average pooling, so
    the Gemm emits one 512-d descriptor per patch. This keeps the graph
    boundary batch-1 (Myriad X blobs collapse a true batch dim > 1 to a
    single patch) while amortizing the ~5 ms per-inference overhead over
    ``strip`` patches. Adjacent patches bleed slightly through conv receptive
    fields at block boundaries (acceptable for bring-up; retrain in strip
    mode for production quality).
    """
    initializers = [numpy_helper.from_array(v, name=k) for k, v in weights.items()]
    # In-graph normalization: the Myriad X blob is compiled with a U8 input
    # (GRAY8 crops are 1 byte/px; an FP16 input would need 2 bytes/px and the
    # NN node rejects the undersized frame). The device feeds raw 0-255
    # grayscale, the input-precision boundary converts U8->float, and this Div
    # maps it to the [0,1] range the conv stack expects.
    nodes = []
    prev = "input"
    if color:
        # Opponent-color variant: the strip holds the 3 opponent planes of
        # every patch folded into the height axis, channel-major
        # ([O1 all patches; O2; O3]; see core/color.py). Re-expose the
        # channel axis, then map the zero-centered-at-127.5 storage to the
        # [-1, 1] range the network was trained on.
        initializers.append(
            numpy_helper.from_array(
                np.array([1, 3, 32 * strip, 32], dtype=np.int64), name="color_shape"
            )
        )
        nodes.append(
            helper.make_node("Reshape", inputs=["input", "color_shape"],
                             outputs=["input_3ch"], name="color_reshape")
        )
        initializers.append(
            numpy_helper.from_array(np.array(127.5, dtype=np.float32), name="norm_sub")
        )
        initializers.append(
            numpy_helper.from_array(np.array(127.5, dtype=np.float32), name="norm_div_c")
        )
        nodes.append(
            helper.make_node("Sub", inputs=["input_3ch", "norm_sub"],
                             outputs=["input_sub"], name="normalize_sub")
        )
        nodes.append(
            helper.make_node("Div", inputs=["input_sub", "norm_div_c"],
                             outputs=["input_norm"], name="normalize_div")
        )
        prev = "input_norm"
    else:
        initializers.append(numpy_helper.from_array(np.array(255.0, dtype=np.float32), name="norm_div"))
        nodes = [
            helper.make_node("Div", inputs=["input", "norm_div"], outputs=["input_norm"], name="normalize")
        ]
        prev = "input_norm"
    for i, (cout, _, stride, relu) in enumerate(CONV_STACK):
        conv_out = f"conv{i}_out"
        nodes.append(
            helper.make_node(
                "Conv",
                inputs=[prev, f"conv{i}_w", f"conv{i}_b"],
                outputs=[conv_out],
                name=f"conv{i}",
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[stride, stride],
            )
        )
        prev = conv_out
        if relu:
            nodes.append(helper.make_node("Relu", inputs=[prev], outputs=[f"relu{i}_out"], name=f"relu{i}"))
            prev = f"relu{i}_out"

    if strip > 1:
        # Conv output is (1, C, 4*strip, 4). Fold the strip axis back out:
        # transpose to (1, 4*strip, 4, C), reshape to (strip, 4, 4, C) (row
        # block n of the strip maps to index n since 32 px / 8 stride = 4
        # rows), then transpose to (strip, C, 4, 4) for per-patch pooling.
        nodes.append(
            helper.make_node("Transpose", inputs=[prev], outputs=["tr1_out"],
                             name="strip_transpose", perm=[0, 2, 3, 1])
        )
        initializers.append(
            numpy_helper.from_array(np.array([strip, 4, 4, GEMM_IN], dtype=np.int64), name="strip_shape")
        )
        nodes.append(
            helper.make_node("Reshape", inputs=["tr1_out", "strip_shape"],
                             outputs=["rs_out"], name="strip_reshape")
        )
        nodes.append(
            helper.make_node("Transpose", inputs=["rs_out"], outputs=["tr2_out"],
                             name="strip_transpose2", perm=[0, 3, 1, 2])
        )
        prev = "tr2_out"

    nodes.append(helper.make_node("GlobalAveragePool", inputs=[prev], outputs=["gap_out"], name="gap"))
    nodes.append(helper.make_node("Flatten", inputs=["gap_out"], outputs=["flat_out"], name="flatten", axis=1))
    nodes.append(
        helper.make_node(
            "Gemm",
            inputs=["flat_out", "gemm_w", "gemm_b"],
            outputs=["desc"],
            name="gemm",
            transB=1,  # gemm_w is (512, 128); transB makes it (1,128) @ (128,512)
        )
    )

    graph = helper.make_graph(
        nodes,
        "descriptor_base",
        inputs=[
            helper.make_tensor_value_info(
                "input", onnx.TensorProto.FLOAT,
                [batch, 1, 32 * strip * (3 if color else 1), 32],
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "desc", onnx.TensorProto.FLOAT, [strip if strip > 1 else batch, DESC_DIM]
            ),
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="oak-examples binary-descriptor",
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.ir_version = IR_VERSION
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional PyTorch HardNet .pth checkpoint (requires torch to be importable).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the random fallback initialization (default: 0).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size of the model input (default: 1). NOTE: Myriad X blobs "
        "do not support batch > 1 (the compiler silently collapses outputs to "
        "batch 1), so keep this at 1 for on-device runs.",
    )
    parser.add_argument(
        "--slim",
        action="store_true",
        help="Use the half-width conv stack (~4x fewer MACs) for higher "
        "inference throughput on the NCE.",
    )
    parser.add_argument(
        "--strip",
        type=int,
        default=1,
        help="Stack N 32x32 patches into one (1,1,32*N,32) strip input so one "
        "inference describes N patches (default: 1). Requires --batch 1; the "
        "strip axis is folded out inside the graph (Myriad X collapses a real "
        "batch dim > 1).",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Opponent-color variant: input is a (1,1,96*strip,32) strip of "
        "(o1,o2,o3) planes (channel-major, zero at 127.5; see core/color.py); "
        "conv0 takes 3 input channels. Requires weights trained with "
        "train_descriptor.py --color.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output ONNX path (default: {DEFAULT_OUT}).",
    )
    args = parser.parse_args()

    if args.slim:
        global CONV_STACK, GEMM_IN
        CONV_STACK = SLIM_CONV_STACK
        GEMM_IN = SLIM_GEMM_IN
    if args.color:
        CONV_STACK[0] = (CONV_STACK[0][0], 3, CONV_STACK[0][2], CONV_STACK[0][3])

    if args.strip > 1 and args.batch != 1:
        parser.error("--strip requires --batch 1 (the strip axis replaces the batch dim).")

    if args.weights:
        weights = _load_torch_weights(args.weights)
        if weights is None:
            weights = _random_weights(args.seed)
    else:
        print(
            "WARNING: no --weights given; using seeded random initialization. "
            "The exported descriptor is UNTRAINED and only suitable for "
            "pipeline bring-up, not for real feature matching.",
            file=sys.stderr,
        )
        weights = _random_weights(args.seed)

    model = build_model(weights, batch=args.batch, strip=args.strip, color=args.color)
    onnx.checker.check_model(model, full_check=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.out)
    print(f"Wrote {args.out} (opset {OPSET}, ir_version {IR_VERSION}); onnx.checker passed.")


if __name__ == "__main__":
    main()
