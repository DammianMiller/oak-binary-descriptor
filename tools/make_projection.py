"""Build a 512 -> 64 bit compression projection and optionally bake it into ONNX.

Wraps ``core.projection.make_projection`` in a CLI so the (64, 512) float32
matrix W can be saved as ``.npy`` and/or injected into the base descriptor
ONNX graph produced by ``tools/export_descriptor_onnx.py``.

Injection appends three nodes after the ``desc`` output of the base graph::

    code_logits = MatMul(desc (1,512), W^T initializer (512,64))
    code_bool   = Greater(code_logits, 0.0 scalar initializer)
    code        = Cast(code_bool, to=UINT8)

so the compiled model has two outputs:

    desc : (1,512) float32  - raw 512-d descriptor (sign = 512-bit descriptor)
    code : (1,64)  uint8    - packed 64-bit compressed descriptor (zorder)

Both outputs come from the same graph, so one device-side inference yields the
full descriptor and the compressed code together.

Examples
--------
Dump the matrix only::

    python3 tools/make_projection.py --strategy lsh --seed 0 --matrix_out W_lsh.npy

Fit ITQ on calibration descriptors and inject into the base graph::

    python3 tools/make_projection.py --strategy itq --calibration calib.npy \
        --inject depthai_models/descriptor_base.onnx depthai_models/descriptor64_itq.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

# The example root holds the `core` package (core/projection.py).
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))

from core.projection import IN_BITS, OUT_BITS, STRATEGIES, make_projection  # noqa: E402


def inject_head(base_path: Path, out_path: Path, W: np.ndarray,
                code_only: bool = False) -> None:
    """Append MatMul -> Greater -> Cast to UINT8 after ``desc`` in an ONNX graph.

    With ``code_only`` the ``desc`` tensor is dropped from the graph outputs
    (it stays an intermediate feeding the head): the compiled blob then
    streams only the 64-bit codes, ~2 KB instead of ~66 KB per 32-patch
    strip. On a USB2 host link that lifts the keypoints/frame ceiling
    substantially (measured: color 128 kps/frame goes from ~14 to ~20 fps).
    ``FeaturePipeline`` auto-detects such archives and skips desc decoding.
    """
    model = onnx.load(str(base_path))
    graph = model.graph

    desc_outputs = [o for o in graph.output if o.name == "desc"]
    if not desc_outputs:
        raise ValueError(f"{base_path} has no output named 'desc' to attach the head to")
    if any(o.name == "code" for o in graph.output):
        raise ValueError(f"{base_path} already has a 'code' output; inject into the base graph instead")

    # W is (out_bits, in_bits); MatMul needs (in_bits, out_bits) for desc @ W^T.
    Wt = np.ascontiguousarray(W.T, dtype=np.float32)
    if Wt.shape != (IN_BITS, OUT_BITS):
        raise ValueError(f"projection matrix has shape {W.T.shape}, expected ({IN_BITS}, {OUT_BITS})")
    graph.initializer.append(numpy_helper.from_array(Wt, name="code_projection"))
    graph.initializer.append(numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="code_threshold"))

    graph.node.extend(
        [
            onnx.helper.make_node(
                "MatMul", inputs=["desc", "code_projection"], outputs=["code_logits"], name="code_matmul"
            ),
            onnx.helper.make_node(
                "Greater", inputs=["code_logits", "code_threshold"], outputs=["code_bool"], name="code_greater"
            ),
            onnx.helper.make_node(
                "Cast",
                inputs=["code_bool"],
                outputs=["code"],
                name="code_cast",
                to=onnx.TensorProto.UINT8,
            ),
        ]
    )
    # Match the base graph's batch dimension (desc is (B, IN_BITS)).
    batch_dim = graph.output[0].type.tensor_type.shape.dim[0].dim_value
    graph.output.append(
        onnx.helper.make_tensor_value_info("code", onnx.TensorProto.UINT8, [batch_dim, OUT_BITS])
    )
    if code_only:
        # Keep desc as an intermediate (it feeds the head) but stop streaming
        # it to the host: 64 KB less payload per strip of 32 patches.
        keep = [o for o in graph.output if o.name != "desc"]
        del graph.output[:]
        graph.output.extend(keep)

    onnx.checker.check_model(model, full_check=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    print(f"Injected 512->{OUT_BITS} compression head; wrote {out_path} (onnx.checker passed).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        required=True,
        help="Compression strategy (see core/projection.py docstring).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (used by lsh and itq; default: 0).")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Path to an (n, 512) float .npy of raw descriptors. Required for --strategy itq.",
    )
    parser.add_argument(
        "--matrix_out",
        type=Path,
        default=None,
        help="Optional path to save the (64, 512) float32 projection matrix as .npy.",
    )
    parser.add_argument(
        "--inject",
        nargs=2,
        metavar=("BASE_ONNX", "OUT_ONNX"),
        default=None,
        help="Inject the compression head into BASE_ONNX and write OUT_ONNX.",
    )
    parser.add_argument(
        "--code_only",
        action="store_true",
        help="With --inject: drop the float 'desc' graph output so the blob "
        "streams only the 64-bit codes (~2 KB instead of ~66 KB per 32-patch "
        "strip; lifts the USB2 keypoints/frame ceiling).",
    )
    args = parser.parse_args()

    if args.matrix_out is None and args.inject is None:
        parser.error("nothing to do: pass --matrix_out and/or --inject")

    calibration = None
    if args.strategy == "itq":
        if args.calibration is None:
            parser.error("--strategy itq requires --calibration PATH.npy")
        calibration = np.load(args.calibration)
        print(f"Loaded calibration descriptors {calibration.shape} from {args.calibration}")

    W = make_projection(args.strategy, seed=args.seed, calibration=calibration)
    assert W.shape == (OUT_BITS, IN_BITS) and W.dtype == np.float32

    if args.matrix_out is not None:
        args.matrix_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.matrix_out, W)
        print(f"Wrote projection matrix {W.shape} ({W.dtype}) to {args.matrix_out}")

    if args.inject is not None:
        inject_head(Path(args.inject[0]), Path(args.inject[1]), W,
                    code_only=args.code_only)


if __name__ == "__main__":
    main()
