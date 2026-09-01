"""Compile the injected descriptor ONNX to a Myriad X (RVC2) blob and package it
as a DepthAI v3 NNArchive (.tar.xz).

Pipeline position::

    export_descriptor_onnx.py  ->  descriptor_base.onnx
    make_projection.py --inject ->  descriptor64_<strategy>.onnx
    compile_blob.py             ->  descriptor64_<strategy>.tar.xz   (this tool)

Compilation goes through the Luxonis blobconverter cloud service
(https://blobconverter.luxonis.com), so this tool needs network access; it
fails with a clear message when the service or the network is unavailable.

NNArchive packaging
-------------------
A DepthAI v3 NNArchive is a ``.tar.xz`` containing ``config.json`` plus the
model binary. The ``config.json`` schema (``config_version: 1.0``) was
verified against depthai 3.8 (``dai.nn_archive.v1``) and a real Model Zoo
RVC2 archive; for a raw-output model with no parser, ``heads`` is simply null
and ``ParsingNeuralNetwork`` exposes the raw tensors. The ``input_type`` is
``raw`` (not ``image``): the pipeline feeds pre-normalized 32x32 grayscale
patch crops, so no color conversion or scaling preprocessing is requested.

blobconverter specifics
-----------------------
* ``compile_params=[]`` overrides blobconverter's default ``-ip U8`` flag; the
  graph expects a float32 [0,1] patch, not U8.
* ``optimizer_params=[]`` drops the default mean/scale values ([127.5]/[255])
  which would double-normalize our already-normalized input.
* ``data_type="FP16"`` compiles weights to FP16 as usual for myriad.
* The default OpenVINO version 2022.1 matches the depthai 3.x RVC2 runtime.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BATCH = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--onnx", type=Path, required=True, help="Injected ONNX graph (desc + code outputs).")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Compression strategy name; used for archive naming (e.g. subsample, xorfold, lsh, itq).",
    )
    parser.add_argument("--shaves", type=int, default=4, help="Myriad X SHAVE count (default: 4).")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output NNArchive path (default: depthai_models/descriptor64_<strategy>.tar.xz).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="2022.1",
        help="OpenVINO version for the myriad compiler (default: 2022.1).",
    )
    parser.add_argument(
        "--data_type",
        choices=["FP16", "FP32"],
        default="FP16",
        help="Weight precision for the compiled blob (default: FP16).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size baked into the blob input shape (default: 1). NOTE: "
        "Myriad X collapses batch > 1 to a single patch; use --strip instead.",
    )
    parser.add_argument(
        "--strip",
        type=int,
        default=1,
        help="Strip height multiplier: input is one (1,1,32*N,32) frame holding "
        "N stacked patches; outputs are (N,512)/(N,64). Must match the ONNX "
        "built with export_descriptor_onnx.py --strip N.",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Opponent-color variant: strip height is 96*N (three opponent "
        "planes per patch). Must match export_descriptor_onnx.py --color.",
    )
    parser.add_argument(
        "--code_only",
        action="store_true",
        help="The ONNX exposes only the 'code' output (make_projection.py "
        "--code_only); the archive config lists just that output.",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Bypass the blobconverter client-side cache (recompile even if a "
        "cached blob matches).",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for the blobconverter service. "
        "Only use when the service's certificate is known-expired; the endpoint "
        "is still the official Luxonis one, but traffic is not authenticated.",
    )
    return parser.parse_args()


def _disable_tls_verification() -> None:
    """Force requests (used by blobconverter) to skip TLS verification."""
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    import requests

    original = requests.sessions.Session.request

    def patched(self, method, url, **kwargs):  # noqa: ANN001, ANN202
        kwargs["verify"] = False
        return original(self, method, url, **kwargs)

    requests.sessions.Session.request = patched
    print("WARNING: TLS certificate verification disabled for blobconverter requests.", file=sys.stderr)


def compile_onnx(onnx_path: Path, shaves: int, version: str, data_type: str, use_cache: bool = True) -> Path:
    """Compile ONNX -> myriad blob through the blobconverter cloud service."""
    try:
        import blobconverter
    except ImportError as exc:
        raise SystemExit(
            "blobconverter is not installed. Install it with `pip install blobconverter` "
            "(it is a tool-only dependency, not part of requirements.txt)."
        ) from exc

    # Map "2022.1" -> blobconverter.Versions.v2022_1; fall back to the package default.
    versions = getattr(blobconverter, "Versions", None)
    ov_version = getattr(versions, f"v{version.replace('.', '_')}", None) if versions else None

    kwargs = dict(
        model=str(onnx_path),
        shaves=shaves,
        data_type=data_type,
        # U8 input precision: GRAY8 patch crops are 1 byte/px (1024 B); an FP16
        # input would expect 2048 B and the NN node rejects the undersized
        # frame. The graph normalizes 0-255 -> [0,1] itself (Div at input).
        compile_params=["-ip", "U8"],
        # Drop blobconverter's default mean/scale ([127.5]/[255]) optimizer
        # params; normalization is baked into the graph.
        optimizer_params=[],
        use_cache=use_cache,
    )
    if ov_version is not None:
        kwargs["version"] = ov_version

    try:
        blob_path = blobconverter.from_onnx(**kwargs)
    except Exception as exc:  # noqa: BLE001 - blobconverter raises bare requests errors
        raise SystemExit(
            f"\nERROR: blob compilation failed: {exc}\n"
            "blobconverter compiles through https://blobconverter.luxonis.com; "
            "check network access and that the service is reachable, then retry."
        ) from exc
    return Path(blob_path)


def build_config(model_name: str, blob_filename: str, data_type: str, strategy: str,
                 batch: int = 1, strip: int = 1, color: bool = False,
                 code_only: bool = False) -> dict:
    """NNArchive v1 config.json for the raw descriptor model.

    Field spellings verified against the depthai 3.8 schema validator
    (``dai.NNArchive`` raises "Input JSON does not conform to schema!"
    otherwise): ``precision`` must be lowercase ("float16"/"float32"),
    ``input_type`` is lowercase ("raw"/"image"), and ``heads: null`` is
    accepted for raw-output models.

    ``code_only`` matches make_projection.py --code_only: the graph exposes
    only the 64-bit codes (the float desc stays an in-graph intermediate).
    """
    precision = {"FP16": "float16", "FP32": "float32"}[data_type]
    outputs = []
    if not code_only:
        outputs.append(
            {
                "name": "desc",
                "dtype": "float32",
                "shape": [strip if strip > 1 else batch, 512],
                "layout": None,
            }
        )
    outputs.append(
        {
            "name": "code",
            "dtype": "uint8",
            "shape": [strip if strip > 1 else batch, 64],
            "layout": None,
        }
    )
    return {
        "config_version": "1.0",
        "model": {
            "metadata": {
                "name": model_name,
                "path": blob_filename,
                "precision": precision,
            },
            "inputs": [
                {
                    "name": "input",
                    # U8: matches the 1-byte GRAY8 crops; the graph normalizes
                    # to [0,1] internally.
                    "dtype": "uint8",
                    # "raw": patches arrive pre-cropped; no image preprocessing
                    # (color conversion, scaling) is requested.
                    "input_type": "raw",
                    "shape": [batch, 1, 32 * strip * (3 if color else 1), 32],
                    "layout": "NCHW",
                    "preprocessing": {
                        "mean": [0.0],
                        "scale": [1.0],
                        "reverse_channels": False,
                        "interleaved_to_planar": False,
                        "dai_type": None,
                    },
                }
            ],
            "outputs": outputs,
            # No parser heads: ParsingNeuralNetwork exposes raw tensors.
            "heads": None,
        },
    }


def package_archive(blob_path: Path, config: dict, out_path: Path) -> None:
    """Write the .tar.xz NNArchive: config.json + model blob (+ buildinfo.json)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buildinfo = {
        "comment": "Compiled with blobconverter cloud service by tools/compile_blob.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with tarfile.open(out_path, "w:xz") as tf:
        blob_name = config["model"]["metadata"]["path"]
        tf.add(blob_path, arcname=blob_name)
        for name, payload in [("config.json", config), ("buildinfo.json", buildinfo)]:
            raw = json.dumps(payload, indent=2).encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(raw)
            tf.addfile(info, io.BytesIO(raw))


def verify_archive(archive_path: Path) -> None:
    """Sanity-check the archive by loading it with depthai itself.

    Note: NNArchive construction parses config.json AND opens the blob, so
    this fully validates the package (a "BlobReader" error here means the
    blobconverter output was truncated or corrupt).
    """
    try:
        import depthai as dai
    except ImportError:
        print("WARNING: depthai not importable; skipping archive verification.", file=sys.stderr)
        return
    archive = dai.NNArchive(str(archive_path))
    cfg = archive.getConfigV1().model
    out_names = [o.name for o in cfg.outputs]
    print(
        f"Verified with depthai {dai.__version__}: model type {archive.getModelType()}, "
        f"input size {archive.getInputSize()}, outputs {out_names}"
    )
    if "code" not in out_names or any(n not in ("desc", "code") for n in out_names):
        raise SystemExit(f"ERROR: unexpected archive outputs {out_names}; expected 'code' (+ optional 'desc')")


def main() -> None:
    args = parse_args()
    suffix = "_color" if args.color else ""
    out_path = args.out or (
        EXAMPLE_ROOT / "depthai_models" / f"descriptor64_{args.strategy}{suffix}.tar.xz"
    )
    model_name = f"descriptor64_{args.strategy}{suffix}"

    if args.insecure:
        _disable_tls_verification()
    blob_path = compile_onnx(args.onnx, args.shaves, args.version, args.data_type, use_cache=not args.no_cache)
    print(f"Compiled blob: {blob_path}")

    config = build_config(model_name, f"{model_name}.blob", args.data_type, args.strategy,
                          batch=args.batch, strip=args.strip, color=args.color,
                          code_only=args.code_only)
    package_archive(blob_path, config, out_path)
    print(f"Wrote NNArchive: {out_path}")

    verify_archive(out_path)


if __name__ == "__main__":
    main()
