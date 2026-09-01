#!/usr/bin/env python3
"""Host-side quality check for a trained descriptor checkpoint.

Computes raw 512-d descriptors for held-out synthetic warp pairs with torch,
then compares the four 64-bit compression strategies (from
core/projection.py) on:
  - positive-pair Hamming (same patch, warped): want LOW
  - negative-pair Hamming (different patches): want HIGH (~32 = random)
  - AUC-ish separation score: fraction of (pos < neg) pairs

Usage:
    python3 tools/eval_descriptor.py [--weights depthai_models/descriptor_weights_slim32.pth] \
        [--calibration depthai_models/descriptor_calib.npy]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT / "tools"))

from core.projection import (  # noqa: E402
    STRATEGIES,
    apply_projection,
    hamming64,
    make_projection,
    pack_bits,
)
from train_descriptor import SlimDescriptorNet, make_batch, make_color_batch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path,
                        default=EXAMPLE_ROOT / "depthai_models" / "descriptor_weights_slim32.pth")
    parser.add_argument("--calibration", type=Path,
                        default=EXAMPLE_ROOT / "depthai_models" / "descriptor_calib.npy")
    parser.add_argument("--pairs", type=int, default=2048)
    parser.add_argument("--color", action="store_true",
                        help="Evaluate the opponent-color model (3-channel input).")
    args = parser.parse_args()

    model = SlimDescriptorNet(in_channels=3 if args.color else 1)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    # checkpoint uses conv{i}.weight naming; map onto the ModuleList
    mapped = {}
    for k, v in state.items():
        if k.startswith("conv"):
            i = int(k.split(".")[0][4:])
            suffix = k.split(".")[1]
            mapped[f"convs.{i}.{suffix}"] = v
        else:
            mapped[k] = v
    model.load_state_dict(mapped)
    model.eval()

    rng = np.random.default_rng(5)
    gen = make_color_batch if args.color else make_batch
    with torch.no_grad():
        a, p = gen(rng, args.pairs, strong=True)
        da = model(a).numpy()
        dp = model(p).numpy()
        # negatives: anchors vs fresh patches
        dn = model(gen(rng, args.pairs, strong=False)[0]).numpy()
        iso_da = iso_dp = None
        if args.color:
            # Isoluminant pairs: intensity channel flat, color must carry the
            # match. A grayscale descriptor cannot separate these by
            # construction; the color model's separation here is the uplift.
            ia, ip = make_color_batch(rng, args.pairs, strong=True,
                                      isoluminant_only=True)
            iso_da = model(ia).numpy()
            iso_dp = model(ip).numpy()

    na = F.normalize(torch.from_numpy(da), dim=1).numpy()
    np_ = F.normalize(torch.from_numpy(dp), dim=1).numpy()
    print(f"raw 512-d: positive cos-sim mean={float((na * np_).sum(1).mean()):.3f}")

    # 64-bit codes per strategy: positive pairs (a<->p) want LOW Hamming,
    # negative pairs (a<->dn) want HIGH (~32 = random).
    itq_calib = np.load(args.calibration) if args.calibration.is_file() else None
    print(f"{'strategy':<10} {'pos_mean':>8} {'neg_mean':>8} {'sep':>6}")
    for name in STRATEGIES:
        kwargs = {}
        if name == "itq":
            if itq_calib is None:
                print(f"{name:<10} (skipped: no calibration file)")
                continue
            kwargs["calibration"] = itq_calib
        W = make_projection(name, seed=0, **kwargs)
        ca = pack_bits(apply_projection(W, da))
        cp_ = pack_bits(apply_projection(W, dp))
        cn = pack_bits(apply_projection(W, dn))
        pos = hamming64(ca, cp_)
        neg = hamming64(ca, cn)
        sep = float(np.mean(pos[:, None] < neg[None, :]))
        line = f"{name:<10} {pos.mean():8.1f} {neg.mean():8.1f} {sep:6.3f}"
        if iso_da is not None:
            cia = pack_bits(apply_projection(W, iso_da))
            cip = pack_bits(apply_projection(W, iso_dp))
            iso_pos = hamming64(cia, cip)
            iso_sep = float(np.mean(iso_pos[:, None] < neg[None, :]))
            line += f" | iso_pos {iso_pos.mean():5.1f} iso_sep {iso_sep:6.3f}"
        print(line)


if __name__ == "__main__":
    main()
