"""Host-side benchmark: does 512 -> 64 bit compression preserve matching?

Runs entirely on the host (no OAK device, no onnxruntime needed):

1. Generate a synthetic textured image (multi-scale Gaussian noise + blur).
2. Detect cv2.BRISK keypoints and compute 512-bit (64-byte) descriptors.
3. Apply 6 synthetic warps (small rotation / scale / perspective jitter) and
   recompute descriptors on each warped image.
4. Ground truth: nearest-neighbor matching on the FULL 512-bit descriptors
   (Hamming over all 512 bits, distance <= GT_MAX_DISTANCE).
5. For each compression strategy (subsample / xorfold / lsh / itq), treat the
   512 descriptor bits as 0/1 floats (this stands in for the raw float
   descriptor output, whose sign is the bit vector), project with
   ``core.projection.make_projection`` + ``apply_projection``, pack to uint64
   with ``pack_bits``, and match with ``core.packer.match_codes``.
6. Print a precision/recall table per strategy and per 64-bit distance
   threshold against the 512-bit ground truth.

This is the pre-device evidence that the compression head preserves enough
matching quality to be worth compiling. ITQ is fit on the base-image
descriptors only (that is the role calibration descriptors play on device).

Exit code is always 0 unless something crashes; degenerate inputs (too few
keypoints/descriptors) are reported as warnings instead of failures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# The example root holds the `core` package (core/projection.py, core/packer.py).
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))

from core.packer import match_codes  # noqa: E402
from core.projection import apply_projection, make_projection, pack_bits  # noqa: E402

IMAGE_SIZE = 512
N_WARPS = 6
# Ground-truth acceptance threshold on the full 512-bit BRISK Hamming distance.
# BRISK matched pairs on mildly warped images typically sit well below 100/512.
GT_MAX_DISTANCE = 100
# 64-bit acceptance thresholds swept for the compressed codes.
CODE_THRESHOLDS = [6, 10, 14, 18]
STRATEGIES = ("subsample", "xorfold", "lsh", "itq")
MIN_DESCRIPTORS = 50


def synthetic_texture(size: int, seed: int = 0) -> np.ndarray:
    """Multi-scale random noise + blur: plenty of gradients for BRISK to bite on."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float64)
    for octaves, (scale, weight) in enumerate([(4, 1.0), (8, 0.7), (16, 0.5), (32, 0.3), (64, 0.2)]):
        coarse = rng.standard_normal((scale, scale))
        img += weight * cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


def detect(brisk: cv2.BRISK, img: np.ndarray) -> tuple[list, np.ndarray | None]:
    kp, desc = brisk.detectAndCompute(img, None)
    if desc is None or len(desc) == 0:
        return [], None
    return kp, desc


def random_warp(rng: np.random.Generator, size: int) -> np.ndarray:
    """Small random similarity transform plus perspective jitter."""
    angle = rng.uniform(-10.0, 10.0)
    scale = rng.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
    M[:, 2] += rng.uniform(-8.0, 8.0, size=2)
    # Add mild perspective jitter on the 4 corners.
    corners = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    jitter = rng.uniform(-6.0, 6.0, size=(4, 2)).astype(np.float32)
    P = cv2.getPerspectiveTransform(corners, corners + jitter)
    H = P @ np.vstack([M, [0, 0, 1]])
    return H


def full_hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(m,) Hamming distances between one 512-bit descriptor and n candidates.

    ``a``: (64,) uint8, ``b``: (n, 64) uint8 packed BRISK descriptors.
    """
    return np.unpackbits(np.bitwise_xor(a[None, :], b), axis=1).sum(axis=1)


def ground_truth_matches(desc_a: np.ndarray, desc_b: np.ndarray) -> set[tuple[int, int]]:
    """Nearest-neighbor matches on full 512-bit descriptors, within threshold."""
    matches = set()
    for i in range(len(desc_a)):
        dists = full_hamming(desc_a[i], desc_b)
        j = int(np.argmin(dists))
        if int(dists[j]) <= GT_MAX_DISTANCE:
            matches.add((i, j))
    return matches


def compress_codes(W: np.ndarray, desc: np.ndarray) -> np.ndarray:
    """512 packed BRISK bits -> 0/1 floats -> projection -> uint64 codes."""
    bits = np.unpackbits(desc, axis=1).astype(np.float32)  # (n, 512), 0/1
    return pack_bits(apply_projection(W, bits))


def evaluate(gt: set[tuple[int, int]], pred: list[tuple[int, int, int]]) -> tuple[float, float, int]:
    pred_set = {(q, r) for q, r, _ in pred}
    tp = len(gt & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt) if gt else 0.0
    return precision, recall, tp


def main() -> int:
    rng = np.random.default_rng(42)
    brisk = cv2.BRISK_create(thresh=20)

    base_img = synthetic_texture(IMAGE_SIZE)
    _, desc_base = detect(brisk, base_img)
    if desc_base is None or len(desc_base) < MIN_DESCRIPTORS:
        print(
            f"WARNING: only {0 if desc_base is None else len(desc_base)} base descriptors "
            f"(need >= {MIN_DESCRIPTORS}); synthetic texture degenerate, skipping benchmark."
        )
        return 0
    print(f"Base image: {len(desc_base)} BRISK keypoints/descriptors (512-bit each)")
    print(f"Ground truth: full 512-bit nearest-neighbor matching, distance <= {GT_MAX_DISTANCE}")
    print(f"Warps: {N_WARPS} random rotation/scale/perspective transforms\n")

    warped: list[tuple[np.ndarray, set[tuple[int, int]]]] = []
    for i in range(N_WARPS):
        H = random_warp(rng, IMAGE_SIZE)
        img_w = cv2.warpPerspective(base_img, H, (IMAGE_SIZE, IMAGE_SIZE), borderMode=cv2.BORDER_REFLECT)
        _, desc_w = detect(brisk, img_w)
        if desc_w is None or len(desc_w) == 0:
            print(f"WARNING: warp {i} produced no descriptors; skipping it")
            continue
        warped.append((desc_w, ground_truth_matches(desc_base, desc_w)))
    print(f"Ground-truth matches per warp: {[len(g) for _, g in warped]}\n")

    # Fit projections once on the base-image descriptors. The 0/1 bit floats
    # stand in for the raw 512-d float descriptor (sign == bit).
    base_bits = np.unpackbits(desc_base, axis=1).astype(np.float32)
    projections = {
        s: make_projection(s, seed=0, calibration=base_bits if s == "itq" else None) for s in STRATEGIES
    }

    # Aggregate over warps, per strategy and per threshold.
    header = f"{'strategy':<10} {'thresh':>6} {'precision':>10} {'recall':>8} {'tp':>6} {'pred':>6} {'gt':>6}"
    print(header)
    print("-" * len(header))
    for strategy in STRATEGIES:
        W = projections[strategy]
        codes_base = compress_codes(W, desc_base)
        codes_warped = [compress_codes(W, d) for d, _ in warped]
        for t in CODE_THRESHOLDS:
            tp_total = pred_total = gt_total = 0
            for warped_codes, (_, gt) in zip(codes_warped, warped):
                pred = match_codes(codes_base, warped_codes, max_distance=t)
                _, _, tp = evaluate(gt, pred)
                tp_total += tp
                pred_total += len(pred)
                gt_total += len(gt)
            precision = tp_total / pred_total if pred_total else 0.0
            recall = tp_total / gt_total if gt_total else 0.0
            print(f"{strategy:<10} {t:>6} {precision:>10.3f} {recall:>8.3f} {tp_total:>6} {pred_total:>6} {gt_total:>6}")
        print()

    print("Note: this uses packed BRISK bits as a stand-in for the network's raw")
    print("512-d float output (sign == bit). With trained HardNet weights the raw")
    print("descriptor is matched to begin with, so compressed-code quality only")
    print("improves. This table validates the compression head, not the CNN.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
