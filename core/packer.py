"""ROS-agnostic packing of keypoints + on-device descriptor codes.

Produces plain dicts mirroring the oak_features_msgs/Keypoint fields:

    x          : float, pixel coordinate (always present)
    y          : float, pixel coordinate (always present)
    track_id   : int, FeatureTracker ID (stable across consecutive frames)
    age        : int, tracking age in frames
    desc       : bytes, optional full 512-bit descriptor (64 bytes), empty if absent
    has_zorder : bool, True when a 64-bit compressed descriptor code is present
    zorder     : int, 64-bit compressed descriptor code (Morton/LSH/ITQ/... head)

The 64-bit zorder is the compressed *descriptor* code (not a spatial code):
it is designed to stay stable as the keypoint moves across pixels, enabling
cheap temporal re-identification with XOR+popcount (see hamming64 in
core.projection).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FULL_DESC_BYTES = 64  # 512 bits


@dataclass
class KeypointRecord:
    """One keypoint as delivered by the pipeline host stage.

    ``has_code``/``code_age`` are written by DescribeGate mode: has_code is
    False for never-described keypoints (codes are then parallel to the
    keypoints list, not a prefix) and code_age counts frames since the code
    was actually described (0 = fresh this frame). Both keep their defaults
    (True / 0) in legacy prefix mode, so old consumers are unaffected.
    """

    x: float
    y: float
    track_id: int
    age: int = 0
    has_code: bool = True
    code_age: int = 0


def pack_frame(
    keypoints: list[KeypointRecord],
    codes: np.ndarray | None = None,
    full_descs: np.ndarray | None = None,
    sort_by_code: bool = True,
) -> list[dict]:
    """Pack one frame of keypoints into message-shaped dicts.

    Args:
        keypoints: per-keypoint records from FeatureTracker.
        codes: optional (m,) uint64 compressed descriptor codes, aligned with
            the FIRST m ``keypoints`` (oldest tracks first — FeaturePipeline
            with describe_budget < max_keypoints returns m < n; keypoints
            beyond m are tracking-only and get has_zorder=False). In
            DescribeGate mode codes are PARALLEL (m == n) and each record's
            has_code flag decides validity instead.
        full_descs: optional (m, FULL_DESC_BYTES) uint8 full descriptors,
            same alignment as ``codes``.
        sort_by_code: sort output ascending by zorder (keypoints without a
            code go last, ordered by x) so subscribers get a spatially and
            code-locality-friendly scan order.

    Returns:
        List of dicts with the Keypoint message fields.
    """
    n = len(keypoints)
    if codes is not None and len(codes) > n:
        raise ValueError("codes must be a prefix-aligned subset of keypoints")
    if full_descs is not None and full_descs.ndim == 2 and full_descs.shape[0] > n:
        raise ValueError("full_descs must be a prefix-aligned subset of keypoints")
    if full_descs is not None and full_descs.ndim == 2 and full_descs.shape[1] != FULL_DESC_BYTES:
        raise ValueError(f"full_descs must be (m, {FULL_DESC_BYTES}) uint8")

    rows = []
    for i, kp in enumerate(keypoints):
        # kp.has_code is the gate-mode per-record validity flag; it defaults
        # to True, so legacy prefix mode is governed purely by i < len(...).
        has_code = codes is not None and i < len(codes) and kp.has_code
        has_desc = full_descs is not None and i < len(full_descs) and kp.has_code
        rows.append(
            {
                "x": float(kp.x),
                "y": float(kp.y),
                "track_id": int(kp.track_id),
                "age": int(kp.age),
                "desc": bytes(full_descs[i].tobytes()) if has_desc else b"",
                "has_zorder": has_code,
                "zorder": int(codes[i]) if has_code else 0,
            }
        )

    if sort_by_code:
        # Python ints sort fine; np.uint64 wraps on subtraction, so key off int.
        rows.sort(key=lambda r: (not r["has_zorder"], r["zorder"] if r["has_zorder"] else 0, r["x"]))
    return rows


def match_codes(
    query_codes: np.ndarray,
    ref_codes: np.ndarray,
    max_distance: int = 12,
) -> list[tuple[int, int, int]]:
    """Brute-force Hamming matching between two sets of 64-bit codes.

    Args:
        query_codes: (m,) uint64 codes from the current frame.
        ref_codes: (n,) uint64 codes to match against (previous frame, map...).
        max_distance: maximum Hamming distance to accept a match.

    Returns:
        List of (query_index, ref_index, distance), one per query, using the
        nearest ref code within ``max_distance``.
    """
    from core.projection import hamming64

    query_codes = np.atleast_1d(np.asarray(query_codes, dtype=np.uint64))
    ref_codes = np.atleast_1d(np.asarray(ref_codes, dtype=np.uint64))
    matches = []
    for qi, q in enumerate(query_codes):
        dists = hamming64(q, ref_codes)
        ri = int(np.argmin(dists))
        if int(dists[ri]) <= max_distance:
            matches.append((qi, ri, int(dists[ri])))
    return matches
