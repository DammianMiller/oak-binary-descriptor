"""Change-gated descriptor extraction ("describe only what moved").

The pipeline's throughput ceiling is USB2 strip bandwidth: every described
keypoint costs its share of a 164 KB strip frame each frame, even when the
patch content is identical to last frame's. On mostly-static scenes the vast
majority of describes are redundant. DescribeGate keeps a host-side cache of
(keypoint identity -> code, desc, patch fingerprint) and, each frame,
selects only the keypoints that are NEW, whose patch content CHANGED (mean
abs diff of an 8x8 block-mean fingerprint above ``change_thresh``), or whose
code is STALE (not refreshed for ``refresh_every`` frames). The per-frame
selection is capped at ``budget`` (the pipeline's describe_budget), most
stale first, which yields round-robin fairness under budget pressure.

Identity matching: with tracking enabled, keypoints key on track_id. In
no-track mode there are no ids, so cache entries match spatially (greedy
nearest neighbor within ``match_radius`` px) — fingerprints compare patch
CONTENT, not position, so a keypoint that moved over unchanged texture
correctly reuses its cached code (the descriptor describes the patch, and
the patch is unchanged).

Cost: fingerprints are one vectorized gather + block-mean (sub-ms at 684
keypoints), far cheaper than the strips they save. Worst case (everything
moving) degrades gracefully to describe-everything behavior.

Deliberate false-negative bound: fingerprints can miss small changes; the
``refresh_every`` staleness sweep re-describes every keypoint periodically
regardless, bounding code staleness to ~refresh_every frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

PATCH = 32
SMALL = 8  # fingerprint: SMALL x SMALL block means (per channel)


def patch_fingerprints(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Vectorized 8x8 block-mean fingerprints of 32x32 patches.

    Uses an integral image (one cumsum pass over the frame) instead of a
    per-keypoint patch gather: the gather costs ~12 ms at 256 color keypoints
    and was the measured gate bottleneck; integral + 64 box queries per
    keypoint is ~1 ms and produces the exact same block means.

    Args:
        img: (H, W) uint8 grayscale or (H, W, 3) uint8 BGR frame.
        xy: (N, 2) float keypoint positions (patch centers, clamped to the
            same in-bounds crop window crop_patches/crop_opponent_planes use).

    Returns:
        (N, SMALL*SMALL*C) float32 fingerprints (C = 1 gray, 3 color).
    """
    n = len(xy)
    ch = 3 if img.ndim == 3 else 1
    if n == 0:
        return np.zeros((0, SMALL * SMALL * ch), dtype=np.float32)
    # cv2.integral handles multi-channel directly and is ~17x faster than
    # integral3 in this OpenCV build (0.4 ms vs 7 ms at 640x400x3; results
    # verified identical). The binding may return a tuple of outputs.
    res = cv2.integral(img, sdepth=cv2.CV_64F)
    integ = res[0] if isinstance(res, tuple) else res
    h, w = img.shape[:2]
    half = PATCH // 2
    x0 = np.clip(xy[:, 0].astype(np.int32) - half, 0, w - PATCH)
    y0 = np.clip(xy[:, 1].astype(np.int32) - half, 0, h - PATCH)
    step = PATCH // SMALL
    # (N, SMALL+1) grid of block boundaries within each patch.
    xs = x0[:, None] + np.arange(SMALL + 1)[None, :] * step
    ys = y0[:, None] + np.arange(SMALL + 1)[None, :] * step
    corners = integ[ys[:, :, None], xs[:, None, :]]  # (N, SMALL+1, SMALL+1[, C])
    blocks = (
        corners[:, 1:, 1:] - corners[:, :-1, 1:]
        - corners[:, 1:, :-1] + corners[:, :-1, :-1]
    )  # (N, SMALL, SMALL[, C])
    return (blocks / (step * step)).reshape(n, -1).astype(np.float32)


@dataclass
class _Entry:
    code: int
    desc: bytes | None
    fp: np.ndarray
    x: float
    y: float
    last_seen: int = 0
    last_desc: int = 0


@dataclass
class GatePlan:
    """Per-frame selection result; carried in the pipeline job until drain."""

    frame_idx: int
    sel: list[int]  # indices into the frame's keypoints to describe now
    reuse: dict[int, object] = field(default_factory=dict)  # idx -> cache key
    fps: np.ndarray | None = None  # fingerprints of ALL keypoints this frame
    match_keys: list = field(default_factory=list)  # cache key or None per kp


class DescribeGate:
    """Selects which keypoints need (re-)describing each frame.

    Usage per frame (pipeline internals):
        plan = gate.select(features, frame_img)     # before strip send
        ... crop/describe plan.sel, drain outputs ...
        codes_out, descs_out = gate.apply(plan, features, keypoints,
                                          fresh_codes, fresh_descs)
    ``apply`` updates the cache from the fresh codes and writes parallel
    output arrays aligned with ``keypoints`` (has_code / code_age set on each
    KeypointRecord). Note the pipeline's one-frame overlap: apply() for frame
    k runs during next_frame() call k+1, so select() sees codes through k-2.
    """

    def __init__(self, budget: int, change_thresh: float = 6.0,
                 refresh_every: int = 30, match_radius: float = 6.0):
        self._budget = max(1, int(budget))
        self._thresh = float(change_thresh)
        self._refresh = max(1, int(refresh_every))
        self._radius = float(match_radius)
        self._cache: dict[object, _Entry] = {}
        self._frame = 0
        self._next_spatial_key = 0

    def _match(self, features) -> list:
        """Cache key (or None) per feature.

        Tracking mode: direct track_id lookup. No-track mode (all ids -1):
        greedy nearest-neighbor within match_radius, one entry per keypoint.
        """
        ids = [int(f.id) for f in features]
        if any(i >= 0 for i in ids):
            return [i if i in self._cache else None for i in ids]
        keys: list = [None] * len(features)
        if not self._cache or not features:
            return keys
        entries = list(self._cache.items())
        exy = np.array([[e.x, e.y] for _, e in entries], dtype=np.float32)
        fxy = np.array([[f.position.x, f.position.y] for f in features],
                       dtype=np.float32)
        d = np.linalg.norm(fxy[:, None, :] - exy[None, :, :], axis=2)
        cand = np.argwhere(d <= self._radius)
        order = np.argsort(d[d <= self._radius], kind="stable")
        used_f: set[int] = set()
        used_e: set[int] = set()
        for k in order:
            fi, ei = int(cand[k, 0]), int(cand[k, 1])
            if fi in used_f or ei in used_e:
                continue
            used_f.add(fi)
            used_e.add(ei)
            keys[fi] = entries[ei][0]
        return keys

    def select(self, features, frame_img: np.ndarray) -> GatePlan:
        """Choose describe indices for this frame (does not mutate cache)."""
        self._frame += 1
        n = len(features)
        xy = (
            np.array([[f.position.x, f.position.y] for f in features],
                     dtype=np.float32)
            if n
            else np.zeros((0, 2), np.float32)
        )
        plan = GatePlan(frame_idx=self._frame, sel=[],
                        fps=patch_fingerprints(frame_img, xy))
        if n:
            plan.match_keys = self._match(features)
            eligible = []  # (staleness, idx)
            for i, f in enumerate(features):
                key = plan.match_keys[i]
                ent = self._cache.get(key) if key is not None else None
                if ent is None:
                    eligible.append((self._refresh, i))  # new: max priority
                    continue
                staleness = self._frame - ent.last_desc
                if staleness >= self._refresh:
                    eligible.append((staleness, i))
                elif float(np.abs(plan.fps[i] - ent.fp).mean()) > self._thresh:
                    eligible.append((staleness + 0.5, i))
                else:
                    plan.reuse[i] = key
            # Most stale first -> round-robin fairness under budget pressure.
            eligible.sort(reverse=True)
            plan.sel = [i for _, i in eligible[: self._budget]]
            plan.sel.sort()
        return plan

    def apply(self, plan: GatePlan, features, keypoints,
              fresh_codes: np.ndarray, fresh_descs: np.ndarray | None,
              publish_full_desc: bool):
        """Merge fresh codes into the cache; assemble parallel outputs.

        ``fresh_codes``/``fresh_descs`` align with plan.sel (shorter on a
        dropped NN output — the tail of sel is then treated as not
        described). Returns (codes (n,) uint64, descs (n, 64) uint8 or None),
        and sets has_code/code_age on each KeypointRecord.
        """
        n = len(features)
        n_fresh = min(len(fresh_codes), len(plan.sel))
        assigned: dict[int, object] = {}  # kp idx -> cache key for this frame
        for rank, i in enumerate(plan.sel[:n_fresh]):
            f = features[i]
            key = plan.match_keys[i]
            if key is None or int(f.id) >= 0:
                key = int(f.id) if int(f.id) >= 0 else (
                    key if key is not None else self._new_spatial_key())
            assigned[i] = key
            desc = (
                bytes(fresh_descs[rank].tobytes())
                if fresh_descs is not None and rank < len(fresh_descs)
                else None
            )
            self._cache[key] = _Entry(
                code=int(fresh_codes[rank]), desc=desc, fp=plan.fps[i],
                x=float(f.position.x), y=float(f.position.y),
                last_seen=plan.frame_idx, last_desc=plan.frame_idx,
            )
        # Touch entries that were seen-but-reused so eviction tracks presence.
        for i, key in plan.reuse.items():
            ent = self._cache.get(key)
            if ent is not None:
                ent.last_seen = plan.frame_idx
                ent.x = float(features[i].position.x)
                ent.y = float(features[i].position.y)
        # Evict entries not seen for a refresh period (dead tracks).
        dead = [k for k, e in self._cache.items()
                if plan.frame_idx - e.last_seen > self._refresh]
        for k in dead:
            del self._cache[k]

        codes_out = np.zeros((n,), dtype=np.uint64)
        descs_out = (
            np.zeros((n, 64), dtype=np.uint8) if publish_full_desc else None
        )
        for i in range(n):
            kp = keypoints[i]
            # Freshly described keypoints have no cache match at select time;
            # use the key assigned during this apply for them.
            key = assigned.get(i, plan.match_keys[i])
            ent = self._cache.get(key) if key is not None else None
            # A keypoint that was selected but whose NN output was lost keeps
            # its previous cache entry (or none, if brand new).
            if ent is None:
                kp.has_code = False
                kp.code_age = 0
                continue
            kp.has_code = True
            kp.code_age = max(0, plan.frame_idx - ent.last_desc)
            codes_out[i] = np.uint64(ent.code)
            if descs_out is not None and ent.desc is not None:
                descs_out[i] = np.frombuffer(ent.desc, dtype=np.uint8)
        return codes_out, descs_out

    def _new_spatial_key(self):
        self._next_spatial_key -= 1  # negative ints never collide with ids
        return self._next_spatial_key
