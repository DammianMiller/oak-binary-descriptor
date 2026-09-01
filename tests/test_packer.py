"""Host-only tests for core.packer (keypoint + code message packing, matching)."""

import numpy as np
import pytest

from core.packer import FULL_DESC_BYTES, KeypointRecord, match_codes, pack_frame
from core.projection import hamming64


def _kps(n: int) -> list[KeypointRecord]:
    return [KeypointRecord(x=float(i * 10), y=float(i), track_id=i, age=i) for i in range(n)]


class TestPackFrame:
    def test_alignment_codes_length(self):
        # More codes than keypoints is still an error.
        with pytest.raises(ValueError, match="codes must be a prefix"):
            pack_frame(_kps(2), codes=np.array([1, 2, 3], dtype=np.uint64))

    def test_prefix_codes_describe_budget(self):
        # describe_budget < max_keypoints: codes align with the FIRST m
        # keypoints; the rest are tracking-only (has_zorder=False, sorted last).
        kps = _kps(4)
        codes = np.array([42, 7], dtype=np.uint64)
        descs = np.zeros((2, FULL_DESC_BYTES), dtype=np.uint8)
        rows = pack_frame(kps, codes=codes, full_descs=descs)
        coded = [r for r in rows if r["has_zorder"]]
        plain = [r for r in rows if not r["has_zorder"]]
        assert [r["track_id"] for r in coded] == [1, 0]  # sorted by code: 7, 42
        assert [r["track_id"] for r in plain] == [2, 3]  # codeless last, by x
        assert all(r["zorder"] == 0 and r["desc"] == b"" for r in plain)
        assert all(len(r["desc"]) == FULL_DESC_BYTES for r in coded)
        # no-sort keeps list order, prefix flags intact
        rows = pack_frame(kps, codes=codes, sort_by_code=False)
        assert [r["has_zorder"] for r in rows] == [True, True, False, False]

    def test_alignment_full_descs_shape(self):
        with pytest.raises(ValueError, match="full_descs must be"):
            pack_frame(_kps(3), full_descs=np.zeros((3, 32), dtype=np.uint8))
        with pytest.raises(ValueError, match="full_descs must be a prefix"):
            pack_frame(_kps(3), full_descs=np.zeros((4, FULL_DESC_BYTES), dtype=np.uint8))

    def test_optional_fields_absent(self):
        rows = pack_frame(_kps(2))
        for row in rows:
            assert row["has_zorder"] is False
            assert row["zorder"] == 0
            assert row["desc"] == b""

    def test_optional_fields_present(self):
        kps = _kps(2)
        codes = np.array([7, 9], dtype=np.uint64)
        descs = np.arange(2 * FULL_DESC_BYTES, dtype=np.uint8).reshape(2, FULL_DESC_BYTES)
        rows = pack_frame(kps, codes=codes, full_descs=descs, sort_by_code=False)
        assert rows[0]["zorder"] == 7 and rows[1]["zorder"] == 9
        assert all(r["has_zorder"] for r in rows)
        assert rows[0]["desc"] == bytes(descs[0].tobytes())
        assert len(rows[1]["desc"]) == FULL_DESC_BYTES

    def test_core_fields(self):
        row = pack_frame([KeypointRecord(x=1.5, y=2.5, track_id=42, age=7)], sort_by_code=False)[0]
        assert row["x"] == 1.5 and row["y"] == 2.5
        assert row["track_id"] == 42 and row["age"] == 7

    def test_sort_order_by_code(self):
        kps = _kps(4)
        codes = np.array([100, 3, 42, 7], dtype=np.uint64)
        rows = pack_frame(kps, codes=codes)
        assert [r["zorder"] for r in rows] == [3, 7, 42, 100]
        # Sorting is by code, not by keypoint order: track_id follows the code.
        assert [r["track_id"] for r in rows] == [1, 3, 2, 0]

    def test_sort_order_codeless_last_then_by_x(self):
        kps = _kps(3)
        rows = pack_frame(kps, codes=None)
        assert [r["x"] for r in rows] == [0.0, 10.0, 20.0]

    def test_no_sort(self):
        kps = _kps(3)
        codes = np.array([100, 3, 42], dtype=np.uint64)
        rows = pack_frame(kps, codes=codes, sort_by_code=False)
        assert [r["zorder"] for r in rows] == [100, 3, 42]


class TestMatchCodes:
    def test_exact_match(self):
        q = np.array([0b1010], dtype=np.uint64)
        r = np.array([0b1010, 0b1111], dtype=np.uint64)
        assert match_codes(q, r, max_distance=0) == [(0, 0, 0)]

    def test_nearest_within_threshold(self):
        q = np.array([0b0000], dtype=np.uint64)
        r = np.array([0b0011, 0b0111], dtype=np.uint64)  # distances 2 and 3
        assert match_codes(q, r, max_distance=4) == [(0, 0, 2)]

    def test_beyond_threshold_rejected(self):
        q = np.array([0b0000], dtype=np.uint64)
        r = np.array([0b0011], dtype=np.uint64)  # distance 2
        assert match_codes(q, r, max_distance=1) == []

    def test_threshold_boundary_inclusive(self):
        q = np.array([0b0000], dtype=np.uint64)
        r = np.array([0b0011], dtype=np.uint64)  # distance 2
        assert match_codes(q, r, max_distance=2) == [(0, 0, 2)]

    def test_one_match_per_query(self):
        q = np.array([0, 0, 7], dtype=np.uint64)
        r = np.array([0, 7], dtype=np.uint64)
        matches = match_codes(q, r, max_distance=0)
        assert len(matches) == len(q)
        # Two queries may legitimately match the same ref code.
        assert [(m[0], m[1]) for m in matches] == [(0, 0), (1, 0), (2, 1)]

    def test_distance_consistent_with_hamming64(self):
        rng = np.random.default_rng(0)
        q = rng.integers(0, np.iinfo(np.uint64).max, size=5, dtype=np.uint64)
        r = rng.integers(0, np.iinfo(np.uint64).max, size=20, dtype=np.uint64)
        for qi, ri, d in match_codes(q, r, max_distance=64):
            assert d == int(hamming64(q[qi], r[ri]))
            assert d <= 64
