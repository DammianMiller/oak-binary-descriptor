"""DescribeGate (change-gated describing) and its output-contract tests."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.gate import DescribeGate, patch_fingerprints
from core.packer import KeypointRecord, pack_frame
from core.pipeline import FeaturePipeline

ARCHIVE = (
    Path(__file__).resolve().parents[1]
    / "depthai_models"
    / "descriptor64_itq_strip32_slim.tar.xz"
)


def feat(x, y, fid=-1, age=0):
    return SimpleNamespace(
        id=fid, position=SimpleNamespace(x=float(x), y=float(y)), age=age
    )


def kps_of(features):
    return [
        KeypointRecord(x=f.position.x, y=f.position.y,
                       track_id=int(f.id), age=int(f.age))
        for f in features
    ]


def run_frame(gate, features, img, publish_full_desc=False):
    """One select+apply cycle with synthetic fresh codes for the selected."""
    plan = gate.select(features, img)
    fresh = np.arange(1, len(plan.sel) + 1, dtype=np.uint64) * 1000 + gate._frame
    descs = (
        np.full((len(plan.sel), 64), 7, dtype=np.uint8) if publish_full_desc else None
    )
    kps = kps_of(features)
    codes, descs = gate.apply(plan, features, kps, fresh, descs, publish_full_desc)
    return plan, kps, codes, descs


def scene(seed=3, size=(200, 200)):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size, dtype=np.uint8)


def test_fingerprints_vectorized_shapes():
    img = scene()
    xy = np.array([[10.0, 10.0], [199.0, 199.0], [50.5, 60.5]])  # borders clamp
    fp = patch_fingerprints(img, xy)
    assert fp.shape == (3, 64)
    bgr = np.repeat(img[..., None], 3, axis=2)
    fp3 = patch_fingerprints(bgr, xy)
    assert fp3.shape == (3, 192)
    # Identical patches -> identical fingerprints; clamped windows are stable.
    assert np.allclose(patch_fingerprints(img, xy[:1]), fp[:1])
    assert patch_fingerprints(img, np.zeros((0, 2))).shape == (0, 64)


def test_static_scene_describes_once_then_reuses():
    img = scene()
    gate = DescribeGate(budget=10, change_thresh=6.0, refresh_every=30)
    feats = [feat(50, 50, fid=1), feat(120, 80, fid=2), feat(30, 150, fid=3)]
    plan1, _, codes1, _ = run_frame(gate, feats, img)
    assert plan1.sel == [0, 1, 2]  # all new
    plan2, kps2, codes2, _ = run_frame(gate, feats, img)
    assert plan2.sel == []  # nothing changed
    assert np.array_equal(codes2, codes1)  # cached codes published again
    assert all(kp.has_code for kp in kps2)
    assert all(kp.code_age == 1 for kp in kps2)  # described one frame ago


def test_changed_patch_is_redescribed():
    img = scene()
    gate = DescribeGate(budget=10)
    feats = [feat(50, 50, fid=1), feat(120, 80, fid=2)]
    run_frame(gate, feats, img)
    img2 = img.copy()
    img2[34:66, 34:66] = (np.asarray(img2[34:66, 34:66], int) + 120) % 256
    plan, kps, codes, _ = run_frame(gate, feats, img2)
    assert plan.sel == [0]
    assert kps[0].code_age == 0  # fresh
    assert kps[1].code_age == 1  # still cached


def test_no_track_spatial_reuse_over_moved_identical_texture():
    # Uniform-ish texture: a keypoint that moved 3 px onto identical content
    # must reuse its cached code (fingerprints compare content, not position).
    img = np.full((200, 200), 128, dtype=np.uint8)
    img[40:60, 40:60] = 200  # landmark away from both positions
    gate = DescribeGate(budget=10, match_radius=6.0)
    run_frame(gate, [feat(100, 100)], img)
    plan, kps, codes, _ = run_frame(gate, [feat(103, 102)], img)
    assert plan.sel == []
    assert kps[0].has_code and kps[0].code_age == 1


def test_refresh_every_forces_redescribe():
    img = scene()
    gate = DescribeGate(budget=10, refresh_every=3)
    feats = [feat(50, 50, fid=1)]
    run_frame(gate, feats, img)  # frame 1: described (staleness 0)
    p2, kps2, _, _ = run_frame(gate, feats, img)  # 2: staleness 1, reuse
    p3, kps3, _, _ = run_frame(gate, feats, img)  # 3: staleness 2, reuse
    p4, kps4, _, _ = run_frame(gate, feats, img)  # 4: staleness 3 -> refresh
    assert p2.sel == [] and kps2[0].code_age == 1
    assert p3.sel == [] and kps3[0].code_age == 2
    assert p4.sel == [0] and kps4[0].code_age == 0


def test_budget_caps_selection_and_rest_keep_old_codes():
    img = scene()
    gate = DescribeGate(budget=1)
    feats = [feat(50, 50, fid=1), feat(120, 80, fid=2), feat(30, 150, fid=3)]
    plan, kps, codes, _ = run_frame(gate, feats, img)
    assert len(plan.sel) == 1  # capped
    described = plan.sel[0]
    assert [kp.has_code for kp in kps] == [
        i == described for i in range(3)
    ]
    # Next frame: the already-cached one reuses; budget goes to a new one.
    plan2, kps2, _, _ = run_frame(gate, feats, img)
    assert described not in plan2.sel
    assert len(plan2.sel) == 1
    assert sum(kp.has_code for kp in kps2) == 2


def test_dead_entries_are_evicted():
    img = scene()
    gate = DescribeGate(budget=10, refresh_every=3)
    run_frame(gate, [feat(50, 50, fid=9)], img)
    assert len(gate._cache) == 1
    for _ in range(4):  # absent longer than refresh_every
        run_frame(gate, [], img)
    assert len(gate._cache) == 0


def test_lost_nn_output_keeps_previous_codes():
    img = scene()
    gate = DescribeGate(budget=10)
    feats = [feat(50, 50, fid=1)]
    run_frame(gate, feats, img)
    img2 = img.copy()
    img2[34:66, 34:66] = (np.asarray(img2[34:66, 34:66], int) + 120) % 256
    plan = gate.select(feats, img2)
    assert plan.sel == [0]
    # Drain returns nothing (missing NN output): stale code survives.
    kps = kps_of(feats)
    codes, _ = gate.apply(plan, feats, kps, np.zeros(0, np.uint64), None, False)
    assert kps[0].has_code and kps[0].code_age == 1
    assert codes[0] != 0


def test_pack_frame_parallel_codes_with_has_code():
    kps = [
        KeypointRecord(x=1, y=2, track_id=1),
        KeypointRecord(x=3, y=4, track_id=2, has_code=False),
        KeypointRecord(x=5, y=6, track_id=3),
    ]
    codes = np.array([30, 0, 10], dtype=np.uint64)
    rows = pack_frame(kps, codes, None)
    by_tid = {r["track_id"]: r for r in rows}
    assert by_tid[1]["has_zorder"] and by_tid[1]["zorder"] == 30
    assert not by_tid[2]["has_zorder"] and by_tid[2]["zorder"] == 0
    assert by_tid[3]["has_zorder"] and by_tid[3]["zorder"] == 10
    # Codeless row sorts last.
    assert rows[-1]["track_id"] == 2


def test_gate_constructor_plumbing():
    fp = FeaturePipeline(str(ARCHIVE), max_keypoints=100, describe_gate=True,
                         describe_budget=25, change_thresh=4.0, refresh_every=10)
    assert fp._gate is not None
    assert fp._gate._budget == 25
    assert fp._gate._thresh == 4.0
    assert fp._gate._refresh == 10
    fp = FeaturePipeline(str(ARCHIVE), max_keypoints=100)
    assert fp._gate is None


def test_nn_stall_streak_raises_after_three_misses():
    # Camera/tracker streams can keep flowing while the NCE wedges; three
    # consecutive frames with missing NN outputs must fail loudly (not warn
    # forever) and flag the session dead so __exit__ skips close.
    import datetime

    class EmptyQ:
        def get(self, _timeout):
            return None

    fp = FeaturePipeline(str(ARCHIVE), max_keypoints=32)
    fp._q_nns = [EmptyQ()]
    fp._publish_full_desc = False
    job = {
        "n_described": 1, "strip_nodes": [0], "frame": None,
        "keypoints": [], "features": [], "gate_plan": None,
    }
    for _ in range(2):
        fp._drain_job(dict(job), timeout=0.01)  # misses 1-2: warn only
    assert fp._nn_stall_streak == 2 and not fp._dead
    with pytest.raises(RuntimeError, match="NCE wedge"):
        fp._drain_job(dict(job), timeout=0.01)
    assert fp._dead


def test_dead_session_skips_close():
    # A session known to be dead (firmware crash) must NOT call
    # pipeline.stop()/device.close(): depthai's close path RPCs the dead
    # device and can segfault the host (measured via device.crashDevice()).
    fp = FeaturePipeline(str(ARCHIVE), max_keypoints=32)
    pipeline = SimpleNamespace(isRunning=lambda: True,
                               stop=lambda: pytest.fail("stop() called on dead session"))
    device = SimpleNamespace(close=lambda: pytest.fail("close() called on dead session"))
    fp._pipeline = pipeline
    fp._device = device
    fp._dead = True
    fp.__exit__(None, None, None)
    assert fp._pipeline is None and fp._device is None
    # Live sessions close normally.
    calls = []
    fp2 = FeaturePipeline(str(ARCHIVE), max_keypoints=32)
    fp2._pipeline = SimpleNamespace(isRunning=lambda: True,
                                    stop=lambda: calls.append("stop"))
    fp2._device = SimpleNamespace(close=lambda: calls.append("close"))
    fp2.__exit__(None, None, None)
    assert calls == ["stop", "close"]
