"""Host-only tests for FeaturePipeline construction modes.

These construct FeaturePipeline against the committed default archive
(NNArchive parsing is host-side; no device is opened — __enter__ is never
called), so they run anywhere depthai is installed.
"""

from pathlib import Path

import pytest

from core.pipeline import FeaturePipeline

ARCHIVE = (
    Path(__file__).resolve().parents[1]
    / "depthai_models"
    / "descriptor64_itq_strip32_slim.tar.xz"
)

pytestmark = pytest.mark.skipif(not ARCHIVE.is_file(), reason="descriptor archive not built")


def test_no_tracking_forces_decimate_1():
    fp = FeaturePipeline(str(ARCHIVE), max_keypoints=32, tracking=False,
                         tracker_decimate=4)
    assert fp._tracking is False
    assert fp._decimate == 1  # no track IDs -> nothing to propagate


def test_tracking_default_keeps_decimate():
    fp = FeaturePipeline(str(ARCHIVE), max_keypoints=32, tracker_decimate=3)
    assert fp._tracking is True
    assert fp._decimate == 3


def test_describe_budget_defaults_and_clamps():
    fp = FeaturePipeline(str(ARCHIVE), max_keypoints=100)
    assert fp._describe_budget == 100  # legacy describe-everything
    fp = FeaturePipeline(str(ARCHIVE), max_keypoints=100, describe_budget=30)
    assert fp._describe_budget == 30
    fp = FeaturePipeline(str(ARCHIVE), max_keypoints=100, describe_budget=999)
    assert fp._describe_budget == 100  # clamped to the keypoint budget


COLOR_ARCHIVE = ARCHIVE.with_name("descriptor64_itq_strip32_slim_color.tar.xz")


@pytest.mark.skipif(not COLOR_ARCHIVE.is_file(), reason="color archive not built")
def test_no_track_color_halves_color_stream_resolution():
    # No-track color mode defaults the color stream to half resolution (the
    # host upscales x2; frees USB2 budget for strips at full 30 fps cadence).
    # Halving the RATE instead throttles the whole sensor — measured dead end.
    fp = FeaturePipeline(str(COLOR_ARCHIVE), max_keypoints=64, tracking=False)
    assert fp._color and fp._color_scale == 2
    # Tracking mode and explicit overrides are untouched.
    fp = FeaturePipeline(str(COLOR_ARCHIVE), max_keypoints=64)
    assert fp._color_scale == 1
    fp = FeaturePipeline(str(COLOR_ARCHIVE), max_keypoints=64, tracking=False,
                         color_scale=1)
    assert fp._color_scale == 1
