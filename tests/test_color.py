"""Host-only tests for the opponent color transform and color strip packing."""

import numpy as np

from core.color import (
    bgr_to_opponent_u8,
    opponent_u8_to_float,
    stack_color_strip,
    strip_height,
)


def test_known_colors():
    # black: o1=o2=0 (u8 127/128), o3=-1 (u8 0)
    planes = bgr_to_opponent_u8(np.zeros((4, 4, 3), dtype=np.uint8))
    assert planes.shape == (3, 4, 4)
    assert np.all(planes[0] == 128) or np.all(planes[0] == 127)
    assert np.all(planes[2] == 0)
    # white: o3 = +1 (u8 255), chroma zero
    planes = bgr_to_opponent_u8(np.full((4, 4, 3), 255, dtype=np.uint8))
    assert planes[2][0, 0] == 255
    assert abs(int(planes[0][0, 0]) - 128) <= 1
    # pure red (BGR 0,0,255): o1=1, o2=0.5, o3=2/3-1 = -1/3 -> u8 ~85
    red = np.zeros((4, 4, 3), dtype=np.uint8)
    red[..., 2] = 255
    planes = bgr_to_opponent_u8(red)
    assert planes[0][0, 0] == 255
    assert abs(int(planes[1][0, 0]) - 191) <= 1
    assert abs(int(planes[2][0, 0]) - 85) <= 1
    # mid gray: o3 = 0 (full-range intensity channel, no saturation)
    gray = np.full((4, 4, 3), 128, dtype=np.uint8)
    planes = bgr_to_opponent_u8(gray)
    assert abs(int(planes[2][0, 0]) - 128) <= 1


def test_quantization_roundtrip():
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, (17, 32, 32, 3), dtype=np.uint8)
    planes = bgr_to_opponent_u8(bgr)
    back = opponent_u8_to_float(planes)
    # reference float computation
    x = bgr.astype(np.float32)
    o1 = (x[..., 2] - x[..., 1]) / 255.0
    err = np.abs(back[:, 0] - o1)
    assert err.max() <= 1.0 / 127.5 + 1e-6


def test_batch_and_single_agree():
    rng = np.random.default_rng(1)
    bgr = rng.integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)
    batched = bgr_to_opponent_u8(bgr)
    for i in range(5):
        single = bgr_to_opponent_u8(bgr[i])
        np.testing.assert_array_equal(batched[i], single)


def test_stack_layout_channel_major():
    # plane value = 10*channel + patch index: recognizable row pattern
    planes = np.zeros((2, 3, 32, 32), dtype=np.uint8)
    for p in range(2):
        for c in range(3):
            planes[p, c] = 10 * c + p
    strip = stack_color_strip(planes, batch=4)  # pads patches 2..3 with zeros
    assert strip.shape == (3 * 32 * 4, 32)
    B = 4
    for c in range(3):
        for p in range(B):
            block = strip[(c * B + p) * 32 : (c * B + p + 1) * 32]
            expected = (10 * c + p) if p < 2 else 0
            assert np.all(block == expected), f"channel {c} patch {p}"


def test_strip_reshape_matches_graph_semantics():
    # Emulate the graph: (1,1,96B,32) -> Reshape (1,3,32B,32); channel c of
    # the reshaped tensor must be plane c's strip section, patch p occupying
    # rows 32p..32p+31 within it.
    rng = np.random.default_rng(2)
    planes = rng.integers(0, 256, (3, 3, 32, 32), dtype=np.uint8)
    strip = stack_color_strip(planes, batch=3)
    reshaped = strip.reshape(1, 3, 3 * 32, 32)
    for c in range(3):
        for p in range(3):
            np.testing.assert_array_equal(reshaped[0, c, p * 32 : (p + 1) * 32], planes[p, c])


def test_strip_height():
    assert strip_height(32, color=False) == 1024
    assert strip_height(32, color=True) == 3072


class _Feat:
    def __init__(self, x, y):
        self.position = type("P", (), {"x": x, "y": y})


def test_crop_opponent_planes_content():
    from core.pipeline import crop_opponent_planes

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:52, 20:52, 2] = 255  # pure red square at (20..51, 20..51)
    planes = crop_opponent_planes(frame, [_Feat(36, 36)])
    assert planes.shape == (1, 3, 32, 32)
    # crop is the red square exactly: o1 = +1 -> 255, o2 = +0.5 -> ~191
    assert planes[0, 0].min() == 255
    assert abs(int(planes[0, 1, 0, 0]) - 191) <= 1


def test_crop_opponent_planes_border_clamp():
    from core.pipeline import crop_opponent_planes

    frame = np.full((64, 64, 3), 255, dtype=np.uint8)  # all white
    planes = crop_opponent_planes(frame, [_Feat(0, 0), _Feat(63, 63)])
    assert planes.shape == (2, 3, 32, 32)
    assert planes[0, 2].min() == 255  # o3 = +1 everywhere


def test_bgr_to_gray_u8():
    from core.pipeline import bgr_to_gray_u8

    black = np.zeros((4, 4, 3), dtype=np.uint8)
    white = np.full((4, 4, 3), 255, dtype=np.uint8)
    red = np.zeros((4, 4, 3), dtype=np.uint8)
    red[..., 2] = 255
    assert bgr_to_gray_u8(black)[0, 0] == 0
    assert bgr_to_gray_u8(white)[0, 0] == 255
    assert abs(int(bgr_to_gray_u8(red)[0, 0]) - 76) <= 1  # 0.299 * 255
