"""Opponent color space transform for the color (Opponent-LATCH-style)
descriptor variant.

The descriptor CNN consumes the three opponent channels instead of raw
intensity (paper inspiration: Opponent-LATCH, IEEE 9824924, computes LATCH on
the opponent channels and concatenates; here a learned CNN consumes them
jointly):

    O1 = (R - G) / sqrt(2)      red-green
    O2 = (R + G - 2B) / sqrt(6) blue-yellow
    O3 = (R + G + B) / sqrt(3)  intensity

Transport: each channel is affine-mapped into a uint8 plane with zero at
127.5, and the three planes of every patch are folded into the strip height
(channel-major), so the blob keeps a single-channel GRAY8 input of
(1, 1, 96*B, 32). Inside the graph a Reshape re-exposes (1, 3, 32*B, 32) and
(x - 127.5) / 127.5 recovers the [-1, 1] range the network was trained on
(the training data generator quantizes onto the same uint8 grid).

Normalized channel ranges with R, G, B in [0, 255]:
    o1 = (R - G) / 255            in [-1, 1]
    o2 = (R + G - 2B) / 510       in [-1, 1]
    o3 = 2 (R + G + B) / 765 - 1  in [-1, 1]   (mean brightness)

o3 normalizes the paper's O3 = (R+G+B)/sqrt(3) (range [0, 255*sqrt(3)]) to
[-1, 1]; it must span the full range because the training generator samples
o3 uniformly in [-1, 1]. (An earlier version divided by 255*sqrt(3) without
the 1/sqrt(3) on the sum, saturating o3 at +1 for any pixel brighter than
mean 147 and skewing inference away from the training distribution.)
"""

import numpy as np

PATCH_SIZE = 32
_CHANNELS = 3

# Hue rotation on the normalized (o1, o2) plane must account for the
# different axis scalings: O1 = o1 * 255/sqrt(2), O2 = o2 * 510/sqrt(6).
O1_SCALE = 255.0 / np.sqrt(2.0)   # 180.31
O2_SCALE = 510.0 / np.sqrt(6.0)   # 208.17


def bgr_to_opponent_u8(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 patches (..., H, W, 3) to opponent planes
    (..., 3, H, W) uint8, zero-centered at 127.5.

    Vectorized; works on a single (H, W, 3) crop or a batch (N, H, W, 3).
    """
    x = np.asarray(bgr, dtype=np.float32)
    b, g, r = x[..., 0], x[..., 1], x[..., 2]
    o1 = (r - g) / 255.0
    o2 = (r + g - 2.0 * b) / 510.0
    o3 = 2.0 * (r + g + b) / 765.0 - 1.0
    planes = np.stack([o1, o2, o3], axis=-3)  # (..., 3, H, W)
    return np.round(np.clip(planes, -1.0, 1.0) * 127.5 + 127.5).astype(np.uint8)


def opponent_u8_to_float(planes: np.ndarray) -> np.ndarray:
    """Undo the uint8 storage mapping: (..., 3, H, W) uint8 -> [-1, 1] float.

    This is exactly the Sub/Div the exported graph applies at its input.
    """
    return (np.asarray(planes, dtype=np.float32) - 127.5) / 127.5


def stack_color_strip(planes: np.ndarray, batch: int) -> np.ndarray:
    """Pack opponent planes (n, 3, 32, 32) uint8 into one (96*batch, 32)
    GRAY8 strip, channel-major: rows [0..32B) hold all O1 planes, then O2,
    then O3. Zero-pads missing patches. Matches the in-graph Reshape from
    (1, 1, 96*B, 32) to (1, 3, 32*B, 32).
    """
    n = len(planes)
    if n > batch:
        raise ValueError(f"{n} patches do not fit a strip of {batch}")
    padded = np.zeros((batch, _CHANNELS, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    padded[:n] = planes
    return padded.transpose(1, 0, 2, 3).reshape(
        _CHANNELS * batch * PATCH_SIZE, PATCH_SIZE
    )


def strip_height(batch: int, color: bool) -> int:
    """Strip input height in pixels for the given patch batch size."""
    return PATCH_SIZE * batch * (_CHANNELS if color else 1)
