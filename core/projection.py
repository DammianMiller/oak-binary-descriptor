"""512-bit -> 64-bit binary descriptor compression strategies.

Every strategy is expressed as a single (64, 512) float32 projection matrix W,
applied as ``code_bits = (W @ desc512) > 0`` where ``desc512`` is the raw
512-d float output of the base descriptor network (sign of the raw output is
the 512-bit binary descriptor). Because all strategies share this form, W can
be baked into the ONNX graph as a MatMul initializer and compiled into the
Myriad X blob (see tools/make_projection.py).

Strategies
----------
subsample : Morton/stride-style bit selection. W rows are one-hot; output bit
    i samples input bit ``(i % 8) * 64 + (i // 8)``, i.e. a Z-order-style
    interleave across the 8 x 64-bit descriptor chunks. Deterministic, no data.
xorfold : chunk folding. W[r, c] = 1 for c in the 8 chunks at position r;
    sign() of the sum approximates parity (XOR-fold) with a majority vote.
lsh     : SimHash random hyperplanes. W rows are seeded random +/-1 vectors;
    Hamming distance in 64-bit space approximates Hamming in 512-bit space.
itq     : Iterative Quantization. PCA to 64 dims followed by a learned
    rotation minimizing quantization error, fit on calibration descriptors.
"""

from __future__ import annotations

import numpy as np

IN_BITS = 512
OUT_BITS = 64
STRATEGIES = ("subsample", "xorfold", "lsh", "itq")


def morton_interleave_indices(in_bits: int = IN_BITS, out_bits: int = OUT_BITS) -> np.ndarray:
    """Z-order-style spread of out_bits samples across in_bits positions.

    Splits the descriptor into ``in_bits // out_bits`` chunks of ``out_bits``
    bits and picks output bit i from chunk ``i % n_chunks`` at position
    ``i // n_chunks`` -- the bit-interleaving pattern of a Morton code,
    applied to descriptor chunks instead of coordinate bits.
    """
    n_chunks = in_bits // out_bits
    if n_chunks * out_bits != in_bits:
        raise ValueError(f"in_bits ({in_bits}) must be a multiple of out_bits ({out_bits})")
    idx = np.empty(out_bits, dtype=np.int64)
    for i in range(out_bits):
        idx[i] = (i % n_chunks) * out_bits + (i // n_chunks)
    return idx


def _subsample_matrix(in_bits: int, out_bits: int) -> np.ndarray:
    idx = morton_interleave_indices(in_bits, out_bits)
    W = np.zeros((out_bits, in_bits), dtype=np.float32)
    W[np.arange(out_bits), idx] = 1.0
    return W


def _xorfold_matrix(in_bits: int, out_bits: int) -> np.ndarray:
    n_chunks = in_bits // out_bits
    W = np.zeros((out_bits, in_bits), dtype=np.float32)
    for k in range(n_chunks):
        W[np.arange(out_bits), k * out_bits + np.arange(out_bits)] = 1.0
    # Note: applied to raw float descriptors this is a fold-sum vote, not true
    # bit parity; it approximates XOR-fold behavior for roughly zero-mean,
    # symmetric descriptor outputs.
    return W


def _lsh_matrix(in_bits: int, out_bits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(out_bits, in_bits))
    return W


def fit_itq(
    calibration: np.ndarray,
    out_bits: int = OUT_BITS,
    n_iters: int = 50,
    seed: int = 0,
) -> np.ndarray:
    """Fit an ITQ projection on calibration descriptors.

    Args:
        calibration: (n, in_bits) float array of raw (pre-sign) descriptor
            outputs. Rows are individual descriptors.
        out_bits: target code length (must divide workload of PCA dims).
        n_iters: ITQ rotation iterations.
        seed: RNG seed for rotation initialization.

    Returns:
        (out_bits, in_bits) float32 projection matrix W.
    """
    X = np.asarray(calibration, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < out_bits:
        raise ValueError("calibration must be (n, in_bits) with n >= out_bits")
    in_bits = X.shape[1]

    # Center, then PCA via SVD to out_bits components.
    X = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt[:out_bits].T  # (in_bits, out_bits)

    # ITQ: find rotation R minimizing ||B - Z R||^2, B = sign(Z R), Z = X V.
    Z = X @ V
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((out_bits, out_bits))
    R, _ = np.linalg.qr(R)
    for _ in range(n_iters):
        B = np.sign(Z @ R)
        B[B == 0] = 1.0
        C = B.T @ Z
        Ub, _, Vbt = np.linalg.svd(C)
        R = Ub @ Vbt
    return (R.T @ V.T).astype(np.float32)


def make_projection(
    strategy: str,
    in_bits: int = IN_BITS,
    out_bits: int = OUT_BITS,
    seed: int = 0,
    calibration: np.ndarray | None = None,
) -> np.ndarray:
    """Build the (out_bits, in_bits) projection matrix for a strategy."""
    if strategy == "subsample":
        return _subsample_matrix(in_bits, out_bits)
    if strategy == "xorfold":
        return _xorfold_matrix(in_bits, out_bits)
    if strategy == "lsh":
        return _lsh_matrix(in_bits, out_bits, seed)
    if strategy == "itq":
        if calibration is None:
            raise ValueError("itq strategy requires calibration descriptors")
        return fit_itq(calibration, out_bits=out_bits, seed=seed)
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")


def apply_projection(W: np.ndarray, desc: np.ndarray) -> np.ndarray:
    """Apply W to raw descriptor(s) and return 0/1 bits.

    Args:
        W: (out_bits, in_bits) projection matrix.
        desc: (in_bits,) or (n, in_bits) raw float descriptor output.

    Returns:
        (out_bits,) or (n, out_bits) uint8 array of 0/1 bits.
    """
    proj = np.asarray(desc, dtype=np.float32) @ np.asarray(W, dtype=np.float32).T
    return (proj > 0).astype(np.uint8)


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """Pack 0/1 bits into uint64 codes.

    Args:
        bits: (out_bits,) or (n, out_bits) array of 0/1, out_bits <= 64.

    Returns:
        uint64 scalar or (n,) uint64 array. Bit 0 of the code is bits[0].
    """
    bits = np.asarray(bits, dtype=np.uint8)
    single = bits.ndim == 1
    if single:
        bits = bits[None, :]
    n, nbits = bits.shape
    if nbits > 64:
        raise ValueError("pack_bits supports at most 64 bits per code")
    padded = np.zeros((n, 64), dtype=np.uint8)
    padded[:, :nbits] = bits
    codes = np.packbits(padded, axis=1, bitorder="little").view(np.uint64).ravel()
    return codes[0] if single else codes


_POPCOUNT8 = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1, bitorder="little").sum(axis=1)


def hamming64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamming distance between uint64 codes (scalar or broadcastable arrays)."""
    xor = np.bitwise_xor(np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64))
    # View each uint64 as its 8 bytes, popcount per byte, then sum bytes per
    # element. The reshape keeps the trailing 8-byte axis that .view adds;
    # going through a 1-D view also keeps 0-d scalar inputs working.
    per_byte = _POPCOUNT8[xor.reshape(-1).view(np.uint8)].reshape(xor.shape + (8,))
    return per_byte.sum(axis=-1)
