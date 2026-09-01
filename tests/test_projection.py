"""Host-only tests for core.projection (512 -> 64 bit compression strategies)."""

import numpy as np
import pytest

from core.projection import (
    IN_BITS,
    OUT_BITS,
    STRATEGIES,
    apply_projection,
    fit_itq,
    hamming64,
    make_projection,
    morton_interleave_indices,
    pack_bits,
)


class TestMortonInterleave:
    def test_length_and_range(self):
        idx = morton_interleave_indices()
        assert idx.shape == (OUT_BITS,)
        assert len(np.unique(idx)) == OUT_BITS  # all distinct
        assert idx.min() >= 0 and idx.max() < IN_BITS

    def test_ordering_pattern(self):
        """Output bit i samples chunk (i % 8) at position (i // 8)."""
        idx = morton_interleave_indices()
        n_chunks = IN_BITS // OUT_BITS  # 8
        # First n_chunks outputs step across chunk starts: 0, 64, 128, ...
        assert list(idx[:n_chunks]) == [k * OUT_BITS for k in range(n_chunks)]
        # Every n_chunks-th output is the same chunk, next position.
        assert idx[0] == 0 and idx[n_chunks] == 1 and idx[2 * n_chunks] == 2
        # General formula holds for every index.
        for i in range(OUT_BITS):
            assert idx[i] == (i % n_chunks) * OUT_BITS + (i // n_chunks)

    def test_spread_across_chunks(self):
        """Each group of 8 consecutive outputs covers all 8 chunks exactly once."""
        idx = morton_interleave_indices()
        n_chunks = IN_BITS // OUT_BITS
        for start in range(0, OUT_BITS, n_chunks):
            chunks = idx[start : start + n_chunks] // OUT_BITS
            assert sorted(chunks.tolist()) == list(range(n_chunks))

    def test_rejects_non_multiple(self):
        with pytest.raises(ValueError):
            morton_interleave_indices(in_bits=100, out_bits=64)


class TestMakeProjection:
    @pytest.mark.parametrize("strategy", ["subsample", "xorfold", "lsh"])
    def test_shapes_and_dtypes(self, strategy):
        W = make_projection(strategy)
        assert W.shape == (OUT_BITS, IN_BITS)
        assert W.dtype == np.float32

    def test_itq_shape_and_dtype(self):
        calibration = np.random.default_rng(0).standard_normal((500, IN_BITS)).astype(np.float32)
        W = make_projection("itq", calibration=calibration)
        assert W.shape == (OUT_BITS, IN_BITS)
        assert W.dtype == np.float32

    def test_itq_requires_calibration(self):
        with pytest.raises(ValueError):
            make_projection("itq")

    def test_unknown_strategy(self):
        with pytest.raises(ValueError):
            make_projection("does-not-exist")

    def test_subsample_is_one_hot(self):
        W = make_projection("subsample")
        assert np.all((W == 0.0) | (W == 1.0))
        assert np.all(W.sum(axis=1) == 1.0)  # each output samples exactly one input bit

    def test_xorfold_folds_all_chunks(self):
        W = make_projection("xorfold")
        n_chunks = IN_BITS // OUT_BITS
        assert np.all(W.sum(axis=1) == n_chunks)  # each output sums one bit per chunk

    def test_lsh_determinism_by_seed(self):
        W1 = make_projection("lsh", seed=7)
        W2 = make_projection("lsh", seed=7)
        W3 = make_projection("lsh", seed=8)
        np.testing.assert_array_equal(W1, W2)
        assert not np.array_equal(W1, W3)
        # SimHash rows are random +/-1 hyperplanes.
        assert set(np.unique(W1)).issubset({-1.0, 1.0})

    def test_itq_beats_lsh_on_clustered_data(self):
        """ITQ adapts to the data manifold, so same-cluster codes should be
        closer than with data-agnostic LSH hyperplanes."""
        rng = np.random.default_rng(0)
        n_clusters, n_per = 8, 60
        # Clusters live on a low-dimensional, high-variance subspace; noise is
        # small and isotropic -- the regime PCA+ITQ is designed for.
        subspace = rng.standard_normal((IN_BITS, 6))
        centers = rng.standard_normal((n_clusters, 6)) @ subspace.T * 3.0
        X = np.vstack([centers[c] + rng.standard_normal((n_per, IN_BITS)) * 0.1 for c in range(n_clusters)])
        labels = np.repeat(np.arange(n_clusters), n_per)
        X = X.astype(np.float32)

        W_itq = fit_itq(X, seed=0)
        W_lsh = make_projection("lsh", seed=0)

        def within_cluster_ratio(W):
            codes = apply_projection(W, X).astype(bool)
            d_same, d_diff = [], []
            for i in range(0, len(X), 3):  # subsample pairs to keep the test fast
                for j in range(i + 1, min(i + 20, len(X))):
                    d = int(np.count_nonzero(codes[i] != codes[j]))
                    (d_same if labels[i] == labels[j] else d_diff).append(d)
            return np.mean(d_same) / np.mean(d_diff)

        assert within_cluster_ratio(W_itq) < within_cluster_ratio(W_lsh)


class TestApplyAndPack:
    def test_apply_projection_bits_match_sign(self):
        rng = np.random.default_rng(0)
        W = make_projection("lsh", seed=1)
        desc = rng.standard_normal((5, IN_BITS)).astype(np.float32)
        bits = apply_projection(W, desc)
        assert bits.shape == (5, OUT_BITS)
        assert bits.dtype == np.uint8
        np.testing.assert_array_equal(bits, (desc @ W.T > 0).astype(np.uint8))

    def test_apply_projection_single_descriptor(self):
        W = make_projection("subsample")
        desc = np.random.default_rng(0).standard_normal(IN_BITS).astype(np.float32)
        bits = apply_projection(W, desc)
        assert bits.shape == (OUT_BITS,)

    def test_pack_bits_round_trip(self):
        """Pack 64 known bits into a uint64 and verify every bit position."""
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, size=OUT_BITS).astype(np.uint8)
        code = pack_bits(bits)
        assert isinstance(code, np.uint64)
        unpacked = np.unpackbits(np.asarray([code], dtype=np.uint64).view(np.uint8), bitorder="little")
        np.testing.assert_array_equal(unpacked[:OUT_BITS], bits)

    def test_pack_bits_known_value(self):
        bits = np.zeros(OUT_BITS, dtype=np.uint8)
        bits[0] = 1
        bits[3] = 1
        bits[63] = 1
        code = pack_bits(bits)
        assert int(code) == (1 << 0) | (1 << 3) | (1 << 63)

    def test_pack_bits_batch(self):
        bits = np.random.default_rng(0).integers(0, 2, size=(10, OUT_BITS)).astype(np.uint8)
        codes = pack_bits(bits)
        assert codes.shape == (10,) and codes.dtype == np.uint64
        for row, code in zip(bits, codes):
            np.testing.assert_array_equal(pack_bits(row), code)

    def test_pack_bits_rejects_too_many_bits(self):
        with pytest.raises(ValueError):
            pack_bits(np.zeros(65, dtype=np.uint8))

    def test_subsample_projection_selects_bits(self):
        """Sanity check of the whole path: subsampled code bits equal the
        selected input bits of a 0/1 float descriptor."""
        bits_in = np.random.default_rng(0).integers(0, 2, size=IN_BITS).astype(np.float32)
        W = make_projection("subsample")
        idx = morton_interleave_indices()
        bits_out = apply_projection(W, bits_in)
        np.testing.assert_array_equal(bits_out, bits_in[idx].astype(np.uint8))


class TestHamming64:
    def test_scalar_known_values(self):
        assert hamming64(np.uint64(0), np.uint64(0)) == 0
        assert hamming64(np.uint64(0), np.uint64(np.iinfo(np.uint64).max)) == 64
        assert hamming64(np.uint64(1), np.uint64(2)) == 2
        assert hamming64(np.uint64(0b1010), np.uint64(0b0101)) == 4

    def test_arrays_and_broadcast(self):
        a = np.array([0, 1, 3], dtype=np.uint64)
        b = np.array([0, 2, 3], dtype=np.uint64)
        np.testing.assert_array_equal(hamming64(a, b), [0, 2, 0])
        # Scalar against array broadcasts.
        np.testing.assert_array_equal(hamming64(np.uint64(0), b), [0, 1, 2])

    def test_symmetry_and_self_distance(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, np.iinfo(np.uint64).max, size=50, dtype=np.uint64)
        b = rng.integers(0, np.iinfo(np.uint64).max, size=50, dtype=np.uint64)
        np.testing.assert_array_equal(hamming64(a, b), hamming64(b, a))
        assert np.all(hamming64(a, a) == 0)
