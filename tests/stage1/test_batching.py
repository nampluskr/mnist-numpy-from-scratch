# test_batching.py: Unit tests for get_batches().

import numpy as np

from src.utils.batching import get_batches


N = 20
BATCH_SIZE = 6

x = np.arange(N * 3, dtype=np.float32).reshape(N, 3)
y = np.arange(N, dtype=np.float32)


class TestGetBatches:
    def test_iterable(self):
        batches = list(get_batches(x, batch_size=BATCH_SIZE))
        assert len(batches) > 0

    def test_single_array_yields_array(self):
        for batch in get_batches(x, batch_size=BATCH_SIZE):
            assert isinstance(batch, np.ndarray)
            break

    def test_multiple_arrays_yields_tuple(self):
        for batch in get_batches(x, y, batch_size=BATCH_SIZE):
            assert isinstance(batch, tuple)
            assert len(batch) == 2
            break

    def test_batch_size(self):
        for batch in get_batches(x, batch_size=BATCH_SIZE):
            assert batch.shape[0] <= BATCH_SIZE
            break

    def test_last_batch_smaller(self):
        batches = list(get_batches(x, batch_size=BATCH_SIZE))
        # N=20, batch_size=6 -> batches of size 6,6,6,2.
        assert batches[-1].shape[0] == N % BATCH_SIZE

    def test_covers_all_samples(self):
        total = sum(b.shape[0] for b in get_batches(x, batch_size=BATCH_SIZE))
        assert total == N

    def test_no_shuffle_preserves_order(self):
        first_batch = next(iter(get_batches(x, batch_size=BATCH_SIZE, shuffle=False)))
        np.testing.assert_array_equal(first_batch, x[:BATCH_SIZE])

    def test_multiple_arrays_consistent_shuffle(self):
        np.random.seed(0)
        for xb, yb in get_batches(x, y, batch_size=BATCH_SIZE, shuffle=True):
            # Each row of xb sums to 3*i, 3*i+1, 3*i+2 -> sum = 9*i+3.
            # y value equals the original row index
            for j in range(len(xb)):
                expected_idx = int(yb[j])
                np.testing.assert_array_equal(xb[j], x[expected_idx])
            break
