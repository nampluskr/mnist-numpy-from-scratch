# test_random.py: Unit tests for set_seed().

import numpy as np

from src.utils.random import set_seed


class TestSetSeed:
    def test_reproducible(self):
        set_seed(42)
        a = np.random.randn(5)
        set_seed(42)
        b = np.random.randn(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(0)
        a = np.random.randn(5)
        set_seed(1)
        b = np.random.randn(5)
        assert not np.array_equal(a, b)
