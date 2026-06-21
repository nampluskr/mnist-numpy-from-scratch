# test_dataloader.py: Tests for Dataloader in src/data/dataloader.py.

import math

import numpy as np
import pytest

from src.data.dataloader import Dataloader


# --- minimal synthetic dataset (generic __len__ + __getitem__) ---

class ToyDataset:
    def __init__(self, n=20):
        self.images = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
        self.targets = np.arange(n, dtype=np.float32).reshape(n, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


@pytest.fixture
def ds():
    return ToyDataset(n=20)


class TestDataloader:
    # __len__
    def test_len_exact_division(self, ds):
        loader = Dataloader(ds, batch_size=4, shuffle=False)
        assert len(loader) == 5

    def test_len_with_remainder(self, ds):
        loader = Dataloader(ds, batch_size=6, shuffle=False)
        assert len(loader) == math.ceil(20 / 6)  # 4

    def test_len_batch_size_larger_than_dataset(self):
        ds = ToyDataset(n=5)
        loader = Dataloader(ds, batch_size=10, shuffle=False)
        assert len(loader) == 1

    # __iter__: batch shapes
    def test_iter_yields_tuples(self, ds):
        loader = Dataloader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert isinstance(batch, tuple) and len(batch) == 2

    def test_iter_batch_image_shape(self, ds):
        loader = Dataloader(ds, batch_size=4, shuffle=False)
        images, _ = next(iter(loader))
        assert images.shape == (4, 4)

    def test_iter_batch_target_shape(self, ds):
        loader = Dataloader(ds, batch_size=4, shuffle=False)
        _, targets = next(iter(loader))
        assert targets.shape == (4, 1)

    def test_iter_correct_num_batches(self, ds):
        loader = Dataloader(ds, batch_size=4, shuffle=False)
        batches = list(loader)
        assert len(batches) == 5

    def test_iter_last_batch_smaller(self, ds):
        loader = Dataloader(ds, batch_size=6, shuffle=False)
        batches = list(loader)
        last_images, _ = batches[-1]
        # 20 % 6 == 2
        assert last_images.shape[0] == 2

    def test_iter_total_samples(self, ds):
        loader = Dataloader(ds, batch_size=7, shuffle=False)
        total = sum(img.shape[0] for img, _ in loader)
        assert total == 20

    # no shuffle: order preserved
    def test_no_shuffle_order(self, ds):
        loader = Dataloader(ds, batch_size=20, shuffle=False)
        images, targets = next(iter(loader))
        expected_images = ds.images
        np.testing.assert_array_equal(images, expected_images)

    # shuffle: order changes
    def test_shuffle_changes_order(self, ds):
        loader = Dataloader(ds, batch_size=20, shuffle=True)
        images, _ = next(iter(loader))
        # With 20 samples it is astronomically unlikely to be the same order
        assert not np.array_equal(images, ds.images)

    def test_shuffle_covers_all_samples(self, ds):
        loader = Dataloader(ds, batch_size=20, shuffle=True)
        images, _ = next(iter(loader))
        assert images.shape[0] == 20
        # Every original row must appear exactly once
        orig_set = {tuple(r) for r in ds.images}
        shuf_set = {tuple(r) for r in images}
        assert orig_set == shuf_set

    # multiple iterations are independent
    def test_multiple_iter_independent(self, ds):
        loader = Dataloader(ds, batch_size=20, shuffle=True)
        order1, _ = next(iter(loader))
        order2, _ = next(iter(loader))
        # Two independent shuffles should (almost certainly) differ
        assert not np.array_equal(order1, order2)
