# dataloader.py: Generic mini-batch iterator for datasets with __len__ and __getitem__.

import numpy as np


class Dataloader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        indices = np.random.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            images = np.stack([self.dataset[i][0] for i in batch_idx])
            targets = np.stack([self.dataset[i][1] for i in batch_idx])
            yield images, targets
