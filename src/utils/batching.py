# batching.py: Mini-batch generator with optional shuffling for multiple arrays.

import numpy as np


def get_batches(*arrays, batch_size, shuffle=False):
    n = len(arrays[0])
    indices = np.random.permutation(n) if shuffle else np.arange(n)

    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        batches = tuple(arr[idx] for arr in arrays)
        if len(batches) == 1:
            yield batches[0]
        else:
            yield batches
