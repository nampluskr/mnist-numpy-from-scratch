# random.py: Random seed utilities for reproducible experiments.

import numpy as np


def set_seed(seed):
    np.random.seed(seed)
