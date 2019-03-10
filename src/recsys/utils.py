import itertools as it
import time
from contextlib import contextmanager

import numpy as np


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


def group_lengths(group_ids):
    return np.array([sum(1 for _ in i) for k, i in it.groupby(group_ids)])


def jaccard(a, b):
    return len(a & b) / (len(a | b) + 1)
