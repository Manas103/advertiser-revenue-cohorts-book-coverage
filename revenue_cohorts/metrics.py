"""Gini coefficient, the one inequality metric this repo needs."""
from __future__ import annotations

import numpy as np


def gini(values) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * x)) / (n * total) - (n + 1.0) / n)
