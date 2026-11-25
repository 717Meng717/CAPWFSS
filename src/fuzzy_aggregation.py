# -*- coding: utf-8 -*-
"""三角模糊数聚合。"""
from __future__ import annotations

import math

import numpy as np


def aggregate_triangular_fuzzy(assessments, round_mid: bool = False):
    """聚合多个三角模糊评价 (l, m, u)。

    左端取最小值，中值取均值，右端取最大值。
    round_mid=True 时保留原项目的向上取整行为。
    """
    arr = np.asarray(assessments, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("assessments must have shape (n, 3)")
    if arr.shape[0] == 0:
        raise ValueError("assessments must not be empty")

    lower = float(arr[:, 0].min())
    middle = float(arr[:, 1].mean())
    if round_mid:
        middle = float(math.ceil(middle))
    upper = float(arr[:, 2].max())
    return np.array([lower, middle, upper], dtype=float)
