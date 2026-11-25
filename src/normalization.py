# -*- coding: utf-8 -*-
"""指标归一化与熵权计算。"""
from __future__ import annotations

import numpy as np


def min_max(values, reverse: bool = False) -> np.ndarray:
    """Min-Max 归一化。

    reverse=False 用于效益型指标，reverse=True 用于成本型指标。
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("values must not be empty")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("values must contain finite numbers")
    low = float(finite.min())
    high = float(finite.max())
    if np.isclose(high, low):
        return np.zeros_like(arr, dtype=float)
    scaled = (
        (high - arr) / (high - low)
        if reverse
        else (arr - low) / (high - low)
    )
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_matrix(matrix, cost_indices=()) -> np.ndarray:
    """按列归一化矩阵，成本型指标使用反向 Min-Max。"""
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2:
        raise ValueError("matrix must be 2-D")
    cost = set(cost_indices)
    normalized = np.empty_like(data, dtype=float)
    for column in range(data.shape[1]):
        normalized[:, column] = min_max(
            data[:, column],
            reverse=column in cost,
        )
    return normalized


def entropy_weights(matrix, cost_indices=(), eps: float = 1e-12):
    """计算归一化矩阵和熵权。

    返回:
        normalized: 每列归一化后的矩阵
        weights: 每个指标的熵权，形状为 (指标数,)
    """
    normalized = normalize_matrix(matrix, cost_indices)
    sample_count, indicator_count = normalized.shape
    column_sums = normalized.sum(axis=0)
    probabilities = np.divide(
        normalized,
        column_sums,
        out=np.zeros_like(normalized),
        where=column_sums != 0,
    )
    k = 1.0 / np.log(sample_count) if sample_count > 1 else 1.0
    entropy = -k * np.sum(
        probabilities * np.log(probabilities + eps),
        axis=0,
    )
    divergence = 1.0 - entropy
    total = float(divergence.sum())
    if np.isclose(total, 0.0):
        weights = np.full(indicator_count, 1.0 / indicator_count)
    else:
        weights = divergence / total
    return normalized, weights
