# -*- coding: utf-8 -*-
"""熵权 TOPSIS 综合评价。"""
from __future__ import annotations

import numpy as np

from .normalization import entropy_weights


def topsis(matrix, weights, benefit_indices, cost_indices):
    """标准 TOPSIS 计算。

    效益型指标取最大值为正理想解，成本型指标取最小值为正理想解；
    正负理想解同时用于计算所有指标的距离。
    """
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2:
        raise ValueError("matrix must be 2-D")
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if data.shape[1] != weights.size:
        raise ValueError("weights length must match matrix columns")

    ordered_indices = list(benefit_indices) + list(cost_indices)
    if not ordered_indices:
        raise ValueError("at least one indicator index is required")
    if len(set(ordered_indices)) != len(ordered_indices):
        raise ValueError("benefit/cost indices must be unique")
    if min(ordered_indices) < 0 or max(ordered_indices) >= data.shape[1]:
        raise ValueError("indicator index out of range")

    norms = np.sqrt(np.sum(data ** 2, axis=0))
    normalized = np.divide(
        data,
        norms,
        out=np.zeros_like(data),
        where=norms != 0,
    )
    weighted = normalized * weights
    ordered = weighted[:, ordered_indices]
    benefit_count = len(benefit_indices)

    if benefit_count == 0:
        best = ordered.min(axis=0)
        worst = ordered.max(axis=0)
    elif benefit_count == len(ordered_indices):
        best = ordered.max(axis=0)
        worst = ordered.min(axis=0)
    else:
        best = np.concatenate(
            [
                ordered[:, :benefit_count].max(axis=0),
                ordered[:, benefit_count:].min(axis=0),
            ]
        )
        worst = np.concatenate(
            [
                ordered[:, :benefit_count].min(axis=0),
                ordered[:, benefit_count:].max(axis=0),
            ]
        )

    positive_distance = np.sqrt(np.sum((ordered - best) ** 2, axis=1))
    negative_distance = np.sqrt(np.sum((ordered - worst) ** 2, axis=1))
    denominator = positive_distance + negative_distance
    scores = np.divide(
        negative_distance,
        denominator,
        out=np.zeros_like(negative_distance),
        where=denominator != 0,
    )
    return scores, positive_distance, negative_distance, weighted


def entropy_topsis(matrix, benefit_indices, cost_indices):
    """先计算熵权，再执行 TOPSIS，返回完整中间结果。"""
    normalized, weights = entropy_weights(matrix, cost_indices)
    scores, positive_distance, negative_distance, weighted = topsis(
        matrix,
        weights,
        benefit_indices,
        cost_indices,
    )
    return {
        "scores": scores,
        "weights": weights,
        "normalized": normalized,
        "weighted": weighted,
        "positive_distance": positive_distance,
        "negative_distance": negative_distance,
    }


def rank_scores(scores) -> np.ndarray:
    """返回每个样本的排名，分数最高者排名为 1。"""
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, scores.size + 1)
    return ranks
