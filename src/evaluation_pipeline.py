# -*- coding: utf-8 -*-
"""评价主流程：熵权 + TOPSIS + 排名。"""
from __future__ import annotations

from .entropy_topsis import entropy_topsis, rank_scores


def evaluate(matrix, benefit_indices, cost_indices):
    """执行综合评估并补充排名。"""
    result = entropy_topsis(matrix, benefit_indices, cost_indices)
    result["ranks"] = rank_scores(result["scores"])
    return result
