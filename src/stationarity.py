# -*- coding: utf-8 -*-
"""时间序列趋势统计。

说明：原始项目中的 DF-GLS.py 实际执行的是趋势回归并取斜率 t 统计量，
这里将该逻辑整理为可复用的趋势统计模块，不冒充严格的 DF-GLS 检验。
"""
from __future__ import annotations

import numpy as np
import statsmodels.api as sm


def gls_trend_statistic(series):
    """对时间序列做 GLS 趋势回归，返回斜率 t 统计量等信息。"""
    values = np.asarray(series, dtype=float).reshape(-1)
    if values.size < 3:
        raise ValueError("series must contain at least 3 points")
    trend = np.arange(values.size, dtype=float)
    design = sm.add_constant(trend)
    result = sm.GLS(values, design).fit()
    return {
        "slope_t_abs": float(abs(result.tvalues[1])),
        "p_value": float(result.pvalues[1]),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def standardize_statistics(values) -> np.ndarray:
    """对统计量做 Z-score 标准化。"""
    arr = np.asarray(values, dtype=float).reshape(-1)
    std = float(arr.std())
    if np.isclose(std, 0.0):
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std
