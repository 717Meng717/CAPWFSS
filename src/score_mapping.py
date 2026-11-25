# -*- coding: utf-8 -*-
"""评价分数分级与颜色映射。"""
from __future__ import annotations

# (下界, 上界, 等级, 颜色)
SCORE_LEVELS = (
    (0.0, 0.1, "very_low", "#ff0000"),
    (0.1, 0.2, "low", "#ff4e00"),
    (0.2, 0.3, "low_medium", "#ff8400"),
    (0.3, 0.4, "medium_low", "#ffa200"),
    (0.4, 0.5, "medium", "#ffc000"),
    (0.5, 0.6, "medium_high", "#ffea00"),
    (0.6, 0.7, "high", "#f6ff00"),
    (0.7, 0.8, "very_high", "#ccff00"),
    (0.8, 0.9, "excellent", "#7eff00"),
    (0.9, 1.0000001, "best", "#30ff00"),
)


def _matched_level(score):
    value = float(score)
    for lower, upper, level, color in SCORE_LEVELS:
        if lower <= value < upper:
            return level, color
    return "out_of_range", "#808080"


def score_to_level(score: float) -> str:
    """返回分数等级。"""
    return _matched_level(score)[0]


def score_to_color(score: float) -> str:
    """返回分数对应颜色。"""
    return _matched_level(score)[1]
