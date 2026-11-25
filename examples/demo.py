# -*- coding: utf-8 -*-
"""使用合成数据演示 AIS 综合评估核心算法。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation_pipeline import evaluate
from src.fuzzy_aggregation import aggregate_triangular_fuzzy
from src.score_mapping import score_to_color, score_to_level
from src.stationarity import gls_trend_statistic, standardize_statistics


def main() -> None:
    # 合成指标矩阵：每行一个候选区域，每列一个指标
    # 指标 0 为效益型；指标 1、2 为成本型
    matrix = np.array(
        [
            [90.0, 0.20, 12.0],
            [80.0, 0.35, 18.0],
            [70.0, 0.25, 15.0],
            [60.0, 0.40, 20.0],
        ]
    )
    result = evaluate(
        matrix,
        benefit_indices=[0],
        cost_indices=[1, 2],
    )

    print("weights:", np.round(result["weights"], 4))
    for index, (score, rank) in enumerate(
        zip(result["scores"], result["ranks"]),
        start=1,
    ):
        print(
            f"candidate={index}, score={score:.4f}, rank={int(rank)}, "
            f"level={score_to_level(score)}, color={score_to_color(score)}"
        )

    assessments = np.array(
        [
            [5, 7, 9],
            [3, 5, 7],
            [7, 9, 9],
        ],
        dtype=float,
    )
    print("fuzzy_aggregate:", aggregate_triangular_fuzzy(assessments))

    series = np.array([2.1, 2.4, 2.2, 2.8, 3.0, 3.2], dtype=float)
    print("trend_statistic:", gls_trend_statistic(series))
    print(
        "standardized_statistics:",
        np.round(standardize_statistics([1.2, 1.8, 2.1, 2.4]), 4),
    )


if __name__ == "__main__":
    main()
