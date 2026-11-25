# -*- coding: utf-8 -*-
"""深远风电场初步选址综合评估模型（公开算法版）。

项目背景：
    2025.07 - 2025.10，基于 Python 构建多标准决策分析模型，
    融合 10 项环境与工程指标，对 387 个候选网格进行综合评估。

核心方法：
    Entropy Weight - TOPSIS - IF
    1. 指标归一化（效益型 / 成本型）
    2. 熵权法客观赋权
    3. TOPSIS 正负理想解评分
    4. 三角模糊数聚合，作为 IF 模糊信息处理示例
    5. 候选区域评分、排名和等级划分

保密说明：
    本文件不包含任何原始数据、评价结果、区域边界或报告内容。
    默认示例使用纯合成数据，仅用于演示算法。

运行示例：
    python ais_siting_evaluation.py demo
    python ais_siting_evaluation.py demo --candidates 387 --indicators 10
    python ais_siting_evaluation.py run --input indicators.csv --benefit 0,1,2 --cost 3,4,5,6,7,8,9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


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


def min_max_normalize(values, reverse: bool = False) -> np.ndarray:
    """Min-Max 归一化；reverse=True 用于成本型指标。"""
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


def normalize_matrix(matrix, cost_indices: Sequence[int] = ()) -> np.ndarray:
    """按列归一化矩阵，自动区分效益型与成本型指标。"""
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2:
        raise ValueError("matrix must be 2-D")
    cost = set(int(index) for index in cost_indices)
    normalized = np.empty_like(data, dtype=float)
    for column in range(data.shape[1]):
        normalized[:, column] = min_max_normalize(
            data[:, column],
            reverse=column in cost,
        )
    return normalized


def entropy_weights(matrix, cost_indices: Sequence[int] = (), eps: float = 1e-12):
    """熵权法：返回归一化矩阵、熵值和权重。"""
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2:
        raise ValueError("matrix must be 2-D")
    normalized = normalize_matrix(data, cost_indices)
    sample_count, indicator_count = normalized.shape
    if sample_count == 0 or indicator_count == 0:
        raise ValueError("matrix must not be empty")

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

    # 常量指标不提供区分度，权重置零
    ranges = data.max(axis=0) - data.min(axis=0)
    divergence[np.isclose(ranges, 0.0)] = 0.0
    total = float(divergence.sum())
    if np.isclose(total, 0.0):
        weights = np.full(indicator_count, 1.0 / indicator_count)
    else:
        weights = divergence / total
    return normalized, weights


def fuzzy_centroid_weights(fuzzy_assessments) -> np.ndarray:
    """由三角模糊评价 (l, m, u) 计算模糊权重。"""
    arr = np.asarray(fuzzy_assessments, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("fuzzy_assessments must have shape (n, 3)")
    if arr.shape[0] == 0:
        raise ValueError("fuzzy_assessments must not be empty")
    centroids = arr.mean(axis=1)
    total = float(centroids.sum())
    if np.isclose(total, 0.0):
        return np.full(arr.shape[0], 1.0 / arr.shape[0])
    return centroids / total


def aggregate_triangular_fuzzy(assessments, round_mid: bool = False):
    """聚合多个三角模糊评价：左端最小、中值平均、右端最大。"""
    arr = np.asarray(assessments, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("assessments must have shape (n, 3)")
    if arr.shape[0] == 0:
        raise ValueError("assessments must not be empty")
    middle = float(arr[:, 1].mean())
    if round_mid:
        middle = float(np.ceil(middle))
    return np.array(
        [float(arr[:, 0].min()), middle, float(arr[:, 2].max())],
        dtype=float,
    )


def topsis(matrix, weights, benefit_indices, cost_indices):
    """标准 TOPSIS，返回贴近度、正负距离和加权矩阵。"""
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


def rank_scores(scores) -> np.ndarray:
    """返回每个候选区域的排名，最高分排名为 1。"""
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, scores.size + 1)
    return ranks


def score_to_level(score: float) -> str:
    """返回综合得分等级。"""
    value = float(score)
    for lower, upper, level, _color in SCORE_LEVELS:
        if lower <= value < upper:
            return level
    return "out_of_range"


def score_to_color(score: float) -> str:
    """返回综合得分对应的可视化颜色。"""
    value = float(score)
    for lower, upper, _level, color in SCORE_LEVELS:
        if lower <= value < upper:
            return color
    return "#808080"


def evaluate_model(
    matrix,
    benefit_indices: Sequence[int],
    cost_indices: Sequence[int],
    fuzzy_assessments=None,
):
    """执行 熵权 - TOPSIS - IF 综合评估。

    fuzzy_assessments 可选，形状为 (指标数, 3)，每行是 (l, m, u)。
    提供时将熵权与模糊权重按 50% / 50% 融合。
    """
    data = np.asarray(matrix, dtype=float)
    normalized, entropy_weight = entropy_weights(data, cost_indices)
    final_weights = entropy_weight
    fuzzy_weight = None

    if fuzzy_assessments is not None:
        fuzzy_weight = fuzzy_centroid_weights(fuzzy_assessments)
        if fuzzy_weight.size != data.shape[1]:
            raise ValueError("fuzzy assessments count must match indicator count")
        final_weights = 0.5 * entropy_weight + 0.5 * fuzzy_weight
        final_weights = final_weights / final_weights.sum()

    scores, positive_distance, negative_distance, weighted = topsis(
        data,
        final_weights,
        benefit_indices,
        cost_indices,
    )
    ranks = rank_scores(scores)
    levels = [score_to_level(score) for score in scores]
    return {
        "scores": scores,
        "ranks": ranks,
        "levels": levels,
        "normalized": normalized,
        "entropy_weights": entropy_weight,
        "fuzzy_weights": fuzzy_weight,
        "weights": final_weights,
        "positive_distance": positive_distance,
        "negative_distance": negative_distance,
        "weighted": weighted,
    }


def load_numeric_table(path, sheet_name=None):
    """读取用户自备的 CSV/Excel 数值指标表，不在仓库内保存数据。"""
    file_path = Path(path)
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        frame = pd.read_csv(file_path)
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError("no numeric indicator columns found")
    return numeric.to_numpy(dtype=float), list(numeric.columns)


def save_results(path, scores, ranks, levels):
    """将评分结果保存到本地；结果文件不应上传公开仓库。"""
    frame = pd.DataFrame(
        {
            "score": np.asarray(scores, dtype=float),
            "rank": np.asarray(ranks, dtype=int),
            "level": list(levels),
        }
    )
    file_path = Path(path)
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        frame.to_excel(file_path, index=False)
    else:
        frame.to_csv(file_path, index=False, encoding="utf-8-sig")


def run_synthetic_demo(candidates: int = 387, indicators: int = 10, seed: int = 42):
    """生成合成指标矩阵并演示完整评价流程。"""
    if candidates <= 0 or indicators < 2:
        raise ValueError("candidates must be positive and indicators >= 2")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.5, 0.18, size=(candidates, indicators))
    matrix = np.clip(matrix, 0.0, 1.0)
    matrix = matrix * np.linspace(0.6, 1.4, indicators)

    benefit_count = min(3, indicators - 1)
    benefit_indices = list(range(benefit_count))
    cost_indices = list(range(benefit_count, indicators))

    fuzzy_assessments = np.sort(
        rng.integers(1, 10, size=(indicators, 3)),
        axis=1,
    )
    result = evaluate_model(
        matrix,
        benefit_indices,
        cost_indices,
        fuzzy_assessments=fuzzy_assessments,
    )

    top_indices = np.argsort(result["ranks"])[:10]
    rows = []
    for index in top_indices:
        score = float(result["scores"][index])
        rows.append(
            {
                "candidate": int(index) + 1,
                "score": round(score, 4),
                "rank": int(result["ranks"][index]),
                "level": result["levels"][index],
                "color": score_to_color(score),
            }
        )

    return {
        "synthetic_data": True,
        "candidates": candidates,
        "indicators": indicators,
        "method": "Entropy Weight - TOPSIS - IF",
        "weights": np.round(result["weights"], 4).tolist(),
        "top10": rows,
    }


def parse_indices(text: str):
    """解析逗号分隔的指标索引。"""
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("indices must not be empty")
    return [int(item) for item in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="深远风电场初步选址综合评估模型"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="使用纯合成数据演示")
    demo.add_argument("--candidates", type=int, default=387)
    demo.add_argument("--indicators", type=int, default=10)
    demo.add_argument("--seed", type=int, default=42)

    run = sub.add_parser("run", help="使用自备数值指标表运行")
    run.add_argument("--input", required=True)
    run.add_argument("--sheet", default=None)
    run.add_argument("--benefit", required=True, help="效益型指标索引，逗号分隔")
    run.add_argument("--cost", required=True, help="成本型指标索引，逗号分隔")
    run.add_argument("--fuzzy", default=None, help="可选：三角模糊评价表，每行 l,m,u")
    run.add_argument("--output", default="scores.csv")

    args = parser.parse_args()

    if args.command == "demo":
        summary = run_synthetic_demo(
            candidates=args.candidates,
            indicators=args.indicators,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    matrix, columns = load_numeric_table(args.input, args.sheet)
    fuzzy = None
    if args.fuzzy:
        fuzzy, _ = load_numeric_table(args.fuzzy)
        fuzzy = fuzzy[:, :3]
    result = evaluate_model(
        matrix,
        parse_indices(args.benefit),
        parse_indices(args.cost),
        fuzzy_assessments=fuzzy,
    )
    save_results(
        args.output,
        result["scores"],
        result["ranks"],
        result["levels"],
    )
    print(
        json.dumps(
            {
                "input": args.input,
                "columns": columns,
                "rows": int(matrix.shape[0]),
                "indicators": int(matrix.shape[1]),
                "weights": np.round(result["weights"], 4).tolist(),
                "output": args.output,
                "top10_ranks": result["ranks"][:10].astype(int).tolist(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
