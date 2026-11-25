# -*- coding: utf-8 -*-
"""AIS comprehensive evaluation core algorithms."""
from .entropy_topsis import entropy_topsis, topsis
from .evaluation_pipeline import evaluate
from .fuzzy_aggregation import aggregate_triangular_fuzzy
from .grid_aggregation import sum_csv_grids
from .normalization import entropy_weights, min_max, normalize_matrix
from .score_mapping import score_to_color, score_to_level
from .stationarity import gls_trend_statistic, standardize_statistics

__all__ = [
    "aggregate_triangular_fuzzy",
    "entropy_topsis",
    "entropy_weights",
    "evaluate",
    "gls_trend_statistic",
    "min_max",
    "normalize_matrix",
    "score_to_color",
    "score_to_level",
    "standardize_statistics",
    "sum_csv_grids",
    "topsis",
]
