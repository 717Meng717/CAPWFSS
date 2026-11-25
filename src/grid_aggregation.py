# -*- coding: utf-8 -*-
"""网格 CSV 数据汇总。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def sum_csv_grids(csv_paths, columns: int = 3):
    """逐文件读取前若干列并逐格累加。

    原始脚本使用固定桌面路径，本实现改为显式传入文件列表，
    避免在公开代码中泄露本地目录和业务数据位置。
    """
    paths = [Path(path) for path in csv_paths]
    if not paths:
        raise ValueError("csv_paths must not be empty")
    if columns <= 0:
        raise ValueError("columns must be positive")

    total = None
    for path in paths:
        frame = pd.read_csv(path)
        if frame.shape[1] < columns:
            raise ValueError(f"{path} has fewer than {columns} columns")
        data = (
            frame.iloc[:, :columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        total = data if total is None else total.add(data, fill_value=0.0)
    return total
