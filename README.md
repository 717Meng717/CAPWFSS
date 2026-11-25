# 深远风电场初步选址综合评估模型

基于 Python 的多标准决策分析项目，面向深远海风电场初步选址中的多源环境数据与工程约束问题，融合 10 项指标对 387 个候选网格进行综合评估和候选方案优选。

本仓库从原始项目中抽取并重构数据处理与评价算法核心，不包含任何原始数据、评价结果、区域边界或专题报告。



## 项目内容

- 完成风速、波高、水深、台风强度、船舶密度等多源数据的清洗、对齐与指标化处理。
- 参与构建 10 项评价指标体系，区分效益型指标与成本型指标。
- 设计 Entropy Weight-TOPSIS-IF 综合评估方法，完成指标归一化、熵权客观赋权、区域评分与候选方案优选。
- 使用 NumPy、Pandas 完成矩阵化计算，使用 Matplotlib 支持评价结果可视化。

## 方法流程

```text
多源环境与工程指标
        |
        v
数据清洗、空间对齐、缺失值处理
        |
        v
指标归一化（效益型 / 成本型）
        |
        v
熵权法计算客观权重
        |
        v
TOPSIS 正负理想解评分
        |
        v
IF 模糊信息聚合
        |
        v
387 个候选网格综合得分与排序
```

## 核心代码模块

| 模块 | 功能 |
|---|---|
| `src/normalization.py` | Min-Max 归一化、成本型指标反向归一化、熵权计算 |
| `src/entropy_topsis.py` | TOPSIS 正负理想解、距离计算、贴近度与排名 |
| `src/evaluation_pipeline.py` | 归一化、熵权、TOPSIS、排名的一体化评价流程 |
| `src/fuzzy_aggregation.py` | IF / 模糊信息处理中的三角模糊数聚合 |
| `src/grid_aggregation.py` | 多 CSV 网格评价结果逐格汇总 |
| `src/stationarity.py` | 时间序列趋势回归统计与标准化 |
| `src/score_mapping.py` | 综合得分的分级与颜色映射 |
| `examples/demo.py` | 使用合成数据演示模块化评价流程 |
| `ais_siting_evaluation.py` | 自包含单文件版本，可独立演示并读取自备数据 |


## 快速开始

### 方式一：模块化示例

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python examples/demo.py
```

### 方式二：单文件演示

```bash
python ais_siting_evaluation.py demo
```

默认合成演示使用 387 个候选网格、10 项指标，只生成随机数据，不读取任何业务文件。

## 使用自备数据

指标表要求：每行一个候选网格，每列为数值型指标。非数值列会被自动忽略。

```bash
python ais_siting_evaluation.py run ^
  --input your_indicators.csv ^
  --benefit 0,1,2 ^
  --cost 3,4,5,6,7,8,9 ^
  --output scores.csv
```

Excel 输入：

```bash
python ais_siting_evaluation.py run ^
  --input your_indicators.xlsx ^
  --sheet Sheet1 ^
  --benefit 0,1,2 ^
  --cost 3,4,5,6,7,8,9 ^
  --output scores.xlsx
```

可选模糊评价表：每行一个指标的 `(l, m, u)`，共 10 行 3 列。

```bash
python ais_siting_evaluation.py run ^
  --input your_indicators.csv ^
  --benefit 0,1,2 ^
  --cost 3,4,5,6,7,8,9 ^
  --fuzzy fuzzy_assessments.csv ^
  --output scores.csv
```

`run` 命令生成的评分文件只用于本地分析，不要提交到 GitHub。

## 代码使用示例

```python
import numpy as np
from src.evaluation_pipeline import evaluate

matrix = np.array([
    [90.0, 0.20, 12.0],
    [80.0, 0.35, 18.0],
    [70.0, 0.25, 15.0],
])

result = evaluate(
    matrix,
    benefit_indices=[0],
    cost_indices=[1, 2],
)

print(result["scores"])
print(result["weights"])
print(result["ranks"])
```

## 项目结构

```text
ais-core-github/
├── src/
│   ├── normalization.py
│   ├── entropy_topsis.py
│   ├── evaluation_pipeline.py
│   ├── fuzzy_aggregation.py
│   ├── grid_aggregation.py
│   ├── stationarity.py
│   └── score_mapping.py
├── examples/
│   └── demo.py
├── ais_siting_evaluation.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 数据保密说明

本仓库不包含：

- 原始海域环境数据
- 10 项指标 Excel 表
- 387 个候选网格评价结果
- 区域边界 JSON
- 专题报告和图表
- 任何读取真实评价结果的绘图脚本

仓库只保留数据处理、归一化、熵权、TOPSIS、IF 模糊聚合、评分排序和合成示例。


## 依赖

- Python 3.10+
- numpy
- pandas
- openpyxl
- statsmodels

