# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

模型反演曲线拟合模块 — 读取 `all_model_predictions.csv`，按 `model/source_file/experiment_id` 分组，对损伤等级和应力预测进行离群点检测、修正、曲线平滑与致密插值。默认组合为 **Hampel + Savitzky-Golay + PCHIP**。

## Commands

```bash
# 安装依赖
pip install -r requirements.txt

# 先跑一个分组做快速验证
python run_fitting.py --one-group

# 全量运行
python run_fitting.py

# 跳过出图（仅算指标）
python run_fitting.py --no-plots

# 五组算法横向比较（Hampel+PCHIP, Hampel+Savgol, Median+PCHIP, LOWESS, CubicSpline）
python compare_methods.py

# 自定义配置文件
python run_fitting.py --config my_config.yaml
```

输出位于 `outputs/fitting/`，日志写入 `fitting.log`。

## Code Architecture

The pipeline is **linear and modular**, driven by `run_fitting.py:process()`:

```
config.yaml
    │
    ▼
data_loader.py    ── 加载 CSV，字段校验，深度排序，缺失/无穷插值
    │
    ▼
outlier_detection.py ── 检测（Hampel / IQR / median / jump）+ 修正（interpolate / median / confidence_weight）
    │
    ▼
fitting_methods.py    ── 曲线拟合（savgol / lowess / pchip / cubicspline）+ 置信度融合 + 范围约束
    │
    ▼
evaluation.py         ── 总变差、最大跳变、粗糙度、高置信度MAE、峰值保持率、过冲标记
    │
    ▼
visualization.py      ── 三组图（损伤/应力对比 + 离群位置），输出到 outputs/fitting/plots/
```

### Key Design Decisions

- **分组聚合**: 按 `[model, source_file, experiment_id]` 分组，每组独立拟合；相同深度按中位数聚合后再拟合，结果映射回全部原始行。
- **离群保护**: 连续 ≥3 个异常值视为真实边界，不做修正（`outlier_detection.py:30-35`）。
- **置信度融合**: 高置信度点拟合融合度弱（`blend=0.2`），低置信度点融合度强（`blend=0.8`）；缺失置信度按 0.5 中性值处理（`fitting_methods.py:20-23`）。
- **范围约束**: 损伤限制在原组 `[min, max]`；应力默认非负且抑制过冲（`fitting_methods.py:25-29`）。
- **最终流程固定**: `Hampel异常检测 → 线性插值修正 → Savitzky-Golay平滑 → PCHIP致密插值`（`run_fitting.py:49-63`）。
- **仅使用累计孔深**: 目前没有 `collar_x/y/z` / `azimuth_deg` / `dip_deg`，输出只沿 `cumulative_depth_cm`（或回退到 `depth_cm`），不生成空间坐标。

### Module Breakdown

| File | Responsibility |
|---|---|
| `data_loader.py` | CSV加载、字段校验、缺失预测值组内插值、深度排序、重复深度中位数聚合 |
| `outlier_detection.py` | 4种检测方法（hampel/iqr/median/jump）+ 3种修正（interpolate/median/confidence_weight） |
| `fitting_methods.py` | 4种拟合方法（savgol/lowess/pchip/cubicspline）、置信度自适应融合、范围约束、致密网格PCHIP插值 |
| `evaluation.py` | 6项指标计算（总变差/最大跳变/粗糙度/高置信度MAE/峰值保持率/过冲） |
| `visualization.py` | MPL出图（Agg后端），原始/拟合/PCHIP连续曲线叠绘，离群点红色标注 |
| `run_fitting.py` | 主入口：编排完整pipeline，产出 `fitted_predictions.csv`、`fitting_summary.csv`、`fitted_curve_dense.csv` |
| `compare_methods.py` | 5种算法组合对比，产出 `method_comparison.csv` + `algorithm_comparison_summary.csv` |

### Config (`config.yaml`)

- `outlier`: 检测方法、窗口、阈值、修正策略
- `fitting`: 拟合方法、窗口/阶数、置信度阈值、融合系数、约束参数
- `evaluation`: 高置信度阈值、峰值分位数、跳变容忍
- `plot`: DPI、最大采样点数、中文字体候选项

### Input Data Requirements

CSV 文件必须包含：`model`, `source_file`, `experiment_id`, `sample_index`, `pred_damage_level`, `pred_stress_mpa`。
`state_confidence`（可选，缺失则按 0.5 中性值处理）、`cumulative_depth_cm` 或 `depth_cm`（至少一个）。
