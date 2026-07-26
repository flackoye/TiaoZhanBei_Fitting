# 模型反演曲线拟合模块

该模块读取 `all_model_predictions.csv`，按 `model/source_file/experiment_id` 分组，以累计孔深优先排序，检测并修正孤立离群点，再生成连续、受范围约束的损伤和应力曲线。默认组合为 **Hampel + Savitzky-Golay**。

## 安装与运行

```powershell
python -m pip install -r requirements.txt
python run_fitting.py --one-group   # 先做一个分组的小测
python run_fitting.py               # 全量运行
python compare_methods.py           # 五组算法比较
```

输出位于 `outputs/fitting/`。日志写入 `fitting.log`。原始 CSV 只读，不会修改。

`fitted_predictions.csv` 保存原始采样位置上的原始值、Hampel修正值和Savitzky-Golay平滑值；`fitted_curve_dense.csv` 单独保存由平滑结果经PCHIP生成的等间距最终连续曲线。默认网格步长为0.1 cm，可通过 `fitting.output_depth_step` 改为0.5 cm等正数。

最终流程固定为：`Hampel异常检测 → 相邻有效点线性插值 → Savitzky-Golay平滑 → PCHIP致密插值`。当前没有 `collar_x/collar_y/collar_z/azimuth_deg/dip_deg`，因此输出只使用沿孔累计进尺，不生成或推测空间坐标。

## 设计说明

- 优先使用 `cumulative_depth_cm`；若不存在则回退到 `depth_cm`。
- 重复孔深按中位数聚合后拟合，缺失/无穷预测值按组插值并记录日志。
- Hampel、局部 IQR、中值和一阶差分检测均可在 `config.yaml` 切换。
- 异常点可使用邻域中位数、线性插值或置信度降权修正。
- 连续异常段不会按孤立离群点处理，以保护真实边界和连续突变。
- 高置信度点采用较弱融合，低置信度点采用较强融合；缺失置信度按 0.5 中性值处理。
- 损伤限制在原组范围，应力限制为非负并抑制插值过冲。

## 不确定性分析

运行 `python run_uncertainty.py` 对多模型预测进行不确定性估计，输出至 `outputs/uncertainty/`。

### 两种不确定性方法

| 方法 | 原理 | 是否需要训练 |
|---|---|---|
| **模型分歧 (disagreement)** | 同一位置多模型预测的 `ensemble_std`，分歧越大不确定性越高 | 无需训练，直接计算 |
| **残差校准 (calibrated)** | `HistGradientBoostingRegressor`（分位数 0.9）拟合 `ensemble_std` 等特征 → 预测绝对误差的 90 分位 | 需要训练（留一文件交叉验证） |
| MC Dropout | 无模型权重文件，**不可用** | — |

### 方法选型

对 damage / stress 分别独立选型：

1. **留一文件交叉验证**：每个 `source_file` 轮流做测试集，其余文件训练残差校准器
2. **AURC**（Risk-Coverage 曲线下面积）越小越好 — 主排序指标
3. **Spearman 相关** 越大越好 — 辅助排序
4. 选中的方法用全部数据重拟合，输出最终不确定性

### 输出文件

| 文件 | 内容 |
|---|---|
| `point_uncertainty.csv` | 每行一个预测点的 uncertainty / confidence / weight |
| `method_evaluation.csv` | 每个折×方法 的 AURC 和 Spearman |
| `selected_methods.json` | damage / stress 各自选中的方法名 |
| `risk_coverage_curves.csv` | 选中方法的风险-覆盖率曲线数据 |

### 输出字段

- `{damage,stress}_uncertainty` — `[0, 1]`，越大越不确定
- `{damage,stress}_confidence` — `[0, 1]`，越大越可信
- `{damage,stress}_weight` — `[0.1, 1]`，越大越可信
- `{damage,stress}_uncertainty_method` — 该 target 选中的方法名

## 不确定性权重集成

拟合模块支持从 `point_uncertainty.csv` 加载不确定性权重，替代原有的 `state_confidence` 用于损伤和应力的拟合融合。

### 启用方式

在 `config.yaml` 中设置不确定性文件路径：

```yaml
uncertainty:
  file: outputs/uncertainty/point_uncertainty.csv   # 设为 null 则使用 state_confidence（原行为）
  join_columns: [model, source_file, experiment_id, sample_index]
  minimum_weight: 0.1
```

### 权重作用

- **damage_weight** — 作为损伤拟合的置信度数组，用于离群修正和曲线融合
- **stress_weight** — 作为应力拟合的置信度数组，用于离群修正和曲线融合
- 权重接入后使用 **连续融合**（`blend = low - weight * (low - high)`），不再按 `confidence_threshold` 二值分档
  - 权重 1.0 → blend = 0.2（最弱平滑，保留原始特征）
  - 权重 0.1 → blend = 0.74（最强平滑，依赖拟合曲线）
- 未配置 `uncertainty.file` 时，保持原有 `state_confidence` 和 `confidence_threshold` 二值逻辑，完全向后兼容

### 权重加载流程

1. `load_predictions()` 检测 `uncertainty.file` 是否非空
2. 若非空，加载指定 CSV 并校验必需字段：`model`, `source_file`, `experiment_id`, `sample_index`, `damage_weight`, `stress_weight`
3. 校验权重值在 [0, 1] 范围内
4. 以 `many_to_one` 左连接方式合并到主 DataFrame
5. `run_fitting.py` 根据列名 `damage_weight` / `stress_weight` 的存否自动切换权重路径

### 重要说明

- 离群检测 mask 不受权重影响，仅修改 `confidence_weight` 修正和拟合融合权重
- 权重与 `state_confidence` 同时保留在 DataFrame 中，不会互相覆盖
- 权重列不存在时自动退化到 `state_confidence` 路径

## 重点验收

查看每个分组的三张图，重点确认原始/拟合差异、红色离群点位置，以及真实连续突变是否被保留。`fitting_summary.csv` 和 `method_comparison.csv` 提供总变差、最大跳变、粗糙度、高置信度偏差、峰值保持率及过冲指标。
