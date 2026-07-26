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

## 重点验收

查看每个分组的三张图，重点确认原始/拟合差异、红色离群点位置，以及真实连续突变是否被保留。`fitting_summary.csv` 和 `method_comparison.csv` 提供总变差、最大跳变、粗糙度、高置信度偏差、峰值保持率及过冲指标。
