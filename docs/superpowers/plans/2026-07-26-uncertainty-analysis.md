# 不确定性分析与权重输出 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于现有多模型反演 CSV，比较“异构模型分歧”和“残差校准融合”两种不确定性方法，以留一文件 AURC 选出最终方法并生成后续离群修正与拟合可直接使用的点级权重。

**Architecture:** 新建独立 `uncertainty_analysis` 包，只读 `all_model_predictions.csv`，不改动现有拟合流程。先将同一采样位置的多模型预测对齐，再分别计算模型分歧和交叉拟合的残差校准不确定性；统一用留一 `source_file` 评估，按 damage/stress 分别选取平均 AURC 最小的方法，最后输出归一化置信度与下游权重。MC Dropout 只定义兼容接口和状态说明，在没有模型权重时不得模拟或参与排名。

**Tech Stack:** Python 3.10+、pandas、NumPy、SciPy、scikit-learn、matplotlib、PyYAML、pytest。

## Global Constraints

- 输入主键为 `source_file, experiment_id, sample_index`；深度字段优先 `cumulative_depth_cm`，回退 `depth_cm`。
- 评估必须按 `source_file` 留一，测试文件的真值不得参与该折的校准、归一化和方法选择。
- damage 与 stress 独立分析、独立选型、独立生成权重。
- `uncertainty` 取值为 `[0, 1]`，越大越不确定；`confidence` 和 `fitting_weight` 取值为 `[0.1, 1]`，越大越可信。
- 三模型覆盖不足的位置，模型分歧记为缺失，并由残差校准法的其他特征给出结果；不得把缺失分歧填成零。
- MC Dropout 状态固定为 `unavailable_no_model_artifacts`，直至取得模型结构、权重和推理输入。
- 原有 `all_model_predictions.csv` 和拟合输出均只读。

---

## File Structure

- Create `uncertainty_analysis/__init__.py`: 公共接口导出。
- Create `uncertainty_analysis/data.py`: 读取、字段校验、主键对齐和特征表构建。
- Create `uncertainty_analysis/methods.py`: 模型分歧、残差校准及 MC Dropout 预留接口。
- Create `uncertainty_analysis/evaluation.py`: 风险—覆盖率曲线、AURC、Spearman 和折间汇总。
- Create `uncertainty_analysis/pipeline.py`: 留一文件交叉验证、方法选择、全量权重生成。
- Create `run_uncertainty.py`: CLI 入口和 CSV/JSON 输出。
- Create `uncertainty_config.yaml`: 字段、分组、随机种子、权重下限与输出目录。
- Create `tests/test_uncertainty_data.py`: 数据对齐与缺失覆盖测试。
- Create `tests/test_uncertainty_methods.py`: 两种方法及 MC Dropout 状态测试。
- Create `tests/test_uncertainty_evaluation.py`: AURC 与无泄漏测试。
- Create `tests/test_uncertainty_pipeline.py`: 端到端选型和输出契约测试。
- Modify `requirements.txt`: 增加 scikit-learn 与 pytest。
- Modify `README.md`: 增加运行命令、输出字段和下游接入说明。

---

### Task 1: 数据契约与多模型对齐

**Files:**
- Create: `uncertainty_analysis/__init__.py`
- Create: `uncertainty_analysis/data.py`
- Test: `tests/test_uncertainty_data.py`

**Interfaces:**
- Consumes: 原始预测 DataFrame。
- Produces: `validate_input(df) -> None`；`build_position_table(df, target) -> pd.DataFrame`，其中 target 为 `damage` 或 `stress`。

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pandas as pd
import pytest
from uncertainty_analysis.data import build_position_table, validate_input

def sample_rows():
    return pd.DataFrame({
        "source_file": ["a.csv"] * 3,
        "experiment_id": ["A"] * 3,
        "sample_index": [1] * 3,
        "cumulative_depth_cm": [1.0] * 3,
        "segment_index": [0] * 3,
        "model": ["m1", "m2", "m3"],
        "pred_damage_level": [20.0, 40.0, 60.0],
        "pred_stress_mpa": [8.0, 10.0, 12.0],
        "true_damage_level": [40.0] * 3,
        "true_stress_mpa": [10.0] * 3,
        "state_confidence": [0.8, np.nan, 0.6],
    })

def test_position_table_aligns_three_models_without_losing_truth():
    table = build_position_table(sample_rows(), "damage")
    assert len(table) == 1
    assert table.loc[0, "ensemble_count"] == 3
    assert table.loc[0, "true_value"] == 40.0
    assert table.loc[0, "ensemble_std"] == pytest.approx(20.0)

def test_validation_rejects_missing_primary_key():
    with pytest.raises(ValueError, match="sample_index"):
        validate_input(sample_rows().drop(columns="sample_index"))
```

- [ ] **Step 2: 验证测试失败**

Run: `python -m pytest tests/test_uncertainty_data.py -v`

Expected: FAIL with `ModuleNotFoundError: uncertainty_analysis`.

- [ ] **Step 3: 实现最小数据接口**

```python
# uncertainty_analysis/data.py
import numpy as np
import pandas as pd

KEY = ["source_file", "experiment_id", "sample_index"]
REQUIRED = KEY + ["model", "pred_damage_level", "pred_stress_mpa",
                  "true_damage_level", "true_stress_mpa"]

def validate_input(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")

def build_position_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    validate_input(df)
    pred = {"damage": "pred_damage_level", "stress": "pred_stress_mpa"}[target]
    truth = {"damage": "true_damage_level", "stress": "true_stress_mpa"}[target]
    depth = "cumulative_depth_cm" if "cumulative_depth_cm" in df else "depth_cm"
    grouped = df.groupby(KEY, sort=False, dropna=False)
    table = grouped.agg(
        depth_cm=(depth, "median"), segment_index=("segment_index", "first"),
        true_value=(truth, "median"), ensemble_mean=(pred, "mean"),
        ensemble_median=(pred, "median"), ensemble_std=(pred, "std"),
        ensemble_range=(pred, lambda values: values.max() - values.min()),
        ensemble_count=("model", "nunique"),
        mean_state_confidence=("state_confidence", "mean"),
    ).reset_index()
    table.loc[table["ensemble_count"] < 2, ["ensemble_std", "ensemble_range"]] = np.nan
    return table
```

- [ ] **Step 4: 运行测试并提交**

Run: `python -m pytest tests/test_uncertainty_data.py -v`

Expected: 2 passed.

Commit: `git commit -am "feat: add uncertainty data contract"`

---

### Task 2: 两种当前可运行的不确定性方法和 MC Dropout 接口

**Files:**
- Create: `uncertainty_analysis/methods.py`
- Test: `tests/test_uncertainty_methods.py`

**Interfaces:**
- Consumes: Task 1 的位置表；训练折和测试折。
- Produces: `ensemble_disagreement(table) -> np.ndarray`；`fit_residual_calibrator(train, random_state) -> estimator`；`predict_calibrated_uncertainty(estimator, table) -> np.ndarray`；`mc_dropout_status() -> dict`。

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pandas as pd
from uncertainty_analysis.methods import ensemble_disagreement, mc_dropout_status

def test_disagreement_is_monotonic_and_preserves_missing():
    table = pd.DataFrame({"ensemble_std": [0.0, 2.0, np.nan]})
    score = ensemble_disagreement(table)
    assert score[0] < score[1]
    assert np.isnan(score[2])

def test_mc_dropout_is_explicitly_unavailable():
    status = mc_dropout_status()
    assert status == {"available": False, "reason": "unavailable_no_model_artifacts"}
```

- [ ] **Step 2: 验证测试失败**

Run: `python -m pytest tests/test_uncertainty_methods.py -v`

Expected: FAIL because the methods do not exist.

- [ ] **Step 3: 实现模型分歧和残差校准**

```python
# uncertainty_analysis/methods.py
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["ensemble_std", "ensemble_range", "mean_state_confidence",
            "depth_cm", "segment_index"]

def ensemble_disagreement(table):
    return table["ensemble_std"].to_numpy(float)

def fit_residual_calibrator(train, random_state=42):
    numeric = ["ensemble_std", "ensemble_range", "mean_state_confidence", "depth_cm"]
    categorical = ["segment_index"]
    transform = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    model = HistGradientBoostingRegressor(loss="quantile", quantile=0.9,
                                          max_iter=100, random_state=random_state)
    estimator = Pipeline([("features", transform), ("model", model)])
    absolute_error = np.abs(train["ensemble_median"] - train["true_value"])
    return estimator.fit(train[FEATURES], absolute_error)

def predict_calibrated_uncertainty(estimator, table):
    return np.maximum(0.0, estimator.predict(table[FEATURES]))

def mc_dropout_status():
    return {"available": False, "reason": "unavailable_no_model_artifacts"}
```

- [ ] **Step 4: 增加残差校准器测试、运行并提交**

Add a synthetic test where larger `ensemble_std` has larger absolute error and assert the calibrated prediction for high disagreement exceeds low disagreement.

Run: `python -m pytest tests/test_uncertainty_methods.py -v`

Expected: all passed.

Commit: `git commit -am "feat: add uncertainty estimation methods"`

---

### Task 3: AURC 评估与无泄漏留一文件验证

**Files:**
- Create: `uncertainty_analysis/evaluation.py`
- Test: `tests/test_uncertainty_evaluation.py`

**Interfaces:**
- Consumes: 绝对误差和同长度不确定性数组。
- Produces: `risk_coverage(errors, uncertainty) -> pd.DataFrame`；`aurc(errors, uncertainty) -> float`；`spearman_error_correlation(errors, uncertainty) -> float`。

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
from uncertainty_analysis.evaluation import aurc, spearman_error_correlation

def test_perfect_ranking_beats_reversed_ranking():
    errors = np.array([0.0, 1.0, 2.0, 10.0])
    assert aurc(errors, errors) < aurc(errors, -errors)
    assert spearman_error_correlation(errors, errors) == 1.0

def test_metrics_ignore_nonfinite_pairs():
    assert np.isfinite(aurc(np.array([1., 2., np.nan]), np.array([.1, .2, .3])))
```

- [ ] **Step 2: 验证测试失败**

Run: `python -m pytest tests/test_uncertainty_evaluation.py -v`

Expected: FAIL because evaluation functions do not exist.

- [ ] **Step 3: 实现风险—覆盖率与 AURC**

```python
# uncertainty_analysis/evaluation.py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def risk_coverage(errors, uncertainty):
    error = np.asarray(errors, float)
    score = np.asarray(uncertainty, float)
    valid = np.isfinite(error) & np.isfinite(score)
    error, score = error[valid], score[valid]
    order = np.argsort(score)
    sorted_error = error[order]
    count = np.arange(1, len(sorted_error) + 1)
    return pd.DataFrame({"coverage": count / len(sorted_error),
                         "risk_mae": np.cumsum(sorted_error) / count})

def aurc(errors, uncertainty):
    curve = risk_coverage(errors, uncertainty)
    return float(np.trapezoid(curve["risk_mae"], curve["coverage"]))

def spearman_error_correlation(errors, uncertainty):
    error = np.asarray(errors, float)
    score = np.asarray(uncertainty, float)
    valid = np.isfinite(error) & np.isfinite(score)
    return float(spearmanr(error[valid], score[valid]).statistic)
```

- [ ] **Step 4: 运行测试并提交**

Run: `python -m pytest tests/test_uncertainty_evaluation.py -v`

Expected: 2 passed.

Commit: `git commit -am "feat: add uncertainty ranking metrics"`

---

### Task 4: 交叉验证、方法选型与权重转换

**Files:**
- Create: `uncertainty_analysis/pipeline.py`
- Test: `tests/test_uncertainty_pipeline.py`

**Interfaces:**
- Consumes: Task 1 位置表和配置。
- Produces: `cross_validate_methods(table, random_state) -> pd.DataFrame`；`select_method(metrics) -> str`；`uncertainty_to_weight(train_scores, test_scores, minimum=0.1) -> tuple[np.ndarray, np.ndarray]`。

- [ ] **Step 1: 写失败测试，锁定选择规则和权重方向**

```python
import numpy as np
import pandas as pd
from uncertainty_analysis.pipeline import select_method, uncertainty_to_weight

def test_select_method_uses_mean_aurc_then_spearman_tiebreak():
    metrics = pd.DataFrame({
        "method": ["a", "a", "b", "b"],
        "aurc": [1.0, 1.2, 0.8, 0.9],
        "spearman": [.5, .4, .3, .3],
    })
    assert select_method(metrics) == "b"

def test_higher_uncertainty_gets_lower_weight():
    confidence, weight = uncertainty_to_weight(
        np.array([0., 1., 2., 3.]), np.array([0., 3.]), minimum=.1)
    assert confidence[0] > confidence[1]
    assert weight[0] > weight[1]
    assert np.all((weight >= .1) & (weight <= 1.0))
```

- [ ] **Step 2: 验证测试失败**

Run: `python -m pytest tests/test_uncertainty_pipeline.py -v`

Expected: FAIL because pipeline functions do not exist.

- [ ] **Step 3: 实现留一文件验证和经验分位数权重**

`cross_validate_methods` 对每个测试 `source_file`：仅用其他文件拟合残差校准器；模型分歧直接计算；两种方法均用测试折 `abs(ensemble_median - true_value)` 计算 AURC 和 Spearman。`select_method` 按平均 AURC 升序、平均 Spearman 降序、方法名升序确定唯一结果。

```python
def uncertainty_to_weight(train_scores, test_scores, minimum=0.1):
    train = np.sort(np.asarray(train_scores, float))
    test = np.asarray(test_scores, float)
    percentile = np.searchsorted(train, test, side="right") / len(train)
    confidence = np.clip(1.0 - percentile, 0.0, 1.0)
    weight = minimum + (1.0 - minimum) * confidence
    return confidence, weight
```

- [ ] **Step 4: 增加泄漏防护测试**

Monkeypatch `fit_residual_calibrator`，记录每折训练数据的 `source_file`，断言测试文件从未出现在该折训练集合中。

- [ ] **Step 5: 运行测试并提交**

Run: `python -m pytest tests/test_uncertainty_pipeline.py -v`

Expected: all passed.

Commit: `git commit -am "feat: select uncertainty method by leave-one-file-out AURC"`

---

### Task 5: CLI、输出契约与配置

**Files:**
- Create: `run_uncertainty.py`
- Create: `uncertainty_config.yaml`
- Modify: `requirements.txt`
- Test: `tests/test_uncertainty_pipeline.py`

**Interfaces:**
- Consumes: `python run_uncertainty.py --config uncertainty_config.yaml`。
- Produces: `outputs/uncertainty/point_uncertainty.csv`、`method_evaluation.csv`、`selected_methods.json`、`risk_coverage_curves.csv`。

- [ ] **Step 1: 写端到端失败测试**

构造含三个 `source_file`、每处三个模型的小 CSV，运行 CLI 后断言四个文件存在，并检查点级输出至少包含：

```python
EXPECTED = {
    "model", "source_file", "experiment_id", "sample_index",
    "cumulative_depth_cm", "damage_uncertainty", "stress_uncertainty",
    "damage_confidence", "stress_confidence", "damage_weight", "stress_weight",
    "damage_uncertainty_method", "stress_uncertainty_method",
}
```

- [ ] **Step 2: 验证测试失败**

Run: `python -m pytest tests/test_uncertainty_pipeline.py::test_cli_writes_contract -v`

Expected: FAIL because `run_uncertainty.py` does not exist.

- [ ] **Step 3: 实现配置和 CLI**

```yaml
input_file: all_model_predictions.csv
output_dir: outputs/uncertainty
position_columns: [source_file, experiment_id, sample_index]
depth_preference: [cumulative_depth_cm, depth_cm]
random_state: 42
minimum_weight: 0.1
selection_metric: aurc
mc_dropout:
  enabled: false
  status: unavailable_no_model_artifacts
```

CLI 对 damage/stress 分别执行：构建位置表、交叉验证、选型、用全体可校准数据拟合最终估计器、转换权重，再按主键和 model 映射回全部原始预测行。所有 CSV 使用 `utf-8-sig`。

- [ ] **Step 4: 运行完整测试并提交**

Run: `python -m pytest -v`

Expected: all passed.

Commit: `git commit -am "feat: add uncertainty analysis CLI and outputs"`

---

### Task 6: 下游拟合接入与文档

**Files:**
- Modify: `data_loader.py`
- Modify: `run_fitting.py`
- Modify: `config.yaml`
- Modify: `README.md`
- Test: `tests/test_uncertainty_pipeline.py`

**Interfaces:**
- Consumes: `point_uncertainty.csv` 的 `damage_weight` 和 `stress_weight`。
- Produces: 拟合阶段分别对 damage/stress 使用对应权重；未提供文件时保持原有 `state_confidence` 行为。

- [ ] **Step 1: 写失败测试**

验证配置含 `uncertainty_file` 时，主键完整匹配并加载两类权重；重复主键、权重越界或缺失必需列时抛出明确错误；配置不含该文件时，旧流程输出不变。

- [ ] **Step 2: 验证测试失败**

Run: `python -m pytest -v`

Expected: FAIL because the loader does not consume uncertainty weights.

- [ ] **Step 3: 实现可选接入**

在 `config.yaml` 新增：

```yaml
uncertainty:
  file: null
  join_columns: [model, source_file, experiment_id, sample_index]
  minimum_weight: 0.1
```

`data_loader.py` 只在 `uncertainty.file` 非空时左连接，并校验 `many_to_one`；`run_fitting.py` 对 damage 使用 `damage_weight`、stress 使用 `stress_weight`，否则继续使用 `state_confidence`。不要改变离群检测 mask，只改变 `confidence_weight` 修正和拟合融合权重。权重接入时不再按 `confidence_threshold` 二值分档，而使用连续映射
`blend = low_confidence_blend - weight * (low_confidence_blend - high_confidence_blend)`：权重 1 对应最弱平滑，权重 0.1 接近最强平滑；旧的 `state_confidence` 路径保持原二值逻辑，确保向后兼容。

- [ ] **Step 4: 更新 README 并运行回归验证**

Run: `python -m pytest -v`

Expected: all passed.

Run: `python run_fitting.py --one-group`

Expected: exit 0 and sample outputs are generated.

- [ ] **Step 5: 提交**

Commit: `git commit -am "feat: consume calibrated uncertainty weights in fitting"`

---

## Final Verification

- [ ] Run `python -m pytest -v`; expected all passed.
- [ ] Run `python run_uncertainty.py`; expected four uncertainty artifacts under `outputs/uncertainty/`.
- [ ] Confirm every `source_file/target/method` fold appears exactly once in `method_evaluation.csv`.
- [ ] Confirm selected damage/stress methods equal the lowest mean AURC entries in `selected_methods.json`.
- [ ] Confirm all uncertainty values are within `[0, 1]`, all weights within `[0.1, 1]`, and no single-model position has zero disagreement imputed.
- [ ] Run `python run_fitting.py --one-group` once without and once with `uncertainty.file`; both must exit 0.
- [ ] Record MC Dropout as unavailable and do not include it in AURC rankings.
