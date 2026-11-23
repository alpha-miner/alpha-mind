# 数据预处理

预处理函数作用于 NumPy 数值矩阵；Arrow 表通过 `to_numpy` 进入该层，处理后再构造成 Arrow 表。截面研究通常应在每个 `trade_date` 内独立处理，避免不同日期的数据混合。

## 标准流程

```python
from alphamind.data.arrow import to_numpy, with_column
from alphamind.data.processing import factor_processing
from alphamind.data.standardize import standardize
from alphamind.data.winsorize import winsorize_normal

values = to_numpy(section, ["momentum", "value"], dtype=float)
risk = to_numpy(section, ["SIZE"], dtype=float)
processed = factor_processing(
    values,
    pre_process=[winsorize_normal, standardize],
    risk_factors=risk,
    post_process=[standardize],
)
for index, name in enumerate(["momentum", "value"]):
    section = with_column(section, name, processed[:, index])
```

处理顺序固定为：

1. 依次执行 `pre_process`。
2. 若提供 `risk_factors`，剔除全零风险列并回归中性化。
3. 依次执行 `post_process`。

`groups` 会传给每个处理器和中性化函数，用于更细的分组处理。

## 去极值

```python
from alphamind.data.winsorize import NormalWinsorizer, winsorize_normal

clipped = winsorize_normal(values, num_stds=3, ddof=1, method="flat")
processor = NormalWinsorizer(num_stds=3, method="flat")
processor.fit(train_values)
test_values = processor.transform(test_values)
```

- `flat` 将超出正常均值区间的值压到边界。
- 其他 `method` 选择插值路径，`interval` 控制插值区间。
- 无状态函数在当前样本上计算统计量；有状态类可把训练样本统计量应用到新样本。

## 标准化

```python
from alphamind.data.standardize import Standardizer, projection, standardize

zscore = standardize(values, ddof=1)
standardizer = Standardizer(ddof=1)
standardizer.fit(train_values)
new_zscore = standardizer.transform(test_values)
unit_projection = projection(values, axis=1)
```

标准差下限为 `1e-8`，用于避免常数列除零。分组处理时，有状态转换要求新数据标签已在 `fit` 中出现。

## 中性化

```python
from alphamind.data.neutralize import neutralize

residual = neutralize(risk, values)
residual_by_industry = neutralize(risk, values, groups=industry)
residual, detail = neutralize(risk, values, detail=True)
```

`risk` 形状为 `(n_samples, n_risks)`，`values` 为一维或二维。`detail=True` 时，字典包含：

- `exposure`：每个风险因子对应的回归暴露。
- `explained`：风险因子解释的收益或因子部分。

求解失败时回退到伪逆。中性化不等于删除风险数据；组合构建仍可能需要原始暴露矩阵。

## 排名与分位数

```python
from alphamind.data.quantile import quantile
from alphamind.data.rank import percentile, rank

zero_based_rank = rank(values)
percentiles = percentile(values)
bins = quantile(values[:, 0], n_bins=5)
```

`rank` 从 0 开始；`percentile` 映射到 `[0, 1]`；`quantile` 返回从 0 开始的分组编号。`CSRank` 和 `CSQuantiles` 则是 calc-flow 表达式，适用于 Arrow 面板。

## 避免数据泄漏

- 每个截面独立去极值和标准化。
- 只用当时已知的风险暴露和因子值。
- 训练窗口与预测窗口分开，避免使用未来收益。
- 窗口未形成的 null 不应作为零参与训练。
- 训练、预测使用相同的特征顺序和处理流程。

完整逐截面处理见 `examples/parquet_demo/demo.py` 的 `process_features`。
