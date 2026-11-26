# Data Preprocessing

Preprocessing functions operate on NumPy numerical matrices. Arrow tables enter this layer through `to_numpy` and are converted back to Arrow tables afterward. Cross-sectional research should normally process each `trade_date` independently to avoid mixing data from different dates.

## Standard workflow

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

The processing order is fixed:

1. Run each `pre_process` step in order.
2. When `risk_factors` is provided, remove all-zero risk columns and neutralize by regression.
3. Run each `post_process` step in order.

`groups` is passed to every processor and to the neutralization function for finer-grained processing.

## Winsorization

```python
from alphamind.data.winsorize import NormalWinsorizer, winsorize_normal

clipped = winsorize_normal(values, num_stds=3, ddof=1, method="flat")
processor = NormalWinsorizer(num_stds=3, method="flat")
processor.fit(train_values)
test_values = processor.transform(test_values)
```

- `flat` clips values outside the normal mean interval to its boundaries.
- Other `method` values select interpolation paths, and `interval` controls the interpolation interval.
- Stateless functions calculate statistics on the current sample; stateful classes can apply training-sample statistics to new samples.

## Standardization

```python
from alphamind.data.standardize import Standardizer, projection, standardize

zscore = standardize(values, ddof=1)
standardizer = Standardizer(ddof=1)
standardizer.fit(train_values)
new_zscore = standardizer.transform(test_values)
unit_projection = projection(values, axis=1)
```

The standard-deviation floor is `1e-8` to prevent division by zero for constant columns. For grouped processing, stateful transforms require every new data label to have appeared during `fit`.

## Neutralization

```python
from alphamind.data.neutralize import neutralize

residual = neutralize(risk, values)
residual_by_industry = neutralize(risk, values, groups=industry)
residual, detail = neutralize(risk, values, detail=True)
```

`risk` has shape `(n_samples, n_risks)`, while `values` may be one- or two-dimensional. With `detail=True`, the dictionary contains:

- `exposure`: regression exposure corresponding to each risk factor.
- `explained`: the return or factor component explained by the risk factors.

A pseudoinverse fallback is used when the primary solve fails. Neutralization does not mean deleting risk data; portfolio construction may still require the original exposure matrix.

## Ranks and quantiles

```python
from alphamind.data.quantile import quantile
from alphamind.data.rank import percentile, rank

zero_based_rank = rank(values)
percentiles = percentile(values)
bins = quantile(values[:, 0], n_bins=5)
```

`rank` is zero-based; `percentile` maps to `[0, 1]`; and `quantile` returns zero-based bucket numbers. `CSRank` and `CSQuantiles` are instead calc-flow expressions for Arrow panels.

## Avoiding data leakage

- Winsorize and standardize each cross-section independently.
- Use only risk exposures and factor values known at the time.
- Separate training and prediction windows, and never use future returns.
- Do not treat nulls from incomplete windows as zero during training.
- Use identical feature ordering and processing pipelines for training and prediction.

See `process_features` in `examples/parquet_demo/demo.py` for complete cross-section-by-cross-section processing.
