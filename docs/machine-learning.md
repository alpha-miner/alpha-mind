# Machine Learning and Model Persistence

The model layer accepts Arrow feature tables, converts them internally to NumPy, and calls scikit-learn or XGBoost. Feature names, formulas, training time, and the underlying model can all be serialized.

## Supported models

| Type | API |
| --- | --- |
| Fixed linear weights | `ConstLinearModel` |
| Linear regression | `LinearRegression` |
| Lasso | `LassoRegression` |
| Logistic regression | `LogisticRegression` |
| Support-vector regression | `NvSVRModel` |
| Random forest | `RandomForestRegressor`, `RandomForestClassifier` |
| XGBoost | `XGBRegressor`, `XGBClassifier`, `XGBTrainer` |

At model construction, `features` may be a list of column names or a dictionary of expressions. Expressions are managed by `Transformer`, and actual model feature names are the sorted output names.

## Fixed-weight model

```python
from alphamind.model import ConstLinearModel

model = ConstLinearModel(
    features=["momentum", "value"],
    weights={"momentum": 0.6, "value": 0.4},
)
prediction = model.predict(feature_table)
```

Weight dictionaries map keys to features and do not depend on dictionary declaration order. Fixed models do not require `fit`.

## Training a linear regression

```python
import numpy as np
import pyarrow as pa
from alphamind.model import LinearRegression

x_train = pa.table({"momentum": [0.1, 0.3, 0.2], "value": [0.2, 0.1, 0.4]})
y_train = np.array([0.01, 0.02, 0.015])
model = LinearRegression(features=["momentum", "value"], fit_intercept=True)
model.fit(x_train, y_train)

predictions = model.predict(x_train)
score = model.score(x_train, y_train)
ic = model.ic(x_train, y_train)
```

`fit` selects only the columns corresponding to `model.features`, so the input Arrow table may also carry `trade_date`, `code`, and labels. `predict` returns a NumPy array; construct an Arrow table for external tabular results.

## Using Composer

`train_model` and `predict_by_model` process Arrow data directly:

```python
from alphamind.model.composer import Composer, DataMeta, predict_by_model, train_model

model, x_used, y_used = train_model(
    "2026-01-16",
    LinearRegression(features=["momentum", "value"]),
    x_values=x_train,
    y_values=y_train,
)
predictions, features = predict_by_model(
    "2026-01-16",
    model,
    x_values=x_test,
    codes=codes,
)
```

`predictions` is an Arrow table containing `code` and `prediction`.

`Composer` additionally stores multiple models by date:

```python
composer = Composer(alpha_model=template, data_meta=data_meta)
composer.train("2026-01-16", x=x_train, y=y_train)
predictions, features = composer.predict("2026-01-16", x=x_test)
score, x_used, y_used = composer.score("2026-01-16", x=x_test, y=y_test)
ic, x_used, y_used = composer.ic("2026-01-16", x=x_test, y=y_test)
```

When `x` is not passed explicitly, `DataMeta` reads training and prediction data through the SQL engine. Its parameters include rebalancing frequency, universe, number of training batches, risk-neutralization columns, risk model, preprocessing, warm-up window, and data-source URI.

## Data-preparation interfaces

- `fetch_data_package`: construct multiple training/prediction time buckets in one operation.
- `fetch_train_phase`: construct the training window before a specified reference date.
- `fetch_predict_phase`: construct the prediction cross-section for a reference date, optionally filling factors with trading-day medians.
- `prepare_data`: read and join factors, labels, industries, and benchmarks.
- `batch_processing`: split numerical windows into training and prediction buckets.

These interfaces depend on the database data model. Pure Parquet workflows can filter dates directly with Arrow and pass tables to `fit` or `train_model`, as demonstrated by the demo.

## Saving and loading

```python
from alphamind.model import load_model

description = model.save()
restored = load_model(description)
restored_predictions = restored.predict(x_test)
```

The description contains:

- Model class path, language, and save time.
- Feature names, factor formulas, and fit targets.
- Serialized underlying-model content and training time.
- scikit-learn or XGBoost versions.
- Linear-model weights, where applicable.

A warning is logged when the current environment's library version is older than the version used to save the model. Serialization uses Python object encoding; load model descriptions only from trusted sources.

## Training guidelines

- Align training labels and features by key and order.
- Keep future data out of preprocessing, risk neutralization, and training windows.
- Classification labels should be categories; regression labels should be one-dimensional numerical arrays.
- Attach `trade_date` and `code` when saving predictions rather than storing unlabeled arrays alone.
- Evaluate models with IC, quantile returns, and portfolio results together rather than relying only on fit scores.
