# 机器学习与模型持久化

模型层接收 Arrow 特征表，内部转换为 NumPy 后调用 scikit-learn 或 XGBoost。特征名、公式、训练时间和底层模型都可序列化。

## 支持的模型

| 类型 | API |
| --- | --- |
| 固定线性权重 | `ConstLinearModel` |
| 线性回归 | `LinearRegression` |
| Lasso | `LassoRegression` |
| 逻辑回归 | `LogisticRegression` |
| 支持向量回归 | `NvSVRModel` |
| 随机森林 | `RandomForestRegressor`、`RandomForestClassifier` |
| XGBoost | `XGBRegressor`、`XGBClassifier`、`XGBTrainer` |

模型构造时，`features` 可以是列名列表或表达式字典。表达式由 `Transformer` 管理，模型实际特征名为排序后的输出名。

## 固定权重模型

```python
from alphamind.model import ConstLinearModel

model = ConstLinearModel(
    features=["momentum", "value"],
    weights={"momentum": 0.6, "value": 0.4},
)
prediction = model.predict(feature_table)
```

权重字典按键映射到特征，不依赖字典声明顺序。固定模型不需要 `fit`。

## 训练线性回归

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

`fit` 只选择 `model.features` 对应列，因此输入 Arrow 表可以额外携带 `trade_date`、`code` 和标签。`predict` 返回 NumPy 数组，对外表格结果应重新构造 Arrow。

## 使用 Composer

`train_model` 和 `predict_by_model` 可直接处理 Arrow 数据：

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

`predictions` 是包含 `code`、`prediction` 的 Arrow 表。

`Composer` 进一步按日期保存多个模型：

```python
composer = Composer(alpha_model=template, data_meta=data_meta)
composer.train("2026-01-16", x=x_train, y=y_train)
predictions, features = composer.predict("2026-01-16", x=x_test)
score, x_used, y_used = composer.score("2026-01-16", x=x_test, y=y_test)
ic, x_used, y_used = composer.ic("2026-01-16", x=x_test, y=y_test)
```

不显式传 `x` 时，`DataMeta` 通过 SQL 引擎读取训练和预测数据。其参数包括调仓频率、股票池、训练批数、风险中性列、风险模型、预处理、预热窗口和数据源 URI。

## 数据准备接口

- `fetch_data_package`：一次构造多个训练/预测时间桶。
- `fetch_train_phase`：构造指定参考日之前的训练窗口。
- `fetch_predict_phase`：构造指定参考日预测截面，可按交易日中位数填充因子。
- `prepare_data`：读取并连接因子、标签、行业和基准。
- `batch_processing`：将数值窗口切分为训练与预测桶。

这些接口依赖数据库数据模型。纯 Parquet 工作流可直接用 Arrow 过滤日期，再把表传给 `fit` 或 `train_model`，如 demo 所示。

## 保存与加载

```python
from alphamind.model import load_model

description = model.save()
restored = load_model(description)
restored_predictions = restored.predict(x_test)
```

描述包含：

- 模型类路径、语言和保存时间；
- 特征名、因子公式与拟合目标；
- 底层模型序列化内容和训练时间；
- scikit-learn 或 XGBoost 版本；
- 线性模型权重（适用时）。

当前环境库版本低于模型保存版本时会记录警告。序列化使用 Python 对象编码，只加载受信任来源的模型描述。

## 训练规范

- 训练标签与特征必须按键和顺序对齐。
- 避免未来数据进入预处理、风险中性化或训练窗口。
- 分类模型标签应是类别，回归模型标签应为一维数值数组。
- 保存预测时附加 `trade_date` 和 `code`，不要只保存无键数组。
- 用 IC、分层收益和组合结果共同评价，而不是只看拟合分数。
