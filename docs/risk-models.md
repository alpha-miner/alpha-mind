# 风险模型

风险模型位于 `alphamind.portfolio.riskmodel`。构造输入是 Arrow 表，`get_risk_profile` 输出供 cvxpy 构建器使用的 NumPy 矩阵字典。代码选择接受 Python 序列或 Arrow 数组；代码匹配和特异风险对齐使用 Arrow Compute，未知代码会抛出 `KeyError`。

## 全协方差模型

适用于资产数量较少、已经估计好证券协方差矩阵的场景：

```python
import pyarrow as pa
from alphamind.portfolio.riskmodel import FullRiskModel

sec_cov = pa.table(
    {
        "code": ["A", "B"],
        "A": [0.04, 0.01],
        "B": [0.01, 0.02],
    }
)
model = FullRiskModel(sec_cov)
covariance = model.get_cov(["A", "B"])
profile = model.get_risk_profile(["A", "B"])
```

除 `code` 外，每列名称必须是对应证券代码的字符串。`profile` 中 `cov` 为证券协方差，其余因子字段为 `None`。

## 因子风险模型

因子模型使用：

```text
证券协方差 = B @ F @ B.T + diag(idsync)
```

其中 `B` 是资产风险暴露，`F` 是因子协方差，`idsync` 是每只资产的特质方差。

```python
factor_cov = pa.table(
    {
        "Factor": ["BETA", "SIZE"],
        "BETA": [0.0004, 0.00002],
        "SIZE": [0.00002, 0.0009],
    }
)
risk_exp = pa.table(
    {
        "code": ["A", "B"],
        "BETA": [1.1, 0.9],
        "SIZE": [-0.3, 0.4],
    }
)
specific = pa.table({"code": ["A", "B"], "idsync": [0.0005, 0.0007]})

from alphamind.portfolio.riskmodel import FactorRiskModel

model = FactorRiskModel(factor_cov, risk_exp, specific)
profile = model.get_risk_profile(["A", "B"])
```

输入要求：

- 协方差表用 `Factor` 标识行，风险因子列与暴露表列名一致。
- `FactorID`、`trade_date` 等元数据不会作为数值风险因子。
- 暴露表每行一个资产，`code` 决定默认矩阵顺序。
- 特质风险列可命名为 `idsync` 或 `srisk`；直接构造本地模型时，其数值按方差解释。
- 数据库引擎会把数据库中的协方差和百分化特质风险转换为小数方差后，再构造 `FactorRiskModel`。

## 获取子集

```python
exp = model.get_risk_exp(["A"])
cov = model.get_factor_cov()
idsync = model.get_idsync(["A"])
```

代码子集顺序由调用参数决定。优化时必须让预期收益、基准、边界、风险暴露和 `codes` 使用同一顺序，否则数值会错位。

## 用于组合优化

```python
from alphamind.portfolio.meanvariancebuilder import mean_variance_builder

status, objective, weights = mean_variance_builder(
    er,
    risk_model=profile,
    bm=benchmark,
    lbound=lower,
    ubound=upper,
    risk_exposure=constraint_exposure,
    risk_target=(lower_target, upper_target),
    lam=2.0,
)
```

`profile` 包含 `cov` 或 `factor_cov` + `factor_loading` + `idsync` 两条路径之一。`constraint_exposure` 是额外线性约束矩阵，与风险模型中的因子暴露不是同一参数；常见约束包括总权重、SIZE 或行业暴露。

## 数值检查

- 协方差矩阵应有限、对称并尽量保持半正定。
- `idsync` 是方差，应非负，不能直接传标准差。
- 风险时间尺度应与收益持有期一致。
- 求解器内部会对协方差做对称化，但不会替用户修正错误单位。
- 数据缺失和代码错位比求解器容差更值得优先检查。

完整本地构造和优化见 [Parquet demo](setup.md)。
