# 组合构建、约束与 cvxpy

组合模块提供排名、比例、多空、线性、均值方差和目标波动率构建器。Arrow 用于持仓和风险表；预期收益、协方差与边界在数值层使用 NumPy。

## 排名与比例组合

```python
import numpy as np
from alphamind.portfolio.percentbuilder import percent_build
from alphamind.portfolio.rankbuilder import rank_build

selected = rank_build(er, use_rank=20).flatten()
weights = selected / selected.sum()

selected_by_percent = percent_build(er, percent=0.1).flatten()
weights_by_percent = selected_by_percent / selected_by_percent.sum()
```

这两个函数返回 0/1 指示矩阵，不会自动归一化。`groups` 可让选择在每个分组内独立进行，`masks` 可屏蔽部分资产。比例组合使用 `int(percent * n)` 计算数量，样本太小时可能为零，归一化前应检查。

## 多空组合

```python
from alphamind.portfolio.longshortbulder import long_short_builder

weights = long_short_builder(er, leverage=1.0).flatten()
```

输出按绝对权重之和缩放到 `leverage`。默认不做多空中性处理，正负信号决定方向；需要风险中性时应先中性化信号或添加优化约束。

## 线性规划与换手约束

```python
from alphamind.portfolio.linearbuilder import linear_builder

status, objective, weights = linear_builder(
    er,
    lbound=np.zeros(n_assets),
    ubound=np.full(n_assets, 0.05),
    risk_constraints=np.column_stack([np.ones(n_assets), size]),
    risk_target=(np.array([1.0, -0.1]), np.array([1.0, 0.1])),
)
```

目标为最大化 `er @ weights`；内部以 `-er` 传给最小化求解器。`risk_constraints` 形状为 `(n_assets, n_constraints)`，上下目标分别给出每个线性暴露的范围。

限制相对当前持仓的 L1 换手：

```python
status, objective, weights = linear_builder(
    er,
    lower,
    upper,
    risk_constraints,
    risk_target,
    turn_over_target=0.2,
    current_position=current_weights,
)
```

约束为 `sum(abs(weights - current_weights)) <= 0.2`，不是单边成交额的一半口径。`method` 参数仅为旧 API 兼容保留。

## 均值方差

```python
from alphamind.portfolio.meanvariancebuilder import mean_variance_builder

status, objective, weights = mean_variance_builder(
    er,
    risk_model=risk_profile,
    bm=benchmark,
    lbound=np.zeros(n_assets),
    ubound=np.full(n_assets, 0.2),
    risk_exposure=np.column_stack([np.ones(n_assets), size]),
    risk_target=(np.array([1.0, size_target]), np.array([1.0, size_target])),
    lam=2.0,
)
```

求解变量是主动权重 `weights - benchmark`，返回时加回基准。`lbound` 和 `ubound` 是总持仓边界，`risk_target` 是总组合暴露目标。`lam` 越大，风险惩罚相对收益目标越强。

`risk_profile` 可以来自 `FullRiskModel` 或 `FactorRiskModel`。完整示例见 demo 的 `optimize_latest`。

## 目标波动率

```python
from alphamind.portfolio.meanvariancebuilder import target_vol_builder

status, objective, weights = target_vol_builder(
    er,
    risk_profile,
    benchmark,
    lower,
    upper,
    risk_exposure,
    risk_target,
    vol_target=0.1,
)
```

目标波动率约束作用于主动权重 `weights - benchmark`，即约束主动风险，而不是直接约束总组合绝对波动率。`vol_target` 必须非负。

## Box 与线性约束

```python
import pyarrow as pa
from alphamind.portfolio.constraints import (
    BoundaryType,
    LinearConstraints,
    create_box_bounds,
)

bounds = create_box_bounds(
    ["total", "SIZE"],
    BoundaryType.ABSOLUTE,
    l_val=[0.0, -0.05],
    u_val=[0.0, 0.05],
)
exposure = pa.table({"total": np.ones(n_assets), "SIZE": size})
constraints = LinearConstraints(bounds, exposure, backbone=benchmark)
lower_target, upper_target = constraints.risk_targets()
risk_matrix = constraints.risk_exp
```

`LinearConstraints` 以 `backbone @ exposure` 为每个约束的中心。边界类型：

- `ABSOLUTE`：中心加固定偏移。
- `RELATIVE`：中心乘相对比例，要求中心非负。
- `MAXABSREL` / `MINABSREL`：组合绝对和相对阈值，参数为二元组。

旧 `Constraints` 类仍保留，但已弃用；新代码使用 `LinearConstraints`。

## 直接使用 cvxpy 优化器

底层类为 `LPOptimizer`、`L1LPOptimizer`、`QuadraticOptimizer` 和 `TargetVolOptimizer`。它们接收最小化目标，并提供 `status()`、`feval()` 和 `x_value()`。默认线性问题使用 HiGHS，二次/二阶锥问题使用 Clarabel；可通过 `solver` 和 `solver_options` 覆盖。

直接传 `cons_matrix` 时，每行格式为 `[暴露系数..., 下界, 上界]`，共 `n_assets + 2` 列。多数研究代码应优先使用高层构建器。

## 持仓随收益漂移

```python
from alphamind.portfolio.evolver import evolve_positions

next_weights = evolve_positions(weights, dx_return)
```

`dx_return` 是对数收益。函数先按价格变化调整权重，再保持调整前的总绝对杠杆。

## 可行性与状态

- 可接受状态为 `optimal` 和 `optimal_inaccurate`，后者仍需检查约束残差。
- 总权重为 1 时，资产上界之和必须至少为 1。
- 固定暴露目标过多、基准不在边界内或换手限制过紧都可能导致不可行。
- 不要只检查权重之和；同时检查上下界、暴露、换手和有限值。
