# Portfolio Construction, Constraints, and cvxpy

The portfolio module provides ranked, percentage, long-short, linear, mean-variance, and target-volatility builders. Arrow is used for holdings and risk tables; expected returns, covariance, and bounds use NumPy at the numerical layer.

## Ranked and percentage portfolios

```python
import numpy as np
from alphamind.portfolio.percentbuilder import percent_build
from alphamind.portfolio.rankbuilder import rank_build

selected = rank_build(er, use_rank=20).flatten()
weights = selected / selected.sum()

selected_by_percent = percent_build(er, percent=0.1).flatten()
weights_by_percent = selected_by_percent / selected_by_percent.sum()
```

These functions return 0/1 indicator matrices and do not normalize automatically. `groups` performs selection independently within each group, and `masks` can exclude assets. Percentage portfolios calculate the count as `int(percent * n)`, which may be zero for small samples, so check before normalizing.

## Long-short portfolios

```python
from alphamind.portfolio.longshortbulder import long_short_builder

weights = long_short_builder(er, leverage=1.0).flatten()
```

Output is scaled so the sum of absolute weights equals `leverage`. Long-short neutrality is not applied by default; positive and negative signals determine direction. Neutralize the signal first or add optimization constraints when risk neutrality is required.

## Linear programming and turnover constraints

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

The objective maximizes `er @ weights`; internally, `-er` is passed to the minimizing solver. `risk_constraints` has shape `(n_assets, n_constraints)`, and the lower and upper targets define the range of every linear exposure.

Limit L1 turnover relative to current holdings with:

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

The constraint is `sum(abs(weights - current_weights)) <= 0.2`, not a one-sided convention that divides traded value by two. The `method` parameter is retained only for old-API compatibility.

## Mean-variance optimization

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

The solver variable is active weight `weights - benchmark`, and the benchmark is added back to the returned value. `lbound` and `ubound` are total-holding bounds, while `risk_target` defines total-portfolio exposure targets. Larger `lam` values strengthen risk penalties relative to the return objective.

`risk_profile` can come from `FullRiskModel` or `FactorRiskModel`. See `optimize_latest` in the demo for a complete example.

## Target volatility

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

The target-volatility constraint applies to active weight `weights - benchmark`, thereby constraining active risk rather than the absolute volatility of the total portfolio directly. `vol_target` must be nonnegative.

## Box and linear constraints

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

`LinearConstraints` centers each constraint at `backbone @ exposure`. Boundary types are:

- `ABSOLUTE`: center plus a fixed offset.
- `RELATIVE`: center multiplied by a relative proportion; the center must be nonnegative.
- `MAXABSREL` / `MINABSREL`: combines absolute and relative thresholds; the parameter is a pair.

The old `Constraints` class remains available but is deprecated; new code should use `LinearConstraints`.

## Using cvxpy optimizers directly

The low-level classes are `LPOptimizer`, `L1LPOptimizer`, `QuadraticOptimizer`, and `TargetVolOptimizer`. They accept minimization objectives and provide `status()`, `feval()`, and `x_value()`. Linear problems use HiGHS by default, while quadratic and second-order-cone problems use Clarabel; override these choices with `solver` and `solver_options`.

When passing `cons_matrix` directly, each row has the format `[exposure coefficients..., lower bound, upper bound]`, for `n_assets + 2` columns. Most research code should prefer the high-level builders.

## Holding drift with returns

```python
from alphamind.portfolio.evolver import evolve_positions

next_weights = evolve_positions(weights, dx_return)
```

`dx_return` is a logarithmic return. The function first adjusts weights for price changes, then preserves the total absolute leverage from before the adjustment.

## Feasibility and status

- Accepted statuses are `optimal` and `optimal_inaccurate`; the latter still requires checking constraint residuals.
- When total weight is 1, the sum of asset upper bounds must be at least 1.
- Too many fixed exposure targets, a benchmark outside its bounds, or an excessively tight turnover limit can make the problem infeasible.
- Do not check only the weight sum; also check bounds, exposures, turnover, and finite values.
