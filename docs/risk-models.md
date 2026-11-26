# Risk Models

Risk models are located in `alphamind.portfolio.riskmodel`. Constructors accept Arrow tables, while `get_risk_profile` returns a dictionary of NumPy matrices for cvxpy builders. Code selections accept Python sequences or Arrow arrays; code matching and idiosyncratic-risk alignment use Arrow Compute, and unknown codes raise `KeyError`.

## Full-covariance model

This model is appropriate when the number of assets is small and the security covariance matrix has already been estimated:

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

Other than `code`, every column name must be the corresponding security-code string. In `profile`, `cov` is the security covariance and all other factor fields are `None`.

## Factor risk model

The factor model uses:

```text
security covariance = B @ F @ B.T + diag(idsync)
```

Here, `B` is asset risk exposure, `F` is factor covariance, and `idsync` is each asset's idiosyncratic variance.

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

Input requirements:

- The covariance table identifies rows with `Factor`; risk-factor column names must match the exposure-table columns.
- Metadata such as `FactorID` and `trade_date` is not treated as numerical risk factors.
- The exposure table contains one asset per row, and `code` determines the default matrix order.
- The idiosyncratic-risk column may be named `idsync` or `srisk`; when constructing a local model directly, its values are interpreted as variances.
- Database engines convert database covariance and percentage-based idiosyncratic risk to decimal variances before constructing `FactorRiskModel`.

## Retrieving subsets

```python
exp = model.get_risk_exp(["A"])
cov = model.get_factor_cov()
idsync = model.get_idsync(["A"])
```

Code-subset order is determined by the call argument. During optimization, expected returns, benchmark weights, bounds, risk exposures, and `codes` must all use the same order to prevent numerical misalignment.

## Using a model in portfolio optimization

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

`profile` contains either `cov` or the `factor_cov` + `factor_loading` + `idsync` path. `constraint_exposure` is an additional linear-constraint matrix and is not the same parameter as factor exposure in the risk model; common constraints include total weight, SIZE, or industry exposure.

## Numerical checks

- Covariance matrices should be finite and symmetric and should remain positive semidefinite where possible.
- `idsync` is variance, must be nonnegative, and must not be supplied as standard deviation.
- The risk time horizon should match the return holding period.
- Solvers symmetrize covariance internally but do not correct incorrect units.
- Missing data and code misalignment deserve more attention than solver tolerances.

See the [Parquet demo](setup.md) for complete local construction and optimization.
