import numpy as np
import pyarrow as pa
import statsmodels.api as sm

from alphamind.data.arrow import as_arrow, column_values, drop_nulls, replace_inf, to_numpy
from alphamind.data.processing import factor_processing
from alphamind.data.standardize import standardize
from alphamind.data.winsorize import winsorize_normal
from alphamind.utilities import alpha_logger


def cs_impl(
    ref_date, factor_data, factor_name, risk_exposure, constraint_risk, industry_matrix, dx_returns
):
    factors = as_arrow(factor_data)
    total_data = factors.join(as_arrow(risk_exposure), keys="code", join_type="inner")
    total_data = total_data.join(as_arrow(industry_matrix), keys="code", join_type="inner")
    total_data = drop_nulls(replace_inf(total_data))

    if total_data.num_rows < 0.33 * factors.num_rows:
        alpha_logger.warning(
            f"valid data point({total_data.num_rows}) is less than 33% of the total sample"
        )
        return np.nan, np.nan, np.nan

    total_risk_exp = to_numpy(total_data, constraint_risk, dtype=float)
    er = to_numpy(total_data, [factor_name], dtype=float)
    er = factor_processing(
        er, [winsorize_normal, standardize], total_risk_exp, [standardize]
    ).flatten()
    industry = column_values(total_data, "industry_name")
    codes = total_data["code"].to_pylist()
    weights = er / np.abs(er).sum()
    target_pos = pa.table({"code": codes, "weight": weights, "industry": industry})
    target_pos = target_pos.join(as_arrow(dx_returns), keys="code", join_type="inner")
    target_pos = target_pos.join(
        total_data.select(["code"] + constraint_risk), keys="code", join_type="inner"
    )
    total_risk_exp = to_numpy(target_pos, constraint_risk, dtype=float)
    active_weight = column_values(target_pos, "weight")
    excess_return = np.exp(to_numpy(target_pos, ["dx"], dtype=float)) - 1.0
    excess_return = factor_processing(
        excess_return,
        [winsorize_normal, standardize],
        total_risk_exp,
        [winsorize_normal, standardize],
    ).flatten()
    port_ret = np.log(active_weight @ excess_return + 1.0)
    ic = np.corrcoef(excess_return, active_weight)[0, 1]
    results = sm.OLS(excess_return, sm.add_constant(active_weight)).fit()
    t_stats = results.tvalues[1]

    alpha_logger.info(f"{ref_date} is finished with {target_pos.num_rows} stocks for {factor_name}")
    alpha_logger.info(
        f"{ref_date} risk_exposure: {np.sum(np.square(active_weight @ total_risk_exp))}"
    )
    return port_ret, ic, t_stats


def cross_section_analysis(ref_date, factor_name, universe, horizon, constraint_risk, engine):
    codes = engine.fetch_codes(ref_date, universe)
    risk_exposure = engine.fetch_risk_model(ref_date, codes)[1].select(["code"] + constraint_risk)
    factor_data = engine.fetch_factor(ref_date, factor_name, codes)
    industry_matrix = engine.fetch_industry_matrix(ref_date, codes, "sw_adj", 1)
    dx_returns = engine.fetch_dx_return(ref_date, codes, horizon=horizon, offset=1)
    return cs_impl(
        ref_date,
        factor_data,
        factor_name,
        risk_exposure,
        constraint_risk,
        industry_matrix,
        dx_returns,
    )
