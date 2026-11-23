import numpy as np
import pyarrow as pa

from alphamind.analysis.crosssetctions import cross_section_analysis, cs_impl


def input_tables():
    factors = pa.table({"code": ["A", "B", "C"], "factor": [1.0, 2.0, 3.0]})
    risk = pa.table({"code": ["A", "B", "C"], "BETA": [0.2, 0.4, 0.6]})
    industry = pa.table({"code": ["A", "B", "C"], "industry_name": ["Bank", "Bank", "Auto"]})
    returns = pa.table({"code": ["A", "B", "C"], "dx": np.log1p([0.01, 0.02, -0.01])})
    return factors, risk, industry, returns


def test_cross_section_implementation_returns_finite_statistics():
    factors, risk, industry, returns = input_tables()
    portfolio_return, ic, t_stat = cs_impl(
        "2026-01-05", factors, "factor", risk, ["BETA"], industry, returns
    )
    assert np.isfinite(portfolio_return)
    assert np.isfinite(ic)
    assert np.isfinite(t_stat)


def test_cross_section_implementation_rejects_excessive_missing_data():
    factors, risk, industry, returns = input_tables()
    factors = pa.table({"code": ["A", "B", "C", "D"], "factor": [1.0, 2.0, 3.0, 4.0]})
    risk = risk.slice(0, 1)
    industry = industry.slice(0, 1)
    result = cs_impl("2026-01-05", factors, "factor", risk, ["BETA"], industry, returns)
    assert np.isnan(result).all()


def test_cross_section_analysis_orchestrates_engine_queries():
    factors, risk, industry, returns = input_tables()

    class Engine:
        def fetch_codes(self, ref_date, universe):
            assert ref_date == "2026-01-05"
            assert universe == "hs300"
            return ["A", "B", "C"]

        def fetch_risk_model(self, ref_date, codes):
            return None, risk

        def fetch_factor(self, ref_date, factor_name, codes):
            assert factor_name == "factor"
            return factors

        def fetch_industry_matrix(self, ref_date, codes, category, level):
            assert (category, level) == ("sw_adj", 1)
            return industry

        def fetch_dx_return(self, ref_date, codes, horizon, offset):
            assert (horizon, offset) == (1, 1)
            return returns

    result = cross_section_analysis("2026-01-05", "factor", "hs300", 1, ["BETA"], Engine())
    assert np.isfinite(result).all()
