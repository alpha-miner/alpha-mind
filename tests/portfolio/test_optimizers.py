import cvxpy as cp
import numpy as np
import pytest

from alphamind.portfolio.optimizers import (
    L1LPOptimizer,
    LPOptimizer,
    QuadraticOptimizer,
    TargetVolOptimizer,
    _risk_expression,
)


class TestOptimizers:
    def test_lpoptimizer(self):
        er = np.array([-1.0, -2.0])
        lower_bound = np.array([0.0, 0.2])
        upper_bound = np.array([1.0, 0.8])
        optimizer = LPOptimizer(
            objective=-er,
            cons_matrix=np.array([[1.0, 1.0, 1.0, 1.0]]),
            lbound=lower_bound,
            ubound=upper_bound,
        )

        assert optimizer.feval() == pytest.approx(1.2, abs=10.0 ** (-7))
        np.testing.assert_array_almost_equal(optimizer.x_value(), [0.8, 0.2])

    def test_qpoptimizer(self):
        er = np.array([0.01, 0.02, 0.03])
        cov = np.array([[0.02, 0.01, 0.02], [0.01, 0.02, 0.03], [0.02, 0.03, 0.02]])
        ids_var = np.diag([0.01, 0.02, 0.03])
        cov += ids_var
        lbound = np.array([0.0, 0.0, 0.0])
        ubound = np.array([0.4, 0.4, 0.5])
        cons = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        clbound = np.array([1.0, 0.3])
        cubound = np.array([1.0, 0.7])
        cons_matrix = np.concatenate(
            [cons, clbound.reshape((-1, 1)), cubound.reshape((-1, 1))], axis=1
        )
        optimizer = QuadraticOptimizer(
            objective=-er, cov=cov, lbound=lbound, ubound=ubound, cons_matrix=cons_matrix
        )
        np.testing.assert_array_almost_equal(optimizer.x_value(), [0.2, 0.3, 0.5], 4)

    def test_qpoptimizer_with_factor_model(self):
        er = np.array([0.1, 0.2, 0.3])
        lbound = np.array([0.0, 0.0, 0.0])
        ubound = np.array([1.0, 1.0, 1.0])
        factor_var = np.array([[0.5, -0.3], [-0.3, 0.7]])
        factor_load = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
        idsync = np.array([0.1, 0.3, 0.2])
        cons = np.array([[1.0, 1.0, 1.0]])
        clbound = np.array([1.0])
        cubound = np.array([1.0])
        cons_matrix = np.concatenate(
            [cons, clbound.reshape((-1, 1)), cubound.reshape((-1, 1))], axis=1
        )
        optimizer = QuadraticOptimizer(
            objective=-er,
            lbound=lbound,
            ubound=ubound,
            factor_cov=factor_var,
            factor_load=factor_load,
            factor_special=idsync,
            cons_matrix=cons_matrix,
        )
        np.testing.assert_array_almost_equal(
            optimizer.x_value(), [0.2866857, 0.21416417, 0.49915014], 4
        )

    def test_qpoptimizer_with_identity_matrix(self):
        er = np.array([-0.02, 0.01, 0.03])
        cov = np.diag([1.0, 1.0, 1.0])
        optimizer = QuadraticOptimizer(objective=-er, cov=cov)
        np.testing.assert_array_almost_equal(optimizer.x_value(), [-0.02, 0.01, 0.03], 4)

    def test_target_vol_optimizer_without_cons(self):
        er = np.array([0.1, 0.2, 0.3])
        cov = np.array([[0.05, 0.01, 0.02], [0.01, 0.06, 0.03], [0.02, 0.03, 0.07]])
        lbound = np.array([-0.3, -0.3, -0.3])
        ubound = np.array([0.5, 0.5, 0.5])
        target_vol = 0.1
        optimizer = TargetVolOptimizer(
            objective=-er, cov=cov, lbound=lbound, ubound=ubound, target_vol=target_vol
        )
        np.testing.assert_array_almost_equal(
            optimizer.x_value(), [0.0231776, 0.1274768, 0.30130881], 4
        )

    def test_target_vol_optimizer_with_cons(self):
        er = np.array([0.1, 0.2, 0.3])
        cov = np.array([[0.05, 0.01, 0.02], [0.01, 0.06, 0.03], [0.02, 0.03, 0.07]])
        lbound = np.array([-0.3, -0.3, -0.3])
        ubound = np.array([0.5, 0.5, 0.5])
        cons = np.array([[1.0, 1.0, 1.0]])
        clbound = np.array([0.0])
        cubound = np.array([0.0])
        cons_matrix = np.concatenate(
            [cons, clbound.reshape((-1, 1)), cubound.reshape((-1, 1))], axis=1
        )
        target_vol = 0.1
        optimizer = TargetVolOptimizer(
            objective=-er,
            cov=cov,
            lbound=lbound,
            ubound=ubound,
            target_vol=target_vol,
            cons_matrix=cons_matrix,
        )
        np.testing.assert_array_almost_equal(
            optimizer.x_value(), [-0.3, -0.10919033, 0.40919033], 4
        )

    def test_target_vol_optimizer_with_factor_model(self):
        er = np.array([0.1, 0.2, 0.3])
        lbound = np.array([0.0, 0.0, 0.0])
        ubound = np.array([1.0, 1.0, 1.0])
        factor_var = np.array([[0.5, -0.3], [-0.3, 0.7]])
        factor_load = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
        idsync = np.array([0.1, 0.3, 0.2])
        cons = np.array([[1.0, 1.0, 1.0]])
        clbound = np.array([1.0])
        cubound = np.array([1.0])
        target_vol = 0.5
        cons_matrix = np.concatenate(
            [cons, clbound.reshape((-1, 1)), cubound.reshape((-1, 1))], axis=1
        )
        optimizer = TargetVolOptimizer(
            objective=-er,
            factor_cov=factor_var,
            factor_load=factor_load,
            factor_special=idsync,
            lbound=lbound,
            ubound=ubound,
            target_vol=target_vol,
            cons_matrix=cons_matrix,
        )
        np.testing.assert_array_almost_equal(
            optimizer.x_value(), [0.26595552, 0.21675092, 0.51729356], 4
        )

    def test_target_vol_with_cons_and_ieq(self):
        er = np.array([0.1, 0.2, 0.3])
        cov = np.array([[0.05, 0.01, 0.02], [0.01, 0.06, 0.03], [0.02, 0.03, 0.07]])
        lbound = np.array([-0.3, -0.3, -0.3])
        ubound = np.array([0.5, 0.5, 0.5])
        cons = np.array([[1.0, 1.0, 1.0]])
        clbound = np.array([0.0])
        cubound = np.array([0.0])
        target_vol = 0.1
        cons_matrix = np.concatenate(
            [cons, clbound.reshape((-1, 1)), cubound.reshape((-1, 1))], axis=1
        )
        optimizer = TargetVolOptimizer(
            objective=-er,
            cov=cov,
            lbound=lbound,
            ubound=ubound,
            target_vol=target_vol,
            cons_matrix=cons_matrix,
        )
        np.testing.assert_array_almost_equal(
            optimizer.x_value(), [-0.3, -0.10919033, 0.40919033], 4
        )

    def test_lpoptimizer_reuses_dpp_model_with_clarabel(self):
        cons_matrix = np.array([[1.0, 1.0, 1.0, 1.0]])
        optimizer = LPOptimizer(
            objective=np.array([-1.0, -2.0]),
            cons_matrix=cons_matrix,
            lbound=np.zeros(2),
            ubound=np.ones(2),
        )
        assert optimizer.solver == "CLARABEL"
        assert optimizer.is_dpp
        assert optimizer.is_dcp
        assert optimizer.solve_count == 1

        objective = np.array([-2.0, -1.0])
        optimizer.solve(objective=objective)
        reference = LPOptimizer(
            objective=objective,
            cons_matrix=cons_matrix,
            lbound=np.zeros(2),
            ubound=np.ones(2),
        )

        assert optimizer.solve_count == 2
        np.testing.assert_allclose(optimizer.x_value(), reference.x_value(), atol=1e-8)
        assert optimizer.feval() == pytest.approx(reference.feval(), abs=1e-10)

    def test_l1optimizer_reuses_dpp_model_with_new_turnover_data(self):
        cons_matrix = np.array([[1.0, 1.0, 1.0, 1.0]])
        optimizer = L1LPOptimizer(
            objective=np.array([-1.0, -0.5]),
            cons_matrix=cons_matrix,
            current_pos=np.array([1.0, 0.0]),
            target_turn_over=0.2,
            lbound=np.zeros(2),
            ubound=np.ones(2),
        )
        objective = np.array([-0.5, -1.0])
        current = np.array([0.0, 1.0])
        optimizer.solve(
            objective=objective,
            current_pos=current,
            target_turn_over=0.4,
        )
        reference = L1LPOptimizer(
            objective=objective,
            cons_matrix=cons_matrix,
            current_pos=current,
            target_turn_over=0.4,
            lbound=np.zeros(2),
            ubound=np.ones(2),
        )

        assert optimizer.is_dpp
        assert optimizer.solve_count == 2
        np.testing.assert_allclose(optimizer.x_value(), reference.x_value(), atol=1e-7)

    def test_quadratic_optimizer_reuses_model_and_rejects_new_risk(self):
        cov = np.eye(2)
        cons_matrix = np.array([[1.0, 1.0, 1.0, 1.0]])
        optimizer = QuadraticOptimizer(
            objective=np.array([-0.1, -0.2]),
            cons_matrix=cons_matrix,
            lbound=np.zeros(2),
            ubound=np.ones(2),
            cov=cov,
        )
        objective = np.array([-0.2, -0.1])
        optimizer.solve(objective=objective, penalty=2.0)
        reference = QuadraticOptimizer(
            objective=objective,
            cons_matrix=cons_matrix,
            lbound=np.zeros(2),
            ubound=np.ones(2),
            penalty=2.0,
            cov=cov,
        )

        assert optimizer.is_dpp
        np.testing.assert_allclose(optimizer.x_value(), reference.x_value(), atol=1e-7)
        with pytest.raises(ValueError, match="risk model is fixed"):
            optimizer.solve(cov=2.0 * cov)

    def test_target_vol_optimizer_reuses_dpp_model(self):
        cov = np.eye(2)
        cons_matrix = np.array([[1.0, 1.0, 1.0, 1.0]])
        optimizer = TargetVolOptimizer(
            objective=np.array([-0.1, -0.2]),
            cons_matrix=cons_matrix,
            lbound=np.zeros(2),
            ubound=np.ones(2),
            target_vol=0.8,
            cov=cov,
        )
        optimizer.solve(target_vol=1.0)
        reference = TargetVolOptimizer(
            objective=np.array([-0.1, -0.2]),
            cons_matrix=cons_matrix,
            lbound=np.zeros(2),
            ubound=np.ones(2),
            target_vol=1.0,
            cov=cov,
        )

        assert optimizer.is_dpp
        np.testing.assert_allclose(optimizer.x_value(), reference.x_value(), atol=1e-7)
        with pytest.raises(ValueError, match="target_vol must be non-negative"):
            optimizer.solve(target_vol=-0.1)

    def test_cached_optimizer_rejects_changed_constraint_exposures(self):
        optimizer = LPOptimizer(
            objective=np.array([-1.0, -2.0]),
            cons_matrix=np.array([[1.0, 1.0, 1.0, 1.0]]),
            lbound=np.zeros(2),
            ubound=np.ones(2),
        )
        with pytest.raises(ValueError, match="constraint exposures are fixed"):
            optimizer.solve(
                cons_matrix=np.array([[1.0, 2.0, 1.0, 1.0]]),
            )

    def test_factor_risk_uses_factor_covariance_directly(self, monkeypatch):
        matrix_shapes = []
        original_quad_form = cp.quad_form

        def record_quad_form(x, matrix):
            matrix_shapes.append(matrix.shape)
            return original_quad_form(x, matrix)

        monkeypatch.setattr(cp, "quad_form", record_quad_form)
        risk = _risk_expression(
            cp.Variable(3),
            cov=None,
            factor_cov=np.array([[0.5, -0.3], [-0.3, 0.7]]),
            factor_load=np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]),
            factor_special=np.array([0.1, 0.3, 0.2]),
        )

        assert risk.is_convex()
        assert matrix_shapes == [(2, 2)]

    def test_factor_risk_rejects_negative_specific_risk(self):
        with pytest.raises(ValueError, match="factor_special must be non-negative"):
            _risk_expression(
                cp.Variable(1),
                cov=None,
                factor_cov=np.array([[1.0]]),
                factor_load=np.array([[1.0]]),
                factor_special=np.array([-1.0]),
            )
