import numpy as np
import pytest

from alphamind.data.arrow import from_numpy
from alphamind.portfolio.constraints import (
    BoundaryDirection,
    BoundaryImpl,
    BoundaryType,
    BoxBoundary,
    Constraints,
    LinearConstraints,
    create_box_bounds,
)


class TestConstraints:
    @staticmethod
    def test_constraints():
        cons = Constraints()
        test_exp = np.array([[1.0, 2.0], [3.0, 4.0]])
        test_names = np.array(["a", "b"])
        cons.add_exposure(test_names, test_exp)
        test_exp = np.array([[5.0], [6.0]])
        test_names = ["c"]
        cons.add_exposure(test_names, test_exp)
        np.testing.assert_array_almost_equal(
            np.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]), cons.risk_exp
        )
        risk_targets = cons.risk_targets()
        np.testing.assert_array_almost_equal(risk_targets[0], np.array([-np.inf, -np.inf, -np.inf]))
        np.testing.assert_array_almost_equal(risk_targets[1], np.array([np.inf, np.inf, np.inf]))
        cons.set_constraints("a", -0.1, 0.1)
        risk_targets = cons.risk_targets()
        np.testing.assert_array_almost_equal(risk_targets[0], np.array([-0.1, -np.inf, -np.inf]))
        np.testing.assert_array_almost_equal(risk_targets[1], np.array([0.1, np.inf, np.inf]))
        cons.set_constraints("c", -0.1, 0.1)
        risk_targets = cons.risk_targets()
        np.testing.assert_array_almost_equal(risk_targets[0], np.array([-0.1, -np.inf, -0.1]))
        np.testing.assert_array_almost_equal(risk_targets[1], np.array([0.1, np.inf, 0.1]))

    def test_absolute_box_boundary(self):
        lower = BoundaryImpl(BoundaryDirection.LOWER, BoundaryType.ABSOLUTE, -0.8)
        upper = BoundaryImpl(BoundaryDirection.UPPER, BoundaryType.ABSOLUTE, 1.1)
        bound = BoxBoundary(lower, upper)
        center = 2.2
        lower, upper = bound.bounds(center)

        assert lower == pytest.approx(1.4, abs=10.0 ** (-7))

        assert upper == pytest.approx(3.3, abs=10.0 ** (-7))

    def test_relative_box_boundary(self):
        lower = BoundaryImpl(BoundaryDirection.LOWER, BoundaryType.RELATIVE, 0.8)
        upper = BoundaryImpl(BoundaryDirection.UPPER, BoundaryType.RELATIVE, 1.1)
        bound = BoxBoundary(lower, upper)
        center = 2.2
        lower, upper = bound.bounds(center)

        assert lower == pytest.approx(1.76, abs=10.0 ** (-7))

        assert upper == pytest.approx(2.42, abs=10.0 ** (-7))

    def test_max_abs_relative_boundary(self):
        lower = BoundaryImpl(BoundaryDirection.LOWER, BoundaryType.MAXABSREL, (0.02, 0.2))
        upper = BoundaryImpl(BoundaryDirection.UPPER, BoundaryType.MAXABSREL, (0.02, 0.2))
        bound = BoxBoundary(lower, upper)
        center = 2.2
        lower, upper = bound.bounds(center)

        assert lower == pytest.approx(1.76, abs=10.0 ** (-7))

        assert upper == pytest.approx(2.64, abs=10.0 ** (-7))

    def test_min_abs_relative_boundary(self):
        lower = BoundaryImpl(BoundaryDirection.LOWER, BoundaryType.MINABSREL, (0.02, 0.2))
        upper = BoundaryImpl(BoundaryDirection.UPPER, BoundaryType.MINABSREL, (0.02, 0.2))
        bound = BoxBoundary(lower, upper)
        center = 2.2
        lower, upper = bound.bounds(center)

        assert lower == pytest.approx(2.18, abs=10.0 ** (-7))

        assert upper == pytest.approx(2.22, abs=10.0 ** (-7))

    def test_create_box_bounds_single_value(self):
        names = ["a", "b", "c"]
        b_type = BoundaryType.RELATIVE
        l_val = 0.8
        u_val = 1.1
        bounds = create_box_bounds(names, b_type, l_val, u_val)
        for key, bound in bounds.items():
            l_bound = bound.lower
            u_bound = bound.upper

            assert l_bound.b_type == b_type

            assert u_bound.b_type == b_type

            assert l_bound.val == pytest.approx(l_val, abs=10.0 ** (-7))

            assert u_bound.val == pytest.approx(u_val, abs=10.0 ** (-7))

    def test_create_box_bounds_multiple_values(self):
        names = ["a", "b", "c"]
        b_type = BoundaryType.RELATIVE
        l_val = [0.9, 0.8, 1.1]
        u_val = [1.1, 1.2, 1.3]
        bounds = create_box_bounds(names, b_type, l_val, u_val)
        for i, name in enumerate(names):
            bound = bounds[name]
            l_bound = bound.lower
            u_bound = bound.upper

            assert l_bound.b_type == b_type

            assert u_bound.b_type == b_type

            assert l_bound.val == pytest.approx(l_val[i], abs=10.0 ** (-7))

            assert u_bound.val == pytest.approx(u_val[i], abs=10.0 ** (-7))

    def test_linear_constraints(self):
        cons_mat = np.random.randn(100, 3)
        backbone = np.random.randn(100)
        names = ["a", "b", "c"]
        cons_mat = from_numpy(cons_mat, names)
        b_type = BoundaryType.ABSOLUTE
        l_val = -0.8
        u_val = 1.1
        bounds = create_box_bounds(names, b_type, l_val, u_val)
        constraints = LinearConstraints(bounds=bounds, cons_mat=cons_mat, backbone=backbone)
        l_bounds, u_bounds = constraints.risk_targets()
        risk_exp = constraints.risk_exp
        for i, name in enumerate(names):
            center = risk_exp[:, i] @ backbone

            assert center + l_val == pytest.approx(l_bounds[i], abs=10.0 ** (-7))

            assert center + u_val == pytest.approx(u_bounds[i], abs=10.0 ** (-7))
