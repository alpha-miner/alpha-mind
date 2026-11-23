import numpy as np

from alphamind.data.neutralize import (
    ls_explain,
    ls_fit,
    ls_fit_pinv,
    ls_res,
    neutralize,
)
from alphamind.data.rank import percentile, rank
from alphamind.data.winsorize import (
    NormalWinsorizer,
    interp_values_1d,
    interp_values_2d,
    mask_values_1d,
    mask_values_2d,
    winsorize_normal,
)


def test_winsorize_python_and_jit_kernels_match():
    values = np.array([[-5.0, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 5.0], [5.0, 2.0]])
    mean_2d = np.zeros_like(values)
    std_2d = np.ones_like(values)
    mean_1d = np.zeros(2)
    std_1d = np.ones(2)
    groups = np.zeros(len(values), dtype=int)

    for compiled, python, args in (
        (mask_values_2d, mask_values_2d.py_func, (values, mean_2d, std_2d, 1)),
        (interp_values_2d, interp_values_2d.py_func, (values, groups, mean_2d, std_2d, 1, 0.5)),
        (mask_values_1d, mask_values_1d.py_func, (values, mean_1d, std_1d, 1)),
        (interp_values_1d, interp_values_1d.py_func, (values, mean_1d, std_1d, 1, 0.5)),
    ):
        np.testing.assert_allclose(python(*args), compiled(*args))

    np.testing.assert_allclose(
        mask_values_1d.py_func(values, mean_1d, std_1d, 1), np.clip(values, -1, 1)
    )


def test_normal_winsorizer_group_fit_and_transform():
    values = np.array([[1.0], [2.0], [10.0], [2.0], [3.0], [12.0]])
    groups = np.array([0, 0, 0, 1, 1, 1])
    for method in ("flat", "interp"):
        model = NormalWinsorizer(num_stds=1, method=method)
        model.fit(values, groups)
        np.testing.assert_allclose(
            model.transform(values, groups),
            winsorize_normal(values, num_stds=1, groups=groups, method=method),
        )


def test_neutralization_python_and_jit_kernels_match():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(8, 3))
    y = rng.normal(size=(8, 2))
    weights = np.linspace(1.0, 2.0, 8)
    coefficients = ls_fit.py_func(x, y, weights)

    np.testing.assert_allclose(coefficients, ls_fit(x, y, weights))
    np.testing.assert_allclose(ls_fit_pinv.py_func(x, y, weights), ls_fit_pinv(x, y, weights))
    np.testing.assert_allclose(ls_res.py_func(x, y, coefficients), ls_res(x, y, coefficients))
    np.testing.assert_allclose(ls_explain.py_func(x, coefficients), ls_explain(x, coefficients))

    residual, detail = neutralize(x, y, weights=weights, detail=True)
    np.testing.assert_allclose(residual, y - x @ coefficients)
    assert detail["exposure"].shape == (8, 3, 2)
    assert detail["explained"].shape == (8, 3, 2)


def test_neutralization_singular_matrices_use_pseudoinverse():
    x = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    groups = np.array([0, 0, 1, 1])

    ungrouped = neutralize(x, y)
    grouped = neutralize(x, y, groups=groups)
    assert np.isfinite(ungrouped).all()
    assert np.isfinite(grouped).all()
    np.testing.assert_allclose(ungrouped.ravel(), [0.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_one_dimensional_and_grouped_rank_percentile():
    values = np.array([3.0, 1.0, 2.0, 5.0])
    groups = np.array([0, 0, 0, 1])
    np.testing.assert_array_equal(rank(values).ravel(), [2, 0, 1, 3])
    np.testing.assert_allclose(percentile(values).ravel(), [2.0 / 3.0, 0.0, 1.0 / 3.0, 1.0])
    np.testing.assert_array_equal(rank(values, groups).ravel(), [2, 0, 1, 0])
    np.testing.assert_allclose(percentile(values, groups).ravel(), [1.0, 0.0, 0.5, 0.0])
    np.testing.assert_allclose(percentile(np.array([7.0])).ravel(), [0.0])
