import numpy as np
import pytest

from alphamind.utilities import (
    agg_abssum,
    agg_mean,
    agg_sqrsum,
    agg_std,
    agg_sum,
    aggregate,
    array_index,
    copy_value,
    decode,
    encode,
    group_mapping,
    groupby,
    list_eq,
    map_freq,
    require,
    scale_value,
    set_value,
    simple_abssum,
    simple_mean,
    simple_sqrsum,
    simple_std,
    simple_sum,
    transform,
)


def test_list_require_and_frequency_helpers():
    assert list_eq(None, [])
    assert list_eq((1, 2), [1, 2])
    assert not list_eq([1], [2])
    require(True, ValueError, "not raised")
    with pytest.raises(ValueError, match="invalid"):
        require(False, ValueError, "invalid")

    expected = {"1m": 21, "1w": 4, "2w": 9, "3w": 14, "4w": 19, "1d": 0, "5b": 4}
    for frequency, horizon in expected.items():
        assert map_freq(frequency) == horizon
    with pytest.raises(ValueError, match="Unrecognized freq"):
        map_freq("monthly")


def test_group_helpers_preserve_original_positions():
    groups = np.array([2, 1, 2, 1, 3])
    boundaries, order = groupby(groups)
    np.testing.assert_array_equal(boundaries, [1, 3, 5])
    np.testing.assert_array_equal(order, [1, 3, 0, 2, 4])
    np.testing.assert_array_equal(group_mapping(groups), [1, 0, 1, 0, 2])
    np.testing.assert_array_equal(group_mapping.py_func(groups), [1, 0, 1, 0, 2])

    levels = np.array([[0, 1], [1, 0]])
    for function in (set_value, set_value.py_func):
        matrix = np.zeros((2, 2))
        function(matrix, levels, 3.0)
        np.testing.assert_array_equal(matrix, [[3.0, 3.0], [3.0, 3.0]])


@pytest.mark.parametrize(
    ("function", "axis", "expected"),
    [
        (simple_sum, 0, [4.0, 2.0]),
        (simple_sum, 1, [-1.0, 7.0]),
        (simple_abssum, 0, [4.0, 6.0]),
        (simple_abssum, 1, [3.0, 7.0]),
        (simple_sqrsum, 0, np.sqrt([10.0, 20.0])),
        (simple_sqrsum, 1, np.sqrt([5.0, 25.0])),
        (simple_mean, 0, [2.0, 1.0]),
        (simple_mean, 1, [-0.5, 3.5]),
        (simple_std, 0, [np.sqrt(2.0), np.sqrt(18.0)]),
        (simple_std, 1, [np.sqrt(4.5), np.sqrt(0.5)]),
    ],
)
def test_simple_numpy_reductions(function, axis, expected):
    values = np.array([[1.0, -2.0], [3.0, 4.0]])
    for implementation in (function, function.py_func):
        np.testing.assert_allclose(implementation(values, axis=axis), expected)


@pytest.mark.parametrize(
    "function", [simple_sum, simple_abssum, simple_sqrsum, simple_mean, simple_std]
)
def test_simple_reductions_reject_unknown_axis(function):
    for implementation in (function, function.py_func):
        with pytest.raises(ValueError, match="axis value is not supported"):
            implementation(np.ones((2, 2)), axis=2)


@pytest.fixture
def grouped_values():
    groups = np.array([0, 1, 0, 1])
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    return groups, values


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("sum", [[6.0, 8.0], [10.0, 12.0]]),
        ("abssum", [[6.0, 8.0], [10.0, 12.0]]),
        ("sqrsum", np.sqrt([[26.0, 40.0], [58.0, 80.0]])),
        ("mean", [[3.0, 4.0], [5.0, 6.0]]),
        ("std", [[np.sqrt(8.0), np.sqrt(8.0)], [np.sqrt(8.0), np.sqrt(8.0)]]),
    ],
)
def test_group_aggregates(grouped_values, operation, expected):
    groups, values = grouped_values
    implementations = {
        "sum": agg_sum,
        "abssum": agg_abssum,
        "sqrsum": agg_sqrsum,
        "mean": agg_mean,
        "std": agg_std,
    }
    np.testing.assert_allclose(aggregate(groups, values, operation), expected)
    np.testing.assert_allclose(implementations[operation].py_func(groups, values), expected)


def test_transform_maps_or_scales_group_values(grouped_values):
    groups, values = grouped_values
    sums = aggregate(groups, values, "sum")
    np.testing.assert_allclose(transform(groups, values, "sum"), sums[groups])
    np.testing.assert_allclose(copy_value.py_func(groups, sums), sums[groups])
    np.testing.assert_allclose(
        transform(groups, values, "scale", scale=2.0),
        values / (aggregate(groups, values, "abssum")[groups] / 2.0),
    )
    np.testing.assert_allclose(
        scale_value.py_func(groups, aggregate(groups, values, "abssum"), values, 2.0),
        values / (aggregate(groups, values, "abssum")[groups] / 2.0),
    )
    np.testing.assert_allclose(
        transform(groups, values, "project"),
        values / aggregate(groups, values, "sqrsum")[groups],
    )
    for invalid in (aggregate, transform):
        with pytest.raises(ValueError, match="not recognized"):
            invalid(groups, values, "median")


def test_array_index_and_serialization_round_trip():
    labels = np.array([10, 20, 30])
    items = np.array([30, 10, 20])
    np.testing.assert_array_equal(array_index(labels, items), [2, 0, 1])
    np.testing.assert_array_equal(array_index.py_func(labels, items), [2, 0, 1])
    value = {"expression": "x + 1", "versions": [3, 9, 13]}
    assert decode(encode(value)) == value
