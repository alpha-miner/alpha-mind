import numpy as np
import pytest

from alphamind.utilities import (
    aggregate,
    array_index,
    decode,
    encode,
    group_mapping,
    groupby,
    list_eq,
    map_freq,
    require,
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

    matrix = np.zeros((2, 2))
    set_value(matrix, np.array([[0, 1], [1, 0]]), 3.0)
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
    np.testing.assert_allclose(function(values, axis=axis), expected)


@pytest.mark.parametrize(
    "function", [simple_sum, simple_abssum, simple_sqrsum, simple_mean, simple_std]
)
def test_simple_reductions_reject_unknown_axis(function):
    with pytest.raises(ValueError, match="axis value is not supported"):
        function(np.ones((2, 2)), axis=2)


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
    np.testing.assert_allclose(aggregate(groups, values, operation), expected)


def test_transform_maps_or_scales_group_values(grouped_values):
    groups, values = grouped_values
    sums = aggregate(groups, values, "sum")
    np.testing.assert_allclose(transform(groups, values, "sum"), sums[groups])
    np.testing.assert_allclose(
        transform(groups, values, "scale", scale=2.0),
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
    np.testing.assert_array_equal(
        array_index(np.array([10, 20, 30]), np.array([30, 10, 20])), [2, 0, 1]
    )
    value = {"expression": "x + 1", "versions": [3, 9, 13]}
    assert decode(encode(value)) == value
