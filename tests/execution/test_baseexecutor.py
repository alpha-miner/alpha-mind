import pyarrow as pa
import pytest

from alphamind.execution.baseexecutor import ExecutorBase


@pytest.mark.parametrize(
    ("target_codes", "target_weights", "current_codes", "current_weights", "expected"),
    [
        ([1, 2, 3], [0.2, 0.3, 0.5], [1, 2, 3], [0.1, 0.6, 0.3], 0.6),
        (["A", "B", "C"], [0.2, 0.3, 0.5], ["C", "A", "B"], [0.5, 0.1, 0.3], 0.1),
        (["A", "B"], [0.4, 0.6], ["B", "C"], [0.2, 0.5], 1.3),
        (["A", "B"], [0.4, 0.6], ["C", "D"], [0.2, 0.3], 1.5),
        (["A", "A", "B"], [0.2, 0.3, 0.4], ["A", "C"], [0.5, 0.2], 0.8),
        (["A"], [0.2], ["A", "A"], [0.1, 0.5], 0.3),
        (["A", "A", "B"], [0.2, 0.3, 0.4], ["A", "A", "B"], [0.1, 0.5, 0.2], 0.4),
    ],
    ids=[
        "same-order",
        "reordered",
        "partial-overlap",
        "disjoint",
        "target-duplicates",
        "current-duplicates",
        "same-order-duplicates",
    ],
)
def test_calc_turn_over(target_codes, target_weights, current_codes, current_weights, expected):
    target = pa.table({"code": target_codes, "weight": target_weights})
    current = pa.table({"code": current_codes, "weight": current_weights})

    assert ExecutorBase.calc_turn_over(target, current) == pytest.approx(expected)


def test_calc_turn_over_handles_empty_positions_with_inferred_or_explicit_types():
    untyped_empty = pa.table({"code": [], "weight": []})
    typed_empty = pa.table(
        {
            "code": pa.array([], type=pa.string()),
            "weight": pa.array([], type=pa.float64()),
        }
    )
    positions = pa.table({"code": ["A", "B"], "weight": [0.2, -0.3]})

    assert ExecutorBase.calc_turn_over(untyped_empty, untyped_empty) == 0.0
    assert ExecutorBase.calc_turn_over(typed_empty, typed_empty) == 0.0
    assert ExecutorBase.calc_turn_over(untyped_empty, positions) == pytest.approx(0.5)
    assert ExecutorBase.calc_turn_over(positions, untyped_empty) == pytest.approx(0.5)


def test_calc_turn_over_handles_chunked_columns():
    target = pa.table(
        {
            "code": pa.chunked_array([["A"], ["B", "C"]]),
            "weight": pa.chunked_array([[0.2], [0.3, 0.5]]),
        }
    )
    current = pa.table(
        {
            "code": pa.chunked_array([["C", "A"], ["B"]]),
            "weight": pa.chunked_array([[0.5, 0.1], [0.3]]),
        }
    )

    assert ExecutorBase.calc_turn_over(target, current) == pytest.approx(0.1)


def test_calc_turn_over_handles_null_codes():
    target = pa.table(
        {
            "code": pa.array([None, "A"], type=pa.string()),
            "weight": [0.2, 0.3],
        }
    )
    current = pa.table({"code": pa.array([None]), "weight": [0.1]})

    assert ExecutorBase.calc_turn_over(target, current) == pytest.approx(0.4)
