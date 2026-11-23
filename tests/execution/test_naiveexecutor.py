import pyarrow as pa
import pytest

from alphamind.execution.naiveexecutor import NaiveExecutor


class TestNaiveExecutor:
    def test_naive_executor(self):
        target_pos = pa.table(
            {"code": [1, 2, 3], "weight": [0.2, 0.3, 0.5], "industry": ["a", "b", "c"]}
        )
        executor = NaiveExecutor()
        turn_over, executed_pos = executor.execute(target_pos)
        executor.set_current(executed_pos)

        assert turn_over == pytest.approx(1.0, abs=10.0 ** (-7))
        target_pos = pa.table(
            {"code": [1, 2, 4], "weight": [0.3, 0.2, 0.5], "industry": ["a", "b", "d"]}
        )
        turn_over, executed_pos = executor.execute(target_pos)
        executor.set_current(executed_pos)

        assert turn_over == pytest.approx(1.2, abs=10.0 ** (-7))
        target_pos = pa.table(
            {"code": [1, 3, 4], "weight": [0.3, 0.2, 0.5], "industry": ["a", "c", "d"]}
        )
        turn_over, executed_pos = executor.execute(target_pos)
        executor.set_current(executed_pos)

        assert turn_over == pytest.approx(0.4, abs=10.0 ** (-7))
