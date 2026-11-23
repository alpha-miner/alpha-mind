import pyarrow as pa
import pytest

from alphamind.execution.thresholdexecutor import ThresholdExecutor


class TestThresholdExecutor:
    def test_threshold_executor(self):
        target_pos = pa.table(
            {"code": [1, 2, 3], "weight": [0.2, 0.3, 0.5], "industry": ["a", "b", "c"]}
        )
        executor = ThresholdExecutor(turn_over_threshold=0.5)
        turn_over, executed_pos = executor.execute(target_pos)
        executor.set_current(executed_pos)

        assert target_pos.equals(executed_pos)

        assert turn_over == pytest.approx(1.0, abs=10.0 ** (-7))
        target_pos = pa.table(
            {"code": [1, 2, 4], "weight": [0.3, 0.2, 0.5], "industry": ["a", "b", "d"]}
        )
        turn_over, executed_pos = executor.execute(target_pos)
        executor.set_current(executed_pos)

        assert target_pos.equals(executed_pos)

        assert executed_pos.equals(executor.current_pos)

        assert turn_over == pytest.approx(1.2, abs=10.0 ** (-7))
        target_pos = pa.table(
            {"code": [1, 3, 4], "weight": [0.3, 0.2, 0.5], "industry": ["a", "c", "d"]}
        )
        turn_over, executed_pos2 = executor.execute(target_pos)
        executor.set_current(executed_pos2)

        assert executed_pos.equals(executed_pos2)

        assert turn_over == pytest.approx(0.0, abs=10.0 ** (-7))
