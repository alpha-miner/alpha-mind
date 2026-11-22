import numpy as np
import pyarrow as pa

from alphamind.data.arrow import column_values
from alphamind.execution.pipeline import ExecutionPipeline
from alphamind.execution.targetvolexecutor import TargetVolExecutor
from alphamind.execution.thresholdexecutor import ThresholdExecutor


class TestExecutionPipeline:
    def test_execution_pipeline(self):
        n = 100
        executor1 = TargetVolExecutor(window=60, target_vol=0.01)
        executor2 = ThresholdExecutor(turn_over_threshold=0.5)
        pipeline = ExecutionPipeline(executors=[executor1, executor2])
        rng = np.random.default_rng(42)
        returns = np.concatenate((rng.normal(0, 0.05, (2000, n)), rng.normal(0, 0.2, (2000, n))))
        codes = np.arange(n)
        for row in returns:
            weights = rng.integers(0, 100, n).astype(float)
            weights /= weights.sum()
            pos = pa.table({"code": codes, "weight": weights})
            _, executed_pos = pipeline.execute(pos)
            daily_return = row @ column_values(executed_pos, "weight")
            pipeline.update(data_dict={"return": daily_return})

        assert len(executor1.returns) == 60
