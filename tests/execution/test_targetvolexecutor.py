from collections import deque

import numpy as np
import pyarrow as pa

from alphamind.data.arrow import column_values
from alphamind.execution.targetvolexecutor import TargetVolExecutor


class TestTargetVolExecutor:
    def test_target_vol_executor(self):
        n = 100
        window = 30
        target_vol = 0.01
        executor = TargetVolExecutor(window=window, target_vol=target_vol)
        rng = np.random.default_rng(42)
        return_total = np.concatenate(
            (rng.normal(0, 0.05, (2000, n)), rng.normal(0, 0.2, (2000, n)))
        )
        weights = np.ones(n) / n
        codes = np.arange(n)
        returns = deque(maxlen=window)
        leverages = deque(maxlen=window)
        for index, row in enumerate(return_total):
            pos = pa.table({"code": codes, "weight": weights})
            _, executed_pos = executor.execute(pos)
            actual_weights = column_values(executed_pos, "weight")
            if index >= window:
                expected = weights * target_vol / np.std(returns, ddof=1) * np.mean(leverages)
                np.testing.assert_allclose(actual_weights, expected, rtol=1e-10, atol=1e-12)
            else:
                np.testing.assert_array_equal(actual_weights, weights)
            executor.set_current(executed_pos)
            leverages.append(np.abs(actual_weights).sum())
            daily_return = row @ weights
            executor.update(data_dict={"return": daily_return})
            returns.append(daily_return)
