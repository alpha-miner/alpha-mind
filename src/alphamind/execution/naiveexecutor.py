from typing import Tuple

import numpy as np
import pyarrow as pa

from alphamind.data.arrow import as_arrow
from alphamind.execution.baseexecutor import ExecutorBase


class NaiveExecutor(ExecutorBase):
    def execute(self, target_pos) -> Tuple[float, pa.Table]:
        target_pos = as_arrow(target_pos)
        if self.current_pos is None or self.current_pos.num_rows == 0:
            turn_over = float(np.abs(target_pos["weight"].to_numpy(zero_copy_only=False)).sum())
        else:
            turn_over = self.calc_turn_over(target_pos, self.current_pos)
        self.current_pos = target_pos
        return turn_over, target_pos
