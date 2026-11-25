from typing import Tuple

import pyarrow as pa

from alphamind.data.arrow import as_arrow
from alphamind.execution.baseexecutor import ExecutorBase, _sum_absolute


class NaiveExecutor(ExecutorBase):
    def execute(self, target_pos) -> Tuple[float, pa.Table]:
        target_pos = as_arrow(target_pos)
        if self.current_pos is None or self.current_pos.num_rows == 0:
            turn_over = _sum_absolute(target_pos["weight"])
        else:
            turn_over = self.calc_turn_over(target_pos, self.current_pos)
        self.current_pos = target_pos
        return turn_over, target_pos
