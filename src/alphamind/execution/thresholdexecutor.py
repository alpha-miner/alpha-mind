from typing import Tuple

import pyarrow as pa
import pyarrow.compute as pc

from alphamind.data.arrow import as_arrow
from alphamind.execution.baseexecutor import ExecutorBase, _sum_absolute


class ThresholdExecutor(ExecutorBase):
    def __init__(self, turn_over_threshold: float, is_relative=True):
        super().__init__()
        self.threshold = turn_over_threshold
        self.is_relative = is_relative

    def execute(self, target_pos) -> Tuple[float, pa.Table]:
        target_pos = as_arrow(target_pos)
        if self.current_pos is None or self.current_pos.num_rows == 0:
            return _sum_absolute(target_pos["weight"]), target_pos
        turn_over = self.calc_turn_over(target_pos, self.current_pos)
        if self.is_relative:
            current_weight = float(pc.sum(self.current_pos["weight"]).as_py())
            is_break = turn_over >= self.threshold * current_weight
        else:
            is_break = turn_over >= self.threshold
        return (turn_over, target_pos) if is_break else (0.0, self.current_pos)
