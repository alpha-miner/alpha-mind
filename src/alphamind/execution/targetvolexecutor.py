from collections import deque
from typing import Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from alphamind.data.arrow import as_arrow
from alphamind.execution.baseexecutor import ExecutorBase, _sum_absolute


class TargetVolExecutor(ExecutorBase):
    def __init__(self, window=30, target_vol=0.01):
        super().__init__()
        self.window = window
        self.returns = deque(maxlen=window)
        self.leverages = deque(maxlen=window)
        self.target_vol = target_vol
        self.multiplier = 1.0

    def execute(self, target_pos) -> Tuple[float, pa.Table]:
        target_pos = as_arrow(target_pos)
        if len(self.returns) < self.window:
            if self.current_pos is None or self.current_pos.num_rows == 0:
                turn_over = _sum_absolute(target_pos["weight"])
            else:
                turn_over = self.calc_turn_over(target_pos, self.current_pos)
            return turn_over, target_pos
        volatility = float(np.std(self.returns, ddof=1))
        leverage = float(np.mean(self.leverages))
        self.multiplier = self.target_vol / volatility * leverage if volatility > 0 else 1.0
        weights = pc.multiply(target_pos["weight"], self.multiplier)
        candidate_pos = target_pos.set_column(
            target_pos.schema.get_field_index("weight"), "weight", weights
        )
        turn_over = self.calc_turn_over(candidate_pos, self.current_pos)
        return turn_over, candidate_pos

    def set_current(self, current_pos):
        super().set_current(current_pos)
        self.leverages.append(_sum_absolute(self.current_pos["weight"]))

    def update(self, data_dict: dict):
        self.returns.append(float(data_dict["return"]))
