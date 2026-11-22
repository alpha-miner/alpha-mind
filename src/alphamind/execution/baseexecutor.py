import abc

import numpy as np
import pyarrow as pa

from alphamind.data.arrow import as_arrow


class ExecutorBase(metaclass=abc.ABCMeta):
    def __init__(self):
        self.current_pos = None

    @abc.abstractmethod
    def execute(self, target_pos) -> pa.Table:
        pass

    @staticmethod
    def calc_turn_over(target_pos, current_pos) -> float:
        target = as_arrow(target_pos)
        current = as_arrow(current_pos)
        target_weights = dict(zip(target["code"].to_pylist(), target["weight"].to_pylist()))
        current_weights = dict(zip(current["code"].to_pylist(), current["weight"].to_pylist()))
        codes = set(target_weights).union(current_weights)
        return float(
            np.sum(
                [
                    abs(target_weights.get(code, 0.0) - current_weights.get(code, 0.0))
                    for code in codes
                ]
            )
        )

    def set_current(self, current_pos):
        self.current_pos = as_arrow(current_pos)

    def update(self, data_dict: dict):
        pass
