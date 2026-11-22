from typing import List, Tuple

import pyarrow as pa

from alphamind.execution.baseexecutor import ExecutorBase


class ExecutionPipeline:
    def __init__(self, executors: List[ExecutorBase]):
        self.executors = executors

    def execute(self, target_pos) -> Tuple[float, pa.Table]:
        turn_over, planned_pos = 0.0, target_pos
        for executor in self.executors:
            turn_over, planned_pos = executor.execute(planned_pos)
        executed_pos = planned_pos
        for executor in self.executors:
            executor.set_current(executed_pos)
        return turn_over, executed_pos

    def update(self, data_dict):
        for executor in self.executors:
            executor.update(data_dict=data_dict)
