import abc

import pyarrow as pa
import pyarrow.compute as pc

from alphamind.data.arrow import as_arrow


def _sum_absolute(values) -> float:
    total = pc.sum(pc.abs(values)).as_py()
    return 0.0 if total is None else float(total)


def _align_weights(codes, position_codes, position_weights):
    reverse_order = pa.array(range(len(position_codes) - 1, -1, -1), type=pa.int64())
    reversed_codes = pc.take(position_codes, reverse_order)
    reversed_weights = pc.take(position_weights, reverse_order)
    indices = pc.index_in(codes, value_set=reversed_codes)
    weights = pc.take(reversed_weights, indices)
    return pc.fill_null(weights, pa.scalar(0, type=weights.type))


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
        if target.num_rows == 0 and current.num_rows == 0:
            return 0.0

        target_codes = target["code"].combine_chunks()
        current_codes = current["code"].combine_chunks()
        if pa.types.is_null(target_codes.type) and not pa.types.is_null(current_codes.type):
            target_codes = pc.cast(target_codes, current_codes.type)
        elif pa.types.is_null(current_codes.type) and not pa.types.is_null(target_codes.type):
            current_codes = pc.cast(current_codes, target_codes.type)

        target_weights = target["weight"].combine_chunks()
        current_weights = current["weight"].combine_chunks()
        if target.num_rows == 0 and pa.types.is_null(target_weights.type):
            target_weights = pc.cast(target_weights, current_weights.type)
        elif current.num_rows == 0 and pa.types.is_null(current_weights.type):
            current_weights = pc.cast(current_weights, target_weights.type)

        same_unique_order = target_codes.equals(current_codes) and len(
            pc.unique(target_codes)
        ) == len(target_codes)
        if same_unique_order:
            difference = pc.subtract(target_weights, current_weights)
        else:
            codes = pc.unique(pa.concat_arrays([target_codes, current_codes]))
            difference = pc.subtract(
                _align_weights(codes, target_codes, target_weights),
                _align_weights(codes, current_codes, current_weights),
            )
        return _sum_absolute(difference)

    def set_current(self, current_pos):
        self.current_pos = as_arrow(current_pos)

    def update(self, data_dict: dict):
        pass
