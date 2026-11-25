# 执行器与执行管道

执行器把目标持仓转换为计划执行持仓。输入输出均为包含 `code`、`weight` 的 Arrow 表；换手采用 L1 权重变化口径。换手与代码对齐直接由 Arrow Compute 完成，不转换为 Python 列表或字典。

## NaiveExecutor

```python
from alphamind.execution.naiveexecutor import NaiveExecutor

executor = NaiveExecutor()
turn_over, executed = executor.execute(target)
```

该执行器总是接受完整目标持仓。首次执行换手为目标绝对权重之和；只有调用 `set_current` 后，后续执行才相对于当前持仓计算换手。

## ThresholdExecutor

```python
from alphamind.execution.thresholdexecutor import ThresholdExecutor

executor = ThresholdExecutor(turn_over_threshold=0.1, is_relative=False)
turn_over, executed = executor.execute(target)
executor.set_current(executed)
```

- `is_relative=False`：换手达到绝对阈值才交易。
- `is_relative=True`：换手达到 `threshold * sum(current_weight)` 才交易。
- 未达阈值时返回 `(0.0, current_pos)`，不会更新为新目标。
- 首次执行返回目标及初始换手，当前持仓由管道或调用方设置。

阈值比较使用 `>=`。

## TargetVolExecutor

```python
from alphamind.execution.targetvolexecutor import TargetVolExecutor

executor = TargetVolExecutor(window=30, target_vol=0.01)
turn_over, executed = executor.execute(target)
executor.set_current(executed)
executor.update({"return": realized_return})
```

收益窗口未满时不缩放。窗口满后，根据收益标准差和平均历史杠杆计算乘数，再缩放目标权重。每次 `set_current` 会记录当前绝对杠杆，`update` 追加已实现组合收益。

## ExecutionPipeline

```python
from alphamind.execution.pipeline import ExecutionPipeline

pipeline = ExecutionPipeline(
    [
        TargetVolExecutor(window=30, target_vol=0.01),
        ThresholdExecutor(turn_over_threshold=0.02, is_relative=False),
    ]
)

turn_over, executed = pipeline.execute(target)
pipeline.update({"return": portfolio_return})
```

执行顺序为：

1. 第一个执行器接收目标持仓，后续执行器接收上一步结果。
2. 最终计划持仓作为所有执行器的共同当前持仓。
3. 返回最后一个执行器给出的换手和最终持仓。
4. `update` 将同一数据字典广播给所有执行器。

管道会统一调用 `set_current`，因此推荐多执行器场景始终通过管道执行。

## 与组合构建器连接

```python
import pyarrow as pa

# codes 和 optimized_weights 必须顺序一致。
target = pa.table({"code": codes, "weight": optimized_weights})
turn_over, executed = pipeline.execute(target)
weights = executed["weight"].to_numpy(zero_copy_only=False)
```

实际应用中还应加入交易状态、停牌、涨跌停、最小交易单位和费用模型；当前执行器只实现上述权重层面的规则，不模拟订单撮合。

完整逐日执行示例见 `examples/parquet_demo/demo.py` 的 `run_file_backtest`。
