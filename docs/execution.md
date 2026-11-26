# Executors and Execution Pipelines

Executors convert target holdings into planned execution holdings. Inputs and outputs are Arrow tables containing `code` and `weight`; turnover uses the L1 weight-change convention. Turnover and code alignment are performed directly with Arrow Compute without conversion to Python lists or dictionaries.

## NaiveExecutor

```python
from alphamind.execution.naiveexecutor import NaiveExecutor

executor = NaiveExecutor()
turn_over, executed = executor.execute(target)
```

This executor always accepts the complete target holdings. Initial turnover is the sum of absolute target weights; only after `set_current` is called are subsequent turnover values calculated relative to current holdings.

## ThresholdExecutor

```python
from alphamind.execution.thresholdexecutor import ThresholdExecutor

executor = ThresholdExecutor(turn_over_threshold=0.1, is_relative=False)
turn_over, executed = executor.execute(target)
executor.set_current(executed)
```

- `is_relative=False`: trade only when turnover reaches the absolute threshold.
- `is_relative=True`: trade only when turnover reaches `threshold * sum(current_weight)`.
- Below the threshold, return `(0.0, current_pos)` without updating to the new target.
- Initial execution returns the target and initial turnover; current holdings are set by the pipeline or caller.

Threshold comparison uses `>=`.

## TargetVolExecutor

```python
from alphamind.execution.targetvolexecutor import TargetVolExecutor

executor = TargetVolExecutor(window=30, target_vol=0.01)
turn_over, executed = executor.execute(target)
executor.set_current(executed)
executor.update({"return": realized_return})
```

No scaling is applied until the return window is full. Afterward, a multiplier is calculated from the return standard deviation and average historical leverage, then applied to target weights. Every `set_current` records current absolute leverage, and `update` appends realized portfolio returns.

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

Execution proceeds as follows:

1. The first executor receives target holdings, and each subsequent executor receives the previous result.
2. Final planned holdings become the shared current holdings of every executor.
3. The pipeline returns the turnover reported by the final executor and the final holdings.
4. `update` broadcasts the same data dictionary to every executor.

The pipeline calls `set_current` uniformly, so multi-executor scenarios should always run through the pipeline.

## Connecting to a portfolio builder

```python
import pyarrow as pa

# codes and optimized_weights must use identical ordering.
target = pa.table({"code": codes, "weight": optimized_weights})
turn_over, executed = pipeline.execute(target)
weights = executed["weight"].to_numpy(zero_copy_only=False)
```

Production applications should also model trading status, suspensions, price limits, minimum order sizes, and fees. Current executors implement only the weight-level rules above and do not simulate order matching.

See `run_file_backtest` in `examples/parquet_demo/demo.py` for a complete daily-execution example.
