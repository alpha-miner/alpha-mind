# 交易日与调仓日程

交易日功能位于 `alphamind.data.calendar`，底层使用 dal-python。默认中国市场日历名为 `china.sse`，内部映射到 DAL 的 `CN.SSE`；`china.ib` 映射到 `CN.IB`。

## 判断交易日与列出区间

```python
from alphamind.data.calendar import bizDatesList, isBizDay

assert isBizDay("china.sse", "2026-01-05")
dates = bizDatesList("china.sse", "2026-01-05", "2026-01-16")
```

`bizDatesList` 返回 `datetime.datetime` 列表，包含起止日期；当开始日期晚于结束日期时返回空列表。输入支持 `date`、`datetime`、`YYYY-MM-DD`、`YYYY%m%d` 和 `YYYY-MM-DD HH:MM:SS` 字符串。

## 日期偏移

```python
from alphamind.data.calendar import BizDayConventions, advanceDateByCalendar

next_business_day = advanceDateByCalendar("china.sse", "2026-01-05", "1b")
next_week = advanceDateByCalendar("china.sse", "2026-01-05", "1w")
previous_business_day = advanceDateByCalendar("china.sse", "2026-01-05", "-1b")
```

`Period` 支持以下单位：

| 单位 | 含义 | 示例 |
| --- | --- | --- |
| `b` | 交易日 | `5b`、`-3b` |
| `d` | 自然日 | `1d` |
| `w` | 周 | `2w` |
| `m` | 月 | `1m` |
| `y` | 年 | `1y` |

非交易日单位完成日历偏移后，再按交易日约定调整。可用约定为 `Following`、`Preceding`、`ModifiedFollowing` 和 `Unadjusted`。

## 生成调仓日程

```python
from alphamind.data.calendar import DateGeneration, makeSchedule

weekly = makeSchedule(
    "2026-01-05",
    "2026-03-31",
    "1w",
    calendar="china.sse",
    dateRule="following",
    dateGenerationRule=DateGeneration.Forward,
)
monthly = makeSchedule(
    "2026-01-01",
    "2026-12-31",
    "1m",
    dateGenerationRule=DateGeneration.Backward,
)
```

- `Forward` 从调整后的开始日期向结束日期推进。
- `Backward` 从调整后的结束日期向前倒推，再按时间正序返回。
- 频率必须大于零；偏移函数本身允许负周期。
- 返回值是 Python 时间列表，需要表格输出时显式构造 Arrow 表。

```python
import pyarrow as pa
from alphamind.data import write_parquet

schedule = pa.table({"trade_date": weekly})
write_parquet(schedule, "schedule.parquet")
```

## 收益区间与频率

`SqlEngine.fetch_dx_return*` 使用 `horizon` 和 `offset` 计算未来收益。`Strategy` 通过 `map_freq(freq)` 将调仓频率映射为收益持有期。对数收益转简单收益：

```python
import numpy as np

simple_return = np.exp(dx) - 1.0
```

交易日函数只负责日期，不会验证本地 Parquet 是否真正包含这些日期；文件研究流程仍需将日程与数据键连接或过滤。
