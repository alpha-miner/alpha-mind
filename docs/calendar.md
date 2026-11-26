# Trading Calendars and Rebalancing Schedules

Trading-calendar functionality is located in `alphamind.data.calendar` and uses dal-python. The default Chinese market calendar is named `china.sse`, which maps internally to DAL's `CN.SSE`; `china.ib` maps to `CN.IB`.

## Checking trading days and listing ranges

```python
from alphamind.data.calendar import bizDatesList, isBizDay

assert isBizDay("china.sse", "2026-01-05")
dates = bizDatesList("china.sse", "2026-01-05", "2026-01-16")
```

`bizDatesList` returns a list of `datetime.datetime` values including both endpoints. It returns an empty list when the start date is later than the end date. Inputs may be `date`, `datetime`, `YYYY-MM-DD`, `YYYY%m%d`, or `YYYY-MM-DD HH:MM:SS` strings.

## Date offsets

```python
from alphamind.data.calendar import BizDayConventions, advanceDateByCalendar

next_business_day = advanceDateByCalendar("china.sse", "2026-01-05", "1b")
next_week = advanceDateByCalendar("china.sse", "2026-01-05", "1w")
previous_business_day = advanceDateByCalendar("china.sse", "2026-01-05", "-1b")
```

`Period` supports these units:

| Unit | Meaning | Examples |
| --- | --- | --- |
| `b` | Business day | `5b`, `-3b` |
| `d` | Calendar day | `1d` |
| `w` | Week | `2w` |
| `m` | Month | `1m` |
| `y` | Year | `1y` |

After a non-business-day unit is applied, the result is adjusted according to a business-day convention. Available conventions are `Following`, `Preceding`, `ModifiedFollowing`, and `Unadjusted`.

## Generating a rebalancing schedule

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

- `Forward` advances from the adjusted start date toward the end date.
- `Backward` works backward from the adjusted end date, then returns values in chronological order.
- Frequency must be positive; the offset function itself permits negative periods.
- The return value is a Python list of times; construct an Arrow table explicitly when tabular output is required.

```python
import pyarrow as pa
from alphamind.data import write_parquet

schedule = pa.table({"trade_date": weekly})
write_parquet(schedule, "schedule.parquet")
```

## Return horizons and frequency

`SqlEngine.fetch_dx_return*` uses `horizon` and `offset` to calculate forward returns. `Strategy` maps rebalancing frequency to a return holding period through `map_freq(freq)`. Convert logarithmic returns to simple returns with:

```python
import numpy as np

simple_return = np.exp(dx) - 1.0
```

Calendar functions handle dates only; they do not verify that local Parquet data actually contains those dates. File-based research workflows must still join or filter the schedule against the data keys.
