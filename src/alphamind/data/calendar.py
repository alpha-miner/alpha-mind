from __future__ import annotations

import calendar as month_calendar
import datetime as dt
from functools import lru_cache


class BizDayConventions:
    Following = "following"
    Preceding = "preceding"
    ModifiedFollowing = "modified_following"
    Unadjusted = "unadjusted"


class DateGeneration:
    Forward = "forward"
    Backward = "backward"


class Period:
    def __init__(self, frequency=None, length=None, units=None):
        if length is not None and units is not None:
            self._length = int(length)
            self._units = units.lower()
            return
        if isinstance(frequency, Period):
            self._length = frequency.length()
            self._units = frequency.units()
            return
        text = str(frequency).strip().lower()
        self._length = int(text[:-1])
        self._units = text[-1]
        if self._units not in ("b", "d", "w", "m", "y"):
            raise ValueError(f"Unsupported period: {frequency}")

    def length(self):
        return self._length

    def units(self):
        return self._units

    def __str__(self):
        return f"{self._length}{self._units}"


@lru_cache(maxsize=16)
def _holidays(calendar: str):
    import dal

    aliases = {"china.sse": "CN.SSE", "china.ib": "CN.IB"}
    return dal.Holidays_(aliases.get(calendar.lower(), calendar))


def _as_datetime(value) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time())
    text = str(value)
    for pattern in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(text, pattern)
        except ValueError:
            pass
    raise ValueError(f"Invalid date: {value}")


def _dal_date(value):
    import dal

    value = _as_datetime(value)
    return dal.Date_(value.year, value.month, value.day)


def _adjust(value, calendar, convention):
    import dal

    if convention in (None, BizDayConventions.Unadjusted):
        return _as_datetime(value)
    holidays = _holidays(calendar)
    current = _dal_date(value)
    if dal.Is_BizDay(holidays, current):
        return _as_datetime(current)
    if convention == BizDayConventions.Preceding:
        return _as_datetime(dal.PrevBizDay(holidays, current))
    adjusted = _as_datetime(dal.NextBizDay(holidays, current))
    if (
        convention == BizDayConventions.ModifiedFollowing
        and adjusted.month != _as_datetime(value).month
    ):
        adjusted = _as_datetime(dal.PrevBizDay(holidays, current))
    return adjusted


def _shift_months(value: dt.datetime, months: int) -> dt.datetime:
    month_index = value.year * 12 + value.month - 1 + months
    year, month = divmod(month_index, 12)
    month += 1
    day = min(value.day, month_calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def advanceDateByCalendar(calendar, ref_date, period, convention=BizDayConventions.Following):
    import dal

    period = Period(period)
    length = period.length()
    units = period.units()
    current = _as_datetime(ref_date)
    if units == "b":
        holidays = _holidays(calendar)
        native = _dal_date(current)
        for _ in range(abs(length)):
            native = (
                dal.NextBizDay(holidays, native.AddDays(1))
                if length > 0
                else dal.PrevBizDay(holidays, native.AddDays(-1))
            )
        current = _as_datetime(native)
        if length == 0:
            current = _adjust(current, calendar, convention)
        return current
    if units == "d":
        shifted = current + dt.timedelta(days=length)
    elif units == "w":
        shifted = current + dt.timedelta(weeks=length)
    elif units == "m":
        shifted = _shift_months(current, length)
    else:
        shifted = _shift_months(current, 12 * length)
    return _adjust(shifted, calendar, convention)


def isBizDay(calendar, ref_date) -> bool:
    import dal

    return dal.Is_BizDay(_holidays(calendar), _dal_date(ref_date))


def bizDatesList(calendar, start_date, end_date):
    start = _as_datetime(start_date)
    end = _as_datetime(end_date)
    if start > end:
        return []
    dates = []
    current = start
    while current <= end:
        if isBizDay(calendar, current):
            dates.append(current)
        current += dt.timedelta(days=1)
    return dates


def makeSchedule(
    start_date,
    end_date,
    frequency,
    calendar="china.sse",
    dateRule=BizDayConventions.Following,
    dateGenerationRule=DateGeneration.Forward,
):
    start = _as_datetime(start_date)
    end = _as_datetime(end_date)
    if start > end:
        return []
    period = Period(frequency)
    if period.length() <= 0:
        raise ValueError("Schedule frequency must be positive")
    if dateGenerationRule == DateGeneration.Backward:
        current = _adjust(end, calendar, BizDayConventions.Preceding)
        dates = []
        while current >= start:
            dates.append(current)
            current = advanceDateByCalendar(
                calendar,
                current,
                Period(length=-period.length(), units=period.units()),
                BizDayConventions.Preceding,
            )
        return list(reversed(dates))
    current = _adjust(start, calendar, dateRule)
    dates = []
    while current <= end:
        dates.append(current)
        current = advanceDateByCalendar(calendar, current, period, dateRule)
    return dates
