import datetime as dt

import pytest

from alphamind.data.calendar import (
    BizDayConventions,
    DateGeneration,
    Period,
    _as_datetime,
    _shift_months,
    advanceDateByCalendar,
    bizDatesList,
    isBizDay,
    makeSchedule,
)


def test_period_and_datetime_parsing():
    period = Period("3w")
    assert period.length() == 3
    assert period.units() == "w"
    assert str(period) == "3w"
    assert str(Period(period)) == "3w"
    assert str(Period(length=2, units="M")) == "2m"
    with pytest.raises(ValueError, match="Unsupported period"):
        Period("1x")

    assert _as_datetime(dt.datetime(2026, 1, 5)) == dt.datetime(2026, 1, 5)
    assert _as_datetime(dt.date(2026, 1, 5)) == dt.datetime(2026, 1, 5)
    assert _as_datetime("20260105") == dt.datetime(2026, 1, 5)
    assert _as_datetime("2026-01-05 01:02:03") == dt.datetime(2026, 1, 5, 1, 2, 3)
    with pytest.raises(ValueError, match="Invalid date"):
        _as_datetime("not-a-date")


def test_calendar_adjustment_conventions():
    assert advanceDateByCalendar(
        "china.sse", "2026-01-01", "0b", BizDayConventions.Unadjusted
    ) == dt.datetime(2026, 1, 1)
    assert advanceDateByCalendar(
        "china.sse", "2026-01-01", "0b", BizDayConventions.Preceding
    ) == dt.datetime(2025, 12, 31)
    assert advanceDateByCalendar("china.sse", "2026-01-01", "0b") == dt.datetime(2026, 1, 5)
    assert advanceDateByCalendar(
        "china.sse", "2026-02-28", "0b", BizDayConventions.ModifiedFollowing
    ) == dt.datetime(2026, 2, 27)
    assert advanceDateByCalendar("china.sse", "2026-02-28", "0b") == dt.datetime(2026, 3, 2)
    assert advanceDateByCalendar("china.sse", "2026-01-05", "0b", None) == dt.datetime(2026, 1, 5)


def test_calendar_day_week_month_and_year_periods():
    assert advanceDateByCalendar(
        "china.sse", "2026-01-31", "1d", BizDayConventions.Unadjusted
    ) == dt.datetime(2026, 2, 1)
    assert advanceDateByCalendar("china.sse", "2026-01-05", "1w") == dt.datetime(2026, 1, 12)
    assert advanceDateByCalendar(
        "china.sse", "2026-01-31", "1m", BizDayConventions.Unadjusted
    ) == dt.datetime(2026, 2, 28)
    assert advanceDateByCalendar(
        "china.sse", "2024-02-29", "1y", BizDayConventions.Unadjusted
    ) == dt.datetime(2025, 2, 28)
    assert _shift_months(dt.datetime(2026, 1, 31), -1) == dt.datetime(2025, 12, 31)
    assert isBizDay("china.ib", "2026-01-05")


def test_schedule_backward_and_empty_ranges():
    expected = [dt.datetime(2025, 12, 31), dt.datetime(2026, 1, 5), dt.datetime(2026, 1, 6)]
    assert (
        makeSchedule(
            "2025-12-31",
            "2026-01-06",
            "1b",
            "china.sse",
            dateGenerationRule=DateGeneration.Backward,
        )
        == expected
    )
    assert makeSchedule("2026-01-07", "2026-01-06", "1b") == []
    assert bizDatesList("china.sse", "2026-01-07", "2026-01-06") == []
    with pytest.raises(ValueError, match="frequency must be positive"):
        makeSchedule("2026-01-05", "2026-01-06", "0b")
