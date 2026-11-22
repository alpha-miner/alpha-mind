import datetime as dt

from alphamind.data.calendar import advanceDateByCalendar, bizDatesList, isBizDay, makeSchedule


def test_dal_china_sse_calendar():
    assert not isBizDay("china.sse", "2026-01-01")
    assert isBizDay("china.sse", "2026-01-05")
    assert advanceDateByCalendar("china.sse", "2025-12-31", "1b") == dt.datetime(2026, 1, 5)
    assert advanceDateByCalendar("china.sse", "2026-01-05", "-1b") == dt.datetime(2025, 12, 31)


def test_dal_schedule_and_business_date_list():
    schedule = makeSchedule("2025-12-31", "2026-01-06", "1b", "china.sse")

    assert schedule == [dt.datetime(2025, 12, 31), dt.datetime(2026, 1, 5), dt.datetime(2026, 1, 6)]
    assert bizDatesList("china.sse", "2026-01-01", "2026-01-06") == [
        dt.datetime(2026, 1, 5),
        dt.datetime(2026, 1, 6),
    ]
