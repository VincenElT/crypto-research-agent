import pytest

from crypto_research.__main__ import seconds_until_next_candle


def test_watch_waits_until_just_after_the_next_candle_boundary() -> None:
    assert seconds_until_next_candle(3_599.0, 3_600) == 3.0
    assert seconds_until_next_candle(3_600.5, 3_600) == 3_601.5


@pytest.mark.parametrize(
    ("now", "timeframe", "delay"),
    [(100.0, 0, 2), (100.0, 60, -1)],
)
def test_watch_rejects_invalid_interval(now: float, timeframe: int, delay: int) -> None:
    with pytest.raises(ValueError):
        seconds_until_next_candle(now, timeframe, safety_delay=delay)
