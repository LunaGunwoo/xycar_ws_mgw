import math

from xycar_motor_safety.watchdog import MotorCommandWatchdog, STOP_COMMAND


def test_watchdog_accepts_fresh_finite_two_value_commands():
    watchdog = MotorCommandWatchdog(0.25)
    assert watchdog.observe([-18.0, 7.0], 1.0)
    assert watchdog.command(1.24) == (-18.0, 7.0)


def test_watchdog_stops_on_stale_invalid_or_reversed_time():
    watchdog = MotorCommandWatchdog(0.25)
    assert watchdog.command(1.0) == STOP_COMMAND
    watchdog.observe([1.0, 2.0], 1.0)
    assert watchdog.command(1.25) == STOP_COMMAND
    watchdog.observe([1.0, 2.0], 2.0)
    assert watchdog.command(1.9) == STOP_COMMAND
    for invalid in ([1.0], [1.0, 2.0, 3.0], [math.nan, 2.0]):
        assert not watchdog.observe(invalid, 3.0)
        assert watchdog.command(3.0) == STOP_COMMAND
