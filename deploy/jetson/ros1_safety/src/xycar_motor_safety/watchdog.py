"""Pure fail-closed command watchdog used by the ROS 1 adapter."""

from __future__ import annotations

import math

STOP_COMMAND = (0.0, 0.0)


class MotorCommandWatchdog:
    def __init__(self, timeout_sec: float) -> None:
        if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
            raise ValueError('timeout_sec must be finite and positive')
        self.timeout_sec = float(timeout_sec)
        self._last_command = STOP_COMMAND
        self._last_valid_monotonic: float | None = None

    def observe(self, values, now: float) -> bool:
        try:
            valid = (
                math.isfinite(now)
                and len(values) == 2
                and all(math.isfinite(float(value)) for value in values)
            )
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            self.invalidate()
            return False
        self._last_command = (float(values[0]), float(values[1]))
        self._last_valid_monotonic = float(now)
        return True

    def command(self, now: float) -> tuple[float, float]:
        if (
            self._last_valid_monotonic is None
            or not math.isfinite(now)
            or now < self._last_valid_monotonic
            or now - self._last_valid_monotonic >= self.timeout_sec
        ):
            return STOP_COMMAND
        return self._last_command

    def invalidate(self) -> None:
        self._last_command = STOP_COMMAND
        self._last_valid_monotonic = None
