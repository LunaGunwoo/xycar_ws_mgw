"""Pure fail-closed command watchdog used by the ROS 1 adapter."""

from __future__ import annotations

import math

STOP_COMMAND = (0.0, 0.0)
INPUT_ANGLE_MIN = -100.0
INPUT_ANGLE_MAX = 100.0
DRIVER_ANGLE_MIN = -50.0
DRIVER_ANGLE_MAX = 50.0


class MotorCommandWatchdog:
    def __init__(
        self,
        timeout_sec: float,
        *,
        input_angle_min: float = INPUT_ANGLE_MIN,
        input_angle_max: float = INPUT_ANGLE_MAX,
        driver_angle_min: float = DRIVER_ANGLE_MIN,
        driver_angle_max: float = DRIVER_ANGLE_MAX,
    ) -> None:
        if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
            raise ValueError('timeout_sec must be finite and positive')
        ranges = tuple(
            float(value)
            for value in (
                input_angle_min,
                input_angle_max,
                driver_angle_min,
                driver_angle_max,
            )
        )
        if not all(math.isfinite(value) for value in ranges):
            raise ValueError('steering contract ranges must be finite')
        if ranges != (
            INPUT_ANGLE_MIN,
            INPUT_ANGLE_MAX,
            DRIVER_ANGLE_MIN,
            DRIVER_ANGLE_MAX,
        ):
            raise ValueError(
                'normalized_percent_v2 requires input [-100,100] and '
                'driver [-50,50]'
            )
        self.timeout_sec = float(timeout_sec)
        self.input_angle_min = ranges[0]
        self.input_angle_max = ranges[1]
        self.driver_angle_min = ranges[2]
        self.driver_angle_max = ranges[3]
        self._last_command = STOP_COMMAND
        self._last_valid_monotonic: float | None = None

    def observe(self, values, now: float) -> bool:
        try:
            if len(values) != 2:
                raise ValueError('motor command must contain two values')
            angle = float(values[0])
            speed = float(values[1])
            valid = (
                math.isfinite(now)
                and math.isfinite(angle)
                and math.isfinite(speed)
                and self.input_angle_min <= angle <= self.input_angle_max
            )
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            self.invalidate()
            return False
        driver_angle = (
            angle / self.input_angle_max * self.driver_angle_max
        )
        self._last_command = (driver_angle, speed)
        self._last_valid_monotonic = float(now)
        return True

    def command(self, now: float) -> tuple[float, float]:
        if (
            self._last_valid_monotonic is None
            or not math.isfinite(now)
            or now < self._last_valid_monotonic
            or now - self._last_valid_monotonic >= self.timeout_sec
        ):
            self.invalidate()
            return STOP_COMMAND
        return self._last_command

    def invalidate(self) -> None:
        self._last_command = STOP_COMMAND
        self._last_valid_monotonic = None
