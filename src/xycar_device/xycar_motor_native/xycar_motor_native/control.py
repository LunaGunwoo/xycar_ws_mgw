"""Pure control and calibration logic for the native VESC gateway."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MotorCommand:
    angle: float
    speed: float


@dataclass(frozen=True)
class VescSetpoint:
    command: MotorCommand
    erpm: float
    servo: float
    ramping: bool
    clamped: bool


@dataclass(frozen=True)
class NativeMotorContract:
    angle_min: float = -50.0
    angle_max: float = 100.0
    speed_min: float = -50.0
    speed_max: float = 100.0
    angle_to_radians_gain: float = 0.0068
    speed_to_mps_gain: float = 0.08
    speed_to_erpm_gain: float = 4614.0
    speed_to_erpm_offset: float = 0.0
    steering_to_servo_gain: float = -1.2135
    steering_to_servo_offset: float = 0.5004
    erpm_min: float = -20000.0
    erpm_max: float = 40000.0
    servo_min: float = 0.15
    servo_max: float = 0.85
    startup_ramp_duration_sec: float = 0.70
    nominal_frame_rate_hz: float = 30.0

    def validate(self) -> None:
        values = tuple(self.__dict__.values())
        if not all(math.isfinite(value) for value in values):
            raise ValueError('native motor contract values must be finite')
        if self.angle_min >= self.angle_max:
            raise ValueError('angle limits are invalid')
        if self.speed_min >= self.speed_max:
            raise ValueError('speed limits are invalid')
        if self.erpm_min >= self.erpm_max:
            raise ValueError('ERPM limits are invalid')
        if self.servo_min >= self.servo_max:
            raise ValueError('servo limits are invalid')
        if self.startup_ramp_duration_sec <= 0.0:
            raise ValueError('startup ramp duration must be positive')
        if self.nominal_frame_rate_hz <= 0.0:
            raise ValueError('nominal frame rate must be positive')


class NativeMotorMapper:
    """Clamp Xycar units, apply non-blocking ramp, then map to VESC units."""

    def __init__(self, contract: NativeMotorContract) -> None:
        contract.validate()
        self.contract = contract
        self._speed = 0.0
        self._ramp_started: float | None = None
        self._ramp_target = 0.0

    @property
    def speed(self) -> float:
        return self._speed

    def reset(self) -> None:
        self._speed = 0.0
        self._ramp_started = None
        self._ramp_target = 0.0

    def map(self, command: MotorCommand, *, now: float) -> VescSetpoint:
        if not all(
            math.isfinite(value) for value in (command.angle, command.speed, now)
        ):
            raise ValueError('motor command and timestamp must be finite')
        angle = _clamp(
            command.angle,
            self.contract.angle_min,
            self.contract.angle_max,
        )
        target_speed = _clamp(
            command.speed,
            self.contract.speed_min,
            self.contract.speed_max,
        )
        clamped = angle != command.angle or target_speed != command.speed
        speed, ramping = self._safe_speed(target_speed, now=now)
        erpm = (
            self.contract.speed_to_erpm_gain
            * self.contract.speed_to_mps_gain
            * speed
            + self.contract.speed_to_erpm_offset
        )
        steering_radians = self.contract.angle_to_radians_gain * angle
        servo = (
            self.contract.steering_to_servo_gain * steering_radians
            + self.contract.steering_to_servo_offset
        )
        limited_erpm = _clamp(
            erpm,
            self.contract.erpm_min,
            self.contract.erpm_max,
        )
        limited_servo = _clamp(
            servo,
            self.contract.servo_min,
            self.contract.servo_max,
        )
        clamped = clamped or limited_erpm != erpm or limited_servo != servo
        return VescSetpoint(
            command=MotorCommand(angle=angle, speed=speed),
            erpm=limited_erpm,
            servo=limited_servo,
            ramping=ramping,
            clamped=clamped,
        )

    def stop(self) -> VescSetpoint:
        self.reset()
        servo = _clamp(
            self.contract.steering_to_servo_offset,
            self.contract.servo_min,
            self.contract.servo_max,
        )
        return VescSetpoint(
            command=MotorCommand(angle=0.0, speed=0.0),
            erpm=0.0,
            servo=servo,
            ramping=False,
            clamped=False,
        )

    def _safe_speed(self, target: float, *, now: float) -> tuple[float, bool]:
        if target == 0.0:
            self.reset()
            return 0.0, False
        if self._speed != 0.0 and math.copysign(1.0, target) != math.copysign(
            1.0,
            self._speed,
        ):
            self._speed = 0.0
            self._ramp_started = None
        target_sign_changed = (
            self._ramp_target != 0.0
            and math.copysign(1.0, target)
            != math.copysign(1.0, self._ramp_target)
        )
        if target_sign_changed:
            self._speed = 0.0
            self._ramp_started = None
        if self._ramp_started is None and self._speed == 0.0:
            self._ramp_started = now
            self._ramp_target = target
        if self._ramp_started is None:
            self._speed = target
            self._ramp_target = target
            return self._speed, False
        self._ramp_target = target
        elapsed = max(0.0, now - self._ramp_started)
        elapsed += 1.0 / self.contract.nominal_frame_rate_hz
        progress = min(1.0, elapsed / self.contract.startup_ramp_duration_sec)
        self._speed = self._ramp_target * progress
        if progress >= 1.0:
            self._ramp_started = None
            return self._speed, False
        return self._speed, True


class CommandFreshnessWatchdog:
    """Track source-command freshness without resampling normal commands."""

    def __init__(self, *, timeout_sec: float, check_rate_hz: float) -> None:
        if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
            raise ValueError('watchdog timeout must be finite and positive')
        if not math.isfinite(check_rate_hz) or check_rate_hz <= 0.0:
            raise ValueError('watchdog rate must be finite and positive')
        self.timeout_sec = timeout_sec
        self.check_period_sec = 1.0 / check_rate_hz
        self._last_command_time: float | None = None

    @property
    def last_command_time(self) -> float | None:
        return self._last_command_time

    def observe(self, now: float) -> None:
        if not math.isfinite(now):
            raise ValueError('command observation time must be finite')
        self._last_command_time = now

    def stale_age(self, now: float) -> float | None:
        if not math.isfinite(now):
            raise ValueError('watchdog check time must be finite')
        if self._last_command_time is None:
            return None
        age = max(0.0, now - self._last_command_time)
        return age if age > self.timeout_sec else None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))
