"""Pure control-state helpers for the front-camera policy node."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

NUM_COMMAND_CLASSES = 201
COMMAND_OFFSET = 100
COMPACT_ANGLE_CLASSES = 81
COMPACT_SPEED_CLASSES = 31
COMPACT_ANGLE_OFFSET = 40
COMPACT_SPEED_TOKEN_OFFSET = 40
NORMALIZED_TO_DRIVER_SCALE = 0.4


@dataclass(frozen=True)
class DriveCommand:
    angle: float = 0.0
    speed: float = 0.0


STOP_COMMAND = DriveCommand()


@dataclass(frozen=True)
class PolicyPrediction:
    command: DriveCommand
    source_monotonic: float
    completed_monotonic: float
    inference_ms: float


class ToggleAction(str, Enum):
    NONE = 'none'
    ENABLED = 'enabled'
    DISABLED = 'disabled'
    REJECTED = 'rejected'


@dataclass
class ToggleDriveGate:
    """Toggle drive on A rising edges while requiring release after faults."""

    enabled: bool = False
    release_seen: bool = False
    last_pressed: bool = False

    def observe(self, *, pressed: bool, can_enable: bool) -> ToggleAction:
        if not pressed:
            self.release_seen = True
        rising = pressed and not self.last_pressed
        self.last_pressed = pressed
        if not rising or not self.release_seen:
            return ToggleAction.NONE

        self.release_seen = False
        if self.enabled:
            self.enabled = False
            return ToggleAction.DISABLED
        if can_enable:
            self.enabled = True
            return ToggleAction.ENABLED
        return ToggleAction.REJECTED

    def fault(self) -> bool:
        was_enabled = self.enabled
        self.enabled = False
        self.release_seen = False
        self.last_pressed = False
        return was_enabled


@dataclass
class HoldDriveGate:
    """Drive only while a button is held, tolerating short false pulses."""

    release_grace_sec: float
    enabled: bool = False
    last_pressed_monotonic: float | None = None
    blocked_until_release: bool = False

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.release_grace_sec)
            or self.release_grace_sec <= 0.0
        ):
            raise ValueError('release_grace_sec must be finite and positive')

    def observe(
        self,
        *,
        pressed: bool,
        can_enable: bool,
        now_monotonic: float,
    ) -> ToggleAction:
        if not math.isfinite(now_monotonic):
            raise ValueError('now_monotonic must be finite')
        if pressed:
            self.last_pressed_monotonic = now_monotonic

        age = (
            None
            if self.last_pressed_monotonic is None
            else now_monotonic - self.last_pressed_monotonic
        )
        held = pressed or (
            age is not None
            and 0.0 <= age <= self.release_grace_sec
        )
        if not held:
            self.last_pressed_monotonic = None
            self.blocked_until_release = False
            if self.enabled:
                self.enabled = False
                return ToggleAction.DISABLED
            return ToggleAction.NONE

        if self.enabled or self.blocked_until_release:
            return ToggleAction.NONE
        if can_enable:
            self.enabled = True
            return ToggleAction.ENABLED
        self.blocked_until_release = True
        return ToggleAction.REJECTED

    def fault(self) -> bool:
        was_enabled = self.enabled
        self.enabled = False
        self.blocked_until_release = True
        return was_enabled


def decode_class_ids(angle_class_id: int, speed_class_id: int) -> DriveCommand:
    for label, class_id in (
        ('angle', angle_class_id),
        ('speed', speed_class_id),
    ):
        if (
            not isinstance(class_id, int)
            or isinstance(class_id, bool)
            or not 0 <= class_id < NUM_COMMAND_CLASSES
        ):
            raise ValueError(
                f'{label} class id must be in [0, 200]: {class_id!r}'
            )
    angle = float(angle_class_id - COMMAND_OFFSET)
    speed = float(max(0, speed_class_id - COMMAND_OFFSET))
    return DriveCommand(angle=angle, speed=speed)


def command_class_ids(command: DriveCommand) -> tuple[int, int]:
    """Quantize one executed command for the AR history contract."""
    values = (command.angle, command.speed)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('executed command must contain finite values')
    return tuple(
        int(round(max(-100.0, min(100.0, value)))) + COMMAND_OFFSET
        for value in values
    )


def decode_compact_output_ids(
    angle_class_id: int,
    speed_class_id: int,
) -> DriveCommand:
    if (
        isinstance(angle_class_id, bool)
        or not isinstance(angle_class_id, int)
        or not 0 <= angle_class_id < COMPACT_ANGLE_CLASSES
    ):
        raise ValueError(
            f'angle class id must be in [0, 80]: {angle_class_id!r}'
        )
    if (
        isinstance(speed_class_id, bool)
        or not isinstance(speed_class_id, int)
        or not 0 <= speed_class_id < COMPACT_SPEED_CLASSES
    ):
        raise ValueError(
            f'speed class id must be in [0, 30]: {speed_class_id!r}'
        )
    driver_angle = angle_class_id - COMPACT_ANGLE_OFFSET
    return DriveCommand(
        angle=float(driver_angle / NORMALIZED_TO_DRIVER_SCALE),
        speed=float(speed_class_id),
    )


def command_history_token_ids(command: DriveCommand) -> tuple[int, int]:
    """Quantize an actually published normalized command for schema v4."""
    if not all(math.isfinite(value) for value in (command.angle, command.speed)):
        raise ValueError('executed command must contain finite values')
    driver_angle = round(
        max(-40.0, min(40.0, command.angle * NORMALIZED_TO_DRIVER_SCALE))
    )
    speed = round(max(0.0, min(30.0, command.speed)))
    return (
        int(driver_angle) + COMPACT_ANGLE_OFFSET,
        int(speed) + COMPACT_SPEED_TOKEN_OFFSET,
    )


def is_fresh(
    now_monotonic: float,
    timestamp_monotonic: float | None,
    timeout_sec: float,
) -> bool:
    if timestamp_monotonic is None:
        return False
    if not all(
        math.isfinite(value)
        for value in (now_monotonic, timestamp_monotonic, timeout_sec)
    ):
        return False
    age = now_monotonic - timestamp_monotonic
    return timeout_sec > 0.0 and 0.0 <= age <= timeout_sec
