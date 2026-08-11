"""Pure control-state helpers for the front-camera policy node."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

NUM_COMMAND_CLASSES = 201
COMMAND_OFFSET = 100


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
