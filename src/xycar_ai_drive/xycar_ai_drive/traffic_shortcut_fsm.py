"""Pure state machine for Base/traffic-light/shortcut arbitration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from xycar_ai_drive.traffic_light_detector import LampAction


class MissionState(str, Enum):
    OFF = 'OFF'
    BASE = 'BASE'
    RED_STOP = 'RED_STOP'
    SWITCH_TO_SHORTCUT = 'SWITCH_TO_SHORTCUT'
    SHORTCUT = 'SHORTCUT'
    SWITCH_TO_BASE = 'SWITCH_TO_BASE'
    FAULT = 'FAULT'


class PolicyChoice(str, Enum):
    NONE = 'NONE'
    BASE = 'BASE'
    SHORTCUT = 'SHORTCUT'


@dataclass(frozen=True)
class FramePlan:
    state: MissionState
    policy: PolicyChoice
    publish_stop: bool


class TrafficShortcutFsm:
    """Enforce red priority, transition stops and a successful one-shot."""

    def __init__(self, *, shortcut_duration_sec: float = 8.0) -> None:
        if (
            not math.isfinite(shortcut_duration_sec)
            or shortcut_duration_sec <= 0.0
        ):
            raise ValueError('shortcut_duration_sec must be finite and positive')
        self.shortcut_duration_sec = float(shortcut_duration_sec)
        self.state = MissionState.OFF
        self.shortcut_completed = False
        self.shortcut_started_monotonic: float | None = None

    def enable(self) -> None:
        self.state = MissionState.BASE
        self.shortcut_started_monotonic = None

    def disable(self) -> None:
        self.state = MissionState.OFF
        self.shortcut_started_monotonic = None

    def fault(self) -> None:
        self.state = MissionState.FAULT
        self.shortcut_started_monotonic = None

    def on_frame(
        self,
        signal: LampAction,
        *,
        now_monotonic: float,
    ) -> FramePlan:
        self._validate_now(now_monotonic)
        if self.state in {MissionState.OFF, MissionState.FAULT}:
            return self._stop_plan()
        if signal == LampAction.RED:
            self.shortcut_started_monotonic = None
            self.state = MissionState.RED_STOP
            return self._stop_plan()

        if self.state == MissionState.RED_STOP:
            if signal == LampAction.UNKNOWN:
                return self._stop_plan()
            if signal == LampAction.LEFT and not self.shortcut_completed:
                self.state = MissionState.SWITCH_TO_SHORTCUT
                return self._stop_plan()
            self.state = MissionState.BASE
            return self._policy_plan(PolicyChoice.BASE)

        if self.state == MissionState.BASE:
            if signal == LampAction.LEFT and not self.shortcut_completed:
                self.state = MissionState.SWITCH_TO_SHORTCUT
                return self._stop_plan()
            return self._policy_plan(PolicyChoice.BASE)

        if self.state == MissionState.SWITCH_TO_SHORTCUT:
            self.state = MissionState.SHORTCUT
            return self._policy_plan(PolicyChoice.SHORTCUT)

        if self.state == MissionState.SHORTCUT:
            if self._shortcut_deadline_reached(now_monotonic):
                return self._complete_shortcut()
            return self._policy_plan(PolicyChoice.SHORTCUT)

        if self.state == MissionState.SWITCH_TO_BASE:
            self.state = MissionState.BASE
            return self._policy_plan(PolicyChoice.BASE)

        raise RuntimeError(f'unhandled mission state: {self.state}')

    def on_shortcut_command_published(self, *, now_monotonic: float) -> None:
        self._validate_now(now_monotonic)
        if self.state != MissionState.SHORTCUT:
            raise RuntimeError('shortcut command published outside SHORTCUT state')
        if self.shortcut_started_monotonic is None:
            self.shortcut_started_monotonic = now_monotonic

    def on_control_tick(self, *, now_monotonic: float) -> FramePlan | None:
        self._validate_now(now_monotonic)
        if (
            self.state == MissionState.SHORTCUT
            and self._shortcut_deadline_reached(now_monotonic)
        ):
            return self._complete_shortcut()
        return None

    def _shortcut_deadline_reached(self, now_monotonic: float) -> bool:
        return (
            self.shortcut_started_monotonic is not None
            and now_monotonic - self.shortcut_started_monotonic
            >= self.shortcut_duration_sec
        )

    def _complete_shortcut(self) -> FramePlan:
        self.shortcut_completed = True
        self.shortcut_started_monotonic = None
        self.state = MissionState.SWITCH_TO_BASE
        return self._stop_plan()

    def _stop_plan(self) -> FramePlan:
        return FramePlan(
            state=self.state,
            policy=PolicyChoice.NONE,
            publish_stop=True,
        )

    def _policy_plan(self, policy: PolicyChoice) -> FramePlan:
        return FramePlan(
            state=self.state,
            policy=policy,
            publish_stop=False,
        )

    @staticmethod
    def _validate_now(value: float) -> None:
        if not math.isfinite(value):
            raise ValueError('now_monotonic must be finite')
