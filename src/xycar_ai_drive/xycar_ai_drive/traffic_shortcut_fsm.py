"""Pure state machine for Base/traffic-light/shortcut arbitration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from xycar_ai_drive.traffic_light_detector import LampAction


class MissionState(str, Enum):
    OFF = 'OFF'
    WAIT_FOR_SIGNAL = 'WAIT_FOR_SIGNAL'
    BASE = 'BASE'
    RED_STOP = 'RED_STOP'
    INITIAL_STOP = 'INITIAL_STOP'
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
    promote_base_shadow: bool = False


class TrafficShortcutFsm:
    """Enforce red priority, transition stops and a successful one-shot."""

    def __init__(
        self,
        *,
        shortcut_duration_sec: float = 8.0,
        seamless_base_handoff: bool = False,
        one_shot_initial_stop: bool = False,
        initial_left_direct_shortcut: bool = False,
        rearm_shortcut_on_enable: bool = False,
        repeat_initial_stop_until_shortcut: bool = False,
    ) -> None:
        if (
            not math.isfinite(shortcut_duration_sec)
            or shortcut_duration_sec <= 0.0
        ):
            raise ValueError(
                'shortcut_duration_sec must be finite and positive'
            )
        self.shortcut_duration_sec = float(shortcut_duration_sec)
        self.seamless_base_handoff = bool(seamless_base_handoff)
        self.one_shot_initial_stop = bool(one_shot_initial_stop)
        self.initial_left_direct_shortcut = bool(initial_left_direct_shortcut)
        self.rearm_shortcut_on_enable = bool(rearm_shortcut_on_enable)
        self.repeat_initial_stop_until_shortcut = bool(
            repeat_initial_stop_until_shortcut
        )
        if (
            self.initial_left_direct_shortcut
            and not self.one_shot_initial_stop
        ):
            raise ValueError(
                'initial LEFT shortcut requires one-shot mission mode'
            )
        if (
            self.repeat_initial_stop_until_shortcut
            and not self.one_shot_initial_stop
        ):
            raise ValueError(
                'repeated initial STOP requires one-shot mission mode'
            )
        self.state = MissionState.OFF
        self.shortcut_completed = False
        self.shortcut_started_monotonic: float | None = None
        self.initial_stop_armed = False
        self.initial_stop_consumed = False

    def enable(
        self,
        *,
        initial_stop_armed: bool = False,
        wait_for_signal: bool = False,
    ) -> None:
        if not self.one_shot_initial_stop and (
            initial_stop_armed or wait_for_signal
        ):
            raise ValueError(
                'initial STOP options require one-shot mission mode'
            )
        if wait_for_signal and not initial_stop_armed:
            raise ValueError('signal wait requires initial STOP arm')
        self.initial_stop_armed = bool(initial_stop_armed)
        self.initial_stop_consumed = not initial_stop_armed
        if self.rearm_shortcut_on_enable:
            self.shortcut_completed = False
        self.state = (
            MissionState.WAIT_FOR_SIGNAL
            if wait_for_signal
            else MissionState.BASE
        )
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
        if self.one_shot_initial_stop:
            return self._on_one_shot_frame(signal)
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
            return self._policy_plan(PolicyChoice.SHORTCUT)

        if self.state == MissionState.SWITCH_TO_BASE:
            self.state = MissionState.BASE
            return self._policy_plan(PolicyChoice.BASE)

        raise RuntimeError(f'unhandled mission state: {self.state}')

    def _on_one_shot_frame(self, signal: LampAction) -> FramePlan:
        if self.state in {
            MissionState.WAIT_FOR_SIGNAL,
            MissionState.INITIAL_STOP,
        }:
            if signal not in {LampAction.LEFT, LampAction.STRAIGHT}:
                return self._stop_plan()
            self.initial_stop_armed = False
            self.initial_stop_consumed = True
            if (
                signal == LampAction.LEFT
                and self.initial_left_direct_shortcut
                and not self.shortcut_completed
            ):
                self.state = MissionState.SWITCH_TO_SHORTCUT
                return self._stop_plan()
            self.state = MissionState.BASE
            return self._policy_plan(PolicyChoice.BASE)

        if self.state == MissionState.BASE:
            if (
                signal == LampAction.RED
                and self.initial_stop_armed
                and not self.initial_stop_consumed
            ):
                self.state = MissionState.INITIAL_STOP
                return self._stop_plan()
            if (
                signal == LampAction.LEFT
                and not self.initial_stop_armed
                and not self.shortcut_completed
            ):
                self.state = MissionState.SWITCH_TO_SHORTCUT
                return self._stop_plan()
            return self._policy_plan(PolicyChoice.BASE)

        if self.state == MissionState.SWITCH_TO_SHORTCUT:
            self.state = MissionState.SHORTCUT
            return self._policy_plan(PolicyChoice.SHORTCUT)

        if self.state == MissionState.SHORTCUT:
            return self._policy_plan(PolicyChoice.SHORTCUT)

        if self.state == MissionState.SWITCH_TO_BASE:
            self.state = MissionState.BASE
            return self._policy_plan(PolicyChoice.BASE)

        raise RuntimeError(f'unhandled one-shot mission state: {self.state}')

    def on_shortcut_command_published(self, *, now_monotonic: float) -> None:
        self._validate_now(now_monotonic)
        if self.state != MissionState.SHORTCUT:
            raise RuntimeError(
                'shortcut command published outside SHORTCUT state'
            )
        if self.shortcut_started_monotonic is None:
            self.shortcut_started_monotonic = now_monotonic

    def rearm_initial_stop(self) -> bool:
        if not self.repeat_initial_stop_until_shortcut:
            raise RuntimeError('repeated initial STOP is not enabled')
        if self.state != MissionState.BASE or self.shortcut_completed:
            return False
        self.initial_stop_armed = True
        self.initial_stop_consumed = False
        return True

    def on_base_shadow_promoted(self) -> None:
        if not self.seamless_base_handoff:
            raise RuntimeError('Base shadow promotion is not enabled')
        if self.state != MissionState.SWITCH_TO_BASE:
            raise RuntimeError(
                'Base shadow promoted outside SWITCH_TO_BASE state'
            )
        self.shortcut_completed = True
        self.state = MissionState.BASE

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
        self.shortcut_started_monotonic = None
        if self.seamless_base_handoff:
            self.state = MissionState.SWITCH_TO_BASE
            return FramePlan(
                state=self.state,
                policy=PolicyChoice.BASE,
                publish_stop=False,
                promote_base_shadow=True,
            )
        self.shortcut_completed = True
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
