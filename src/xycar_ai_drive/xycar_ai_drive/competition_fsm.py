"""Pure fail-closed mission state machine for competition driving."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

from xycar_ai_drive.competition_artifact import MissionContract
from xycar_ai_drive.competition_gpu_runtime import CompetitionInference
from xycar_ai_drive.control import DriveCommand, STOP_COMMAND


class CompetitionMode(str, Enum):
    DISABLED = "disabled"
    NORMAL = "normal"
    SIGNAL_STOP = "signal_stop"
    SHORTCUT = "shortcut"
    HANDOFF_VERIFY = "handoff_verify"
    FAULT = "fault"


@dataclass(frozen=True)
class MissionDecision:
    command: DriveCommand
    mode: CompetitionMode
    reason: str
    reset_shortcut: bool = False
    completed: bool = False


class MissionStateMachine:
    """Orchestrate deterministic priorities around learned observations."""

    def __init__(
        self,
        contract: MissionContract,
        *,
        shortcut_only: bool = False,
    ) -> None:
        self.contract = contract
        self.shortcut_only = shortcut_only
        self.mode = CompetitionMode.DISABLED
        self.shortcut_used = False
        self._shortcut_started: float | None = None
        self._stop_votes: deque[bool] = deque(
            maxlen=contract.stop_votes_window
        )
        self._left_votes: deque[bool] = deque(maxlen=contract.go_votes_window)
        self._green_votes: deque[bool] = deque(maxlen=contract.go_votes_window)
        self._approach_exit_votes: deque[bool] = deque(maxlen=5)
        self._handoff_stable = 0
        self._straight_committed = False

    @property
    def inference_mode(self) -> str:
        if self.mode == CompetitionMode.NORMAL:
            return "normal"
        if self.mode == CompetitionMode.SIGNAL_STOP:
            return "signal_stop"
        if self.mode == CompetitionMode.SHORTCUT:
            return "shortcut"
        if self.mode == CompetitionMode.HANDOFF_VERIFY:
            return "handoff_verify"
        raise RuntimeError(f"mode {self.mode.value} cannot request inference")

    def enable(self, now_monotonic: float) -> MissionDecision:
        self._validate_time(now_monotonic)
        self._clear_temporal_votes()
        self._straight_committed = False
        self._handoff_stable = 0
        if self.shortcut_only:
            self.mode = CompetitionMode.SHORTCUT
            self._shortcut_started = now_monotonic
            return MissionDecision(
                STOP_COMMAND,
                self.mode,
                "shortcut-only enabled; awaiting fresh shortcut inference",
                reset_shortcut=True,
            )
        self.mode = CompetitionMode.NORMAL
        self._shortcut_started = None
        return MissionDecision(
            STOP_COMMAND,
            self.mode,
            "combined policy enabled; awaiting fresh normal inference",
        )

    def disable(self, reason: str = "operator disabled") -> MissionDecision:
        self.mode = CompetitionMode.DISABLED
        self._shortcut_started = None
        self._clear_temporal_votes()
        self._handoff_stable = 0
        return MissionDecision(STOP_COMMAND, self.mode, reason)

    def fault(self, reason: str) -> MissionDecision:
        self.mode = CompetitionMode.FAULT
        self._shortcut_started = None
        self._clear_temporal_votes()
        self._handoff_stable = 0
        return MissionDecision(STOP_COMMAND, self.mode, reason)

    def update(
        self,
        inference: CompetitionInference,
        *,
        now_monotonic: float,
    ) -> MissionDecision:
        self._validate_time(now_monotonic)
        if self.mode in {CompetitionMode.DISABLED, CompetitionMode.FAULT}:
            return MissionDecision(
                STOP_COMMAND,
                self.mode,
                "motion gate is not enabled",
            )
        if self.mode in {CompetitionMode.NORMAL, CompetitionMode.SIGNAL_STOP}:
            return self._update_signal_mode(inference, now_monotonic)
        if self.mode == CompetitionMode.SHORTCUT:
            return self._update_shortcut(inference, now_monotonic)
        return self._update_handoff(inference, now_monotonic)

    def _update_signal_mode(
        self,
        inference: CompetitionInference,
        now_monotonic: float,
    ) -> MissionDecision:
        if inference.base_command is None or inference.signal is None:
            return self.fault("normal inference omitted base or signal output")
        signal = inference.signal
        threshold = self.contract.probability_threshold
        readable = signal.readable >= threshold
        stop = readable and (signal.red >= threshold or signal.yellow >= threshold)
        left = readable and signal.left >= threshold and not self.shortcut_used
        green = readable and signal.green >= threshold
        self._stop_votes.append(stop)
        self._left_votes.append(left)
        self._green_votes.append(green)
        approach = signal.approach >= threshold

        if self._straight_committed:
            self._approach_exit_votes.append(not approach)
            if _votes(self._approach_exit_votes, 4):
                self._straight_committed = False
                self._clear_temporal_votes()
            self.mode = CompetitionMode.NORMAL
            return MissionDecision(
                self._safe_command(inference.base_command),
                self.mode,
                "straight signal already committed for this encounter",
            )

        if _votes(self._stop_votes, self.contract.stop_votes_required):
            self.mode = CompetitionMode.SIGNAL_STOP
            return MissionDecision(
                STOP_COMMAND,
                self.mode,
                "red or yellow signal confirmed",
            )
        if _votes(self._left_votes, self.contract.go_votes_required):
            was_stopped = self.mode == CompetitionMode.SIGNAL_STOP
            self.mode = CompetitionMode.SHORTCUT
            self.shortcut_used = True
            self._shortcut_started = now_monotonic
            self._handoff_stable = 0
            return MissionDecision(
                (
                    STOP_COMMAND
                    if was_stopped
                    else self._safe_command(inference.base_command)
                ),
                self.mode,
                "left arrow latched; shortcut policy takes control next frame",
                reset_shortcut=True,
            )
        if _votes(self._green_votes, self.contract.go_votes_required):
            self.mode = CompetitionMode.NORMAL
            self._straight_committed = True
            return MissionDecision(
                self._safe_command(inference.base_command),
                self.mode,
                "straight green confirmed",
            )
        if (
            approach
            and signal.progress >= self.contract.decision_progress_deadline
        ):
            self.mode = CompetitionMode.SIGNAL_STOP
            return MissionDecision(
                STOP_COMMAND,
                self.mode,
                "signal remained unknown at the decision deadline",
            )
        if self.mode == CompetitionMode.SIGNAL_STOP:
            return MissionDecision(
                STOP_COMMAND,
                self.mode,
                "waiting for stable green or left signal",
            )
        return MissionDecision(
            self._safe_command(inference.base_command),
            self.mode,
            "normal lap policy",
        )

    def _update_shortcut(
        self,
        inference: CompetitionInference,
        now_monotonic: float,
    ) -> MissionDecision:
        if inference.shortcut is None:
            return self.fault("shortcut inference omitted shortcut output")
        if self._shortcut_started is None:
            return self.fault("shortcut state has no start timestamp")
        if (
            now_monotonic - self._shortcut_started
            > self.contract.shortcut_timeout_sec
        ):
            return self.fault("shortcut policy exceeded its time limit")
        observation = inference.shortcut
        if (
            observation.phase == 5
            and observation.handoff_probability
            >= self.contract.handoff_probability_threshold
        ):
            self.mode = CompetitionMode.HANDOFF_VERIFY
            self._handoff_stable = 0
            return MissionDecision(
                self._safe_command(observation.command),
                self.mode,
                "reacquire candidate; starting base shadow verification",
            )
        return MissionDecision(
            self._safe_command(observation.command),
            self.mode,
            "shortcut policy active",
        )

    def _update_handoff(
        self,
        inference: CompetitionInference,
        now_monotonic: float,
    ) -> MissionDecision:
        if inference.shortcut is None or inference.base_command is None:
            return self.fault("handoff inference omitted base or shortcut output")
        if self._shortcut_started is None:
            return self.fault("handoff state has no shortcut start timestamp")
        if (
            now_monotonic - self._shortcut_started
            > self.contract.shortcut_timeout_sec
        ):
            return self.fault("shortcut handoff exceeded its time limit")
        shortcut = inference.shortcut
        ready = (
            shortcut.phase == 5
            and shortcut.handoff_probability
            >= self.contract.handoff_probability_threshold
        )
        agrees = (
            abs(shortcut.command.angle - inference.base_command.angle)
            <= self.contract.handoff_max_angle_difference
        )
        if ready and agrees:
            self._handoff_stable += 1
        else:
            self._handoff_stable = 0
        if self._handoff_stable >= self.contract.handoff_consecutive_frames:
            if self.shortcut_only:
                self.mode = CompetitionMode.DISABLED
                return MissionDecision(
                    STOP_COMMAND,
                    self.mode,
                    "shortcut-only handoff complete",
                    completed=True,
                )
            self.mode = CompetitionMode.NORMAL
            self._shortcut_started = None
            self._straight_committed = True
            self._clear_temporal_votes()
            return MissionDecision(
                self._safe_command(inference.base_command),
                self.mode,
                "shortcut handoff verified; base policy resumed",
                completed=True,
            )
        if not ready:
            self.mode = CompetitionMode.SHORTCUT
            return MissionDecision(
                self._safe_command(shortcut.command),
                self.mode,
                "handoff readiness dropped; shortcut remains active",
            )
        return MissionDecision(
            self._safe_command(shortcut.command),
            self.mode,
            "verifying persistent handoff and base agreement",
        )

    def _safe_command(self, command: DriveCommand) -> DriveCommand:
        if not all(math.isfinite(value) for value in (command.angle, command.speed)):
            raise ValueError("model command must be finite")
        return DriveCommand(
            angle=max(-100.0, min(100.0, command.angle)),
            speed=max(
                0.0,
                min(self.contract.maximum_forward_speed, command.speed),
            ),
        )

    def _clear_temporal_votes(self) -> None:
        self._stop_votes.clear()
        self._left_votes.clear()
        self._green_votes.clear()
        self._approach_exit_votes.clear()

    @staticmethod
    def _validate_time(value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("monotonic time must be finite")


def _votes(values: deque[bool], required: int) -> bool:
    return len(values) >= required and sum(values) >= required
