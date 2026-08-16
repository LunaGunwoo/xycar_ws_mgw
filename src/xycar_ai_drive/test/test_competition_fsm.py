from xycar_ai_drive.competition_artifact import MissionContract
from xycar_ai_drive.competition_fsm import CompetitionMode, MissionStateMachine
from xycar_ai_drive.competition_gpu_runtime import (
    CompetitionInference,
    SignalObservation,
    ShortcutObservation,
)
from xycar_ai_drive.control import DriveCommand


def _contract():
    return MissionContract(
        probability_threshold=0.5,
        stop_votes_required=2,
        stop_votes_window=3,
        go_votes_required=4,
        go_votes_window=5,
        decision_progress_deadline=0.9,
        handoff_probability_threshold=0.9,
        handoff_consecutive_frames=5,
        handoff_max_angle_difference=25.0,
        shortcut_timeout_sec=12.0,
        maximum_forward_speed=15.0,
    )


def _signal(**overrides):
    values = {
        "approach": 1.0,
        "visible": 1.0,
        "readable": 1.0,
        "red": 0.0,
        "yellow": 0.0,
        "left": 0.0,
        "green": 0.0,
        "bbox": (0.2, 0.1, 0.8, 0.3),
        "progress": 0.5,
    }
    values.update(overrides)
    return SignalObservation(**values)


def _normal_inference(signal):
    return CompetitionInference(
        base_command=DriveCommand(2.0, 15.0),
        base_confidence=0.8,
        signal=signal,
        shortcut=None,
        inference_ms=10.0,
    )


def _shortcut_inference(*, phase=4, handoff=0.0, base=False):
    return CompetitionInference(
        base_command=DriveCommand(4.0, 15.0) if base else None,
        base_confidence=0.8 if base else None,
        signal=None,
        shortcut=ShortcutObservation(
            command=DriveCommand(5.0, 14.0),
            phase=phase,
            handoff_probability=handoff,
        ),
        inference_ms=10.0,
    )


def test_red_stops_and_stable_green_restarts_automatically():
    fsm = MissionStateMachine(_contract())
    fsm.enable(0.0)

    fsm.update(_normal_inference(_signal(red=1.0)), now_monotonic=0.05)
    stopped = fsm.update(
        _normal_inference(_signal(red=1.0)),
        now_monotonic=0.10,
    )
    assert stopped.mode == CompetitionMode.SIGNAL_STOP
    assert stopped.command.speed == 0.0

    for index in range(4):
        resumed = fsm.update(
            _normal_inference(_signal(green=1.0)),
            now_monotonic=0.15 + index * 0.05,
        )
    assert resumed.mode == CompetitionMode.NORMAL
    assert resumed.command.speed == 15.0


def test_left_wins_over_green_and_handoff_requires_persistence():
    fsm = MissionStateMachine(_contract())
    fsm.enable(0.0)
    for index in range(4):
        decision = fsm.update(
            _normal_inference(_signal(left=1.0, green=1.0)),
            now_monotonic=0.05 + index * 0.05,
        )
    assert decision.mode == CompetitionMode.SHORTCUT
    assert decision.reset_shortcut
    assert fsm.shortcut_used

    candidate = fsm.update(
        _shortcut_inference(phase=5, handoff=0.95),
        now_monotonic=0.30,
    )
    assert candidate.mode == CompetitionMode.HANDOFF_VERIFY
    dropped = fsm.update(
        _shortcut_inference(phase=5, handoff=0.2, base=True),
        now_monotonic=0.35,
    )
    assert dropped.mode == CompetitionMode.SHORTCUT

    fsm.update(
        _shortcut_inference(phase=5, handoff=0.95),
        now_monotonic=0.40,
    )
    for index in range(5):
        verified = fsm.update(
            _shortcut_inference(phase=5, handoff=0.95, base=True),
            now_monotonic=0.45 + index * 0.05,
        )
    assert verified.mode == CompetitionMode.NORMAL
    assert verified.completed


def test_unknown_signal_at_deadline_fails_closed():
    fsm = MissionStateMachine(_contract())
    fsm.enable(0.0)

    decision = fsm.update(
        _normal_inference(
            _signal(readable=0.0, visible=0.2, progress=0.95)
        ),
        now_monotonic=0.05,
    )

    assert decision.mode == CompetitionMode.SIGNAL_STOP
    assert decision.command.speed == 0.0
    assert "deadline" in decision.reason


def test_shortcut_timeout_stops_instead_of_handing_off():
    fsm = MissionStateMachine(_contract(), shortcut_only=True)
    fsm.enable(0.0)

    decision = fsm.update(
        _shortcut_inference(),
        now_monotonic=12.1,
    )

    assert decision.mode == CompetitionMode.FAULT
    assert decision.command.speed == 0.0
