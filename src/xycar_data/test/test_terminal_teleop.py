# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import pytest

from xycar_data.teleop_recorder import (
    DriveCommand,
    KeyboardCommandState,
    TeleopRecorderNode,
    _is_recordable_command,
)
from xycar_data.terminal_input import KeySequenceParser
from xycar_data.tuning import ControlConfig, TeleopTuning, validate_tuning


def test_wasd_speed_and_steering_adjust_independent_components():
    state = KeyboardCommandState(ControlConfig())

    assert state.set_drive_key("w", 0.0) == DriveCommand(0.0, 8.0, "w")
    assert state.set_drive_key("a", 0.01) == DriveCommand(-10.0, 8.0, "a")
    assert state.set_drive_key("s", 0.02) == DriveCommand(-10.0, -8.0, "s")
    assert state.set_drive_key("d", 0.03) == DriveCommand(0.0, -8.0, "d")


def test_w_and_shift_w_set_immediate_nonincremental_speed():
    state = KeyboardCommandState(ControlConfig())

    state.set_drive_key("w", 0.0)
    state.set_drive_key("w", 0.01)
    assert state.command.speed == 8.0

    state.set_drive_key("W", 0.02)
    assert state.command.speed == 12.0

    state.set_drive_key("w", 0.03)
    assert state.command.speed == 8.0


def test_s_has_no_shift_boost():
    state = KeyboardCommandState(ControlConfig())

    assert state.set_drive_key("s", 0.0).speed == -8.0
    assert state.set_drive_key("s", 0.01).speed == -8.0


def test_ten_a_or_d_repeats_reach_clamped_full_lock():
    state = KeyboardCommandState(ControlConfig())
    for index in range(10):
        state.set_drive_key("a", index * (0.3 / 9.0))
    assert state.command.angle == -100.0

    for index in range(20):
        state.set_drive_key("d", 1.0 + index * (0.3 / 19.0))
    assert state.command.angle == 100.0


def test_stale_keyboard_input_clears_both_components():
    state = KeyboardCommandState(ControlConfig())
    state.set_drive_key("w", 1.0)
    state.set_drive_key("a", 1.01)

    assert not state.expire_if_stale(1.25)
    assert state.expire_if_stale(1.27)
    assert state.command == DriveCommand()
    assert state.set_drive_key("w", 2.0) == DriveCommand(0.0, 8.0, "w")


def test_zero_speed_steering_only_command_is_not_recordable():
    assert not _is_recordable_command(DriveCommand(-100.0, 0.0, "a"))
    assert _is_recordable_command(DriveCommand(-100.0, 8.0, "w"))
    assert _is_recordable_command(DriveCommand(100.0, -8.0, "s"))


def test_terminal_parser_preserves_shift_w_and_maps_wasd_e_q():
    parser = KeySequenceParser()

    assert parser.feed(b"wWsSaAdDeEqQzZ") == [
        "w",
        "W",
        "s",
        "s",
        "a",
        "a",
        "d",
        "d",
        "e",
        "e",
        "q",
        "q",
    ]


def test_arrow_sequences_are_ignored():
    parser = KeySequenceParser()

    assert parser.feed(b"\x1b[A\x1b[B\x1b[C\x1b[D") == []


def test_e_starts_and_q_finishes_recording_without_stopping_motion():
    calls = []

    class RecorderStub:
        def _start_session(self):
            calls.append(("start",))

        def _close_active_session(self, reason, *, complete):
            calls.append(("finish", reason, complete))

        def stop_motion(self, reason, *, burst):
            calls.append(("stop", reason, burst))

    node = RecorderStub()
    TeleopRecorderNode.handle_key(node, "e", 1.0)
    TeleopRecorderNode.handle_key(node, "q", 2.0)

    assert calls == [
        ("start",),
        ("finish", "operator ended session with Q", True),
    ]


def test_space_stops_and_only_escape_or_ctrl_c_exit():
    calls = []

    class RecorderStub:
        def stop_motion(self, reason, *, burst):
            calls.append(("stop", reason, burst))

        def request_exit(self, reason, *, complete):
            calls.append(("exit", reason, complete))

    node = RecorderStub()
    TeleopRecorderNode.handle_key(node, "space", 1.0)
    TeleopRecorderNode.handle_key(node, "escape", 2.0)
    TeleopRecorderNode.handle_key(node, "ctrl_c", 3.0)

    assert calls == [
        ("stop", "stopped by Space", True),
        ("exit", "operator requested exit", True),
        ("exit", "operator requested exit", True),
    ]


@pytest.mark.parametrize(
    "control",
    [
        ControlConfig(min_angle=0.0),
        ControlConfig(max_angle=0.0),
        ControlConfig(angle_step=0.0),
        ControlConfig(forward_speed=0.0),
        ControlConfig(forward_boost_multiplier=0.5),
        ControlConfig(reverse_speed=0.0),
    ],
)
def test_invalid_keyboard_control_range_is_rejected(control):
    with pytest.raises(ValueError):
        validate_tuning(TeleopTuning(control=control))
