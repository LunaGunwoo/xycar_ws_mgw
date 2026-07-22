# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import math

import pytest

from xycar_data.gamepad_teleop import (
    DriveCommand,
    GamepadConfig,
    InvalidJoyInput,
    is_input_fresh,
    map_joy_axes,
)


def test_neutral_input_stops_with_centered_steering():
    assert map_joy_axes([0.0] * 6) == DriveCommand(0.0, 0.0)


@pytest.mark.parametrize(
    ('steering', 'expected_angle'),
    [(-1.0, 100.0), (-0.5, 50.0), (0.5, -50.0), (1.0, -100.0)],
)
def test_steering_maps_to_full_angle_range(steering, expected_angle):
    axes = [steering, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert map_joy_axes(axes).angle == expected_angle


def test_lt_maps_to_reverse_speed():
    command = map_joy_axes([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert command == DriveCommand(0.0, -5.0)


def test_rt_maps_to_forward_speed():
    command = map_joy_axes([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert command == DriveCommand(0.0, 7.0)


def test_triggers_are_combined_for_partial_and_simultaneous_input():
    partial = map_joy_axes([0.0, 0.0, 0.0, 0.0, 0.4, 0.5])
    both_full = map_joy_axes([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    assert partial.speed == pytest.approx(1.5)
    assert both_full.speed == pytest.approx(2.0)


def test_steering_is_preserved_while_speed_is_zero():
    command = map_joy_axes([0.75, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert command == DriveCommand(-75.0, 0.0)


def test_axes_are_clamped_before_mapping():
    command = map_joy_axes([2.0, 0.0, 0.0, 0.0, 2.0, -1.0])
    assert command == DriveCommand(-100.0, -5.0)


@pytest.mark.parametrize(
    'axes',
    [
        [0.0] * 5,
        [math.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, math.inf, 0.0],
    ],
)
def test_invalid_axes_are_rejected(axes):
    with pytest.raises(InvalidJoyInput):
        map_joy_axes(axes)


def test_custom_mapping_and_limits_are_supported():
    config = GamepadConfig(
        steering_axis=2,
        lt_axis=0,
        rt_axis=1,
        invert_steering=False,
        max_angle=30.0,
        max_reverse_speed=3.0,
        max_forward_speed=4.0,
    )
    command = map_joy_axes([0.25, 0.5, -0.5], config)
    assert command == DriveCommand(-15.0, 1.25)


def test_input_freshness_timeout():
    assert is_input_fresh(10.2, 10.0, 0.25)
    assert is_input_fresh(10.25, 10.0, 0.25)
    assert not is_input_fresh(10.251, 10.0, 0.25)
    assert not is_input_fresh(10.0, None, 0.25)
    assert not is_input_fresh(9.9, 10.0, 0.25)
