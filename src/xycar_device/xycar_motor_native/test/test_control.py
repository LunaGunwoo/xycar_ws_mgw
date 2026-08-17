import math

import pytest

from xycar_motor_native.control import (
    CommandFreshnessWatchdog,
    MotorCommand,
    NativeMotorContract,
    NativeMotorMapper,
    VescFeedback,
    VescFeedbackContract,
    vesc_feedback_error,
)


def test_calibration_matches_vehicle_contract_after_ramp():
    mapper = NativeMotorMapper(NativeMotorContract())
    result = None
    for index in range(22):
        result = mapper.map(
            MotorCommand(angle=10.0, speed=15.0),
            now=index / 30.0,
        )
    assert result is not None
    assert result.ramping is False
    assert result.command == MotorCommand(angle=10.0, speed=15.0)
    assert result.erpm == pytest.approx(4614.0 * 0.08 * 15.0)
    assert result.servo == pytest.approx(-1.2135 * 0.0068 * 10.0 + 0.5004)


def test_30hz_startup_ramp_is_non_blocking_and_monotonic():
    mapper = NativeMotorMapper(NativeMotorContract())
    speeds = [
        mapper.map(MotorCommand(angle=0.0, speed=21.0), now=index / 30.0)
        .command.speed
        for index in range(22)
    ]
    assert speeds[0] > 0.0
    assert speeds[-1] == pytest.approx(21.0)
    assert speeds == sorted(speeds)


def test_stop_is_immediate_and_reverse_restarts_ramp_from_zero():
    mapper = NativeMotorMapper(NativeMotorContract())
    mapper.map(MotorCommand(angle=0.0, speed=15.0), now=0.0)
    forward = mapper.map(MotorCommand(angle=0.0, speed=15.0), now=0.2)
    stopped = mapper.map(MotorCommand(angle=5.0, speed=0.0), now=0.21)
    reverse = mapper.map(MotorCommand(angle=5.0, speed=-7.0), now=0.22)
    assert forward.command.speed > 0.0
    assert stopped.command.speed == 0.0
    assert reverse.command.speed < 0.0
    assert abs(reverse.command.speed) < 7.0
    assert reverse.ramping is True


def test_direction_change_never_emits_old_direction():
    mapper = NativeMotorMapper(NativeMotorContract())
    for index in range(22):
        mapper.map(MotorCommand(angle=0.0, speed=15.0), now=index / 30.0)
    reverse = mapper.map(MotorCommand(angle=0.0, speed=-7.0), now=0.8)
    assert reverse.command.speed < 0.0
    assert abs(reverse.command.speed) < 7.0


def test_limits_are_applied_before_mapping():
    mapper = NativeMotorMapper(NativeMotorContract(startup_ramp_duration_sec=0.01))
    result = mapper.map(MotorCommand(angle=-100.0, speed=500.0), now=1.0)
    result = mapper.map(MotorCommand(angle=-100.0, speed=500.0), now=1.1)
    assert result.command.angle == -50.0
    assert result.command.speed == 100.0
    assert result.erpm == 36912.0
    assert result.clamped is True


def test_non_finite_command_is_rejected():
    mapper = NativeMotorMapper(NativeMotorContract())
    with pytest.raises(ValueError):
        mapper.map(MotorCommand(angle=math.nan, speed=0.0), now=0.0)


def test_synthetic_30hz_has_one_immediate_mapping_per_frame_for_60_seconds():
    mapper = NativeMotorMapper(NativeMotorContract())
    outputs = [
        mapper.map(
            MotorCommand(angle=float(index % 21 - 10), speed=20.0),
            now=index / 30.0,
        )
        for index in range(60 * 30)
    ]
    assert len(outputs) == 1800
    assert all(math.isfinite(output.erpm) for output in outputs)
    assert outputs[-1].ramping is False


def test_watchdog_stops_within_timeout_plus_one_30hz_check_period():
    watchdog = CommandFreshnessWatchdog(timeout_sec=0.25, check_rate_hz=30.0)
    watchdog.observe(10.0)
    assert watchdog.stale_age(10.25) is None
    first_check_after_deadline = 10.0 + 8.0 / 30.0
    age = watchdog.stale_age(first_check_after_deadline)
    assert age is not None
    assert age <= watchdog.timeout_sec + watchdog.check_period_sec


def test_valid_vesc_fw218_feedback_is_accepted():
    feedback = VescFeedback(
        voltage_input=9.6,
        temperature_pcb=31.5,
        current_motor=-12.34,
        current_input=4.56,
        speed=12345.0,
        duty_cycle=-0.125,
        fault_code=0,
    )
    assert vesc_feedback_error(feedback, VescFeedbackContract()) is None


@pytest.mark.parametrize(
    ('feedback', 'expected'),
    (
        (
            VescFeedback(0.0, 0.0, 173017.68, 173017.68, -131072.0, 0.0, 43),
            'fault code',
        ),
        (
            VescFeedback(0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0),
            'voltage',
        ),
        (
            VescFeedback(9.6, 20.0, math.inf, 0.0, 0.0, 0.0, 0),
            'NaN or Inf',
        ),
    ),
)
def test_malformed_or_faulted_vesc_feedback_is_rejected(feedback, expected):
    reason = vesc_feedback_error(feedback, VescFeedbackContract())
    assert reason is not None
    assert expected in reason
