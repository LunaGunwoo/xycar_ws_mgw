# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

from types import SimpleNamespace
import signal
import subprocess

import pytest

from xycar_data.teleop_recorder import (
    CameraStartupError,
    HeadlessCameraProcess,
    TeleopRecorderNode,
)
from xycar_data.tuning import SensorConfig


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))


class _Clock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def advance(self, seconds=0.1):
        self.value += seconds


class _CameraHarness:
    prepare_camera = TeleopRecorderNode.prepare_camera
    _reject_gui_viewer = TeleopRecorderNode._reject_gui_viewer

    def __init__(self, sensors=None):
        self.tuning = SimpleNamespace(
            sensors=sensors or SensorConfig(),
            topics=SimpleNamespace(camera_topic="/image_raw"),
        )
        self._owned_camera = None
        self._last_camera_decode_error = None
        self.publishers = ()
        self.viewers = ()
        self.fresh = False
        self.logger = _Logger()

    def get_logger(self):
        return self.logger

    def camera_publishers(self):
        return self.publishers

    def camera_viewers(self):
        return self.viewers

    def _camera_is_fresh(self, _now):
        return self.fresh


class _FakeCameraProcess:
    def __init__(self, returncode=None):
        self.start_count = 0
        self.returncode = returncode

    def start(self):
        self.start_count += 1


def _spin(clock, callback=None):
    def spin_once(_node, *, timeout_sec):
        assert timeout_sec == 0.05
        clock.advance()
        if callback is not None:
            callback(clock.value)

    return spin_once


def test_reuses_existing_headless_camera_without_starting_process():
    clock = _Clock()
    node = _CameraHarness()
    node.publishers = ("/xycar_cam",)
    node.fresh = True
    factories = []

    node.prepare_camera(
        spin_once=_spin(clock),
        monotonic=clock.monotonic,
        process_factory=lambda: factories.append(object()),
    )

    assert factories == []
    assert node._owned_camera is None


def test_rejects_gui_viewer_without_starting_or_stopping_it():
    clock = _Clock()
    node = _CameraHarness()
    node.publishers = ("/xycar_cam",)
    node.viewers = ("/examine_image",)
    factories = []

    with pytest.raises(CameraStartupError, match="GUI camera viewer"):
        node.prepare_camera(
            spin_once=_spin(clock),
            monotonic=clock.monotonic,
            process_factory=lambda: factories.append(object()),
        )

    assert factories == []
    assert node._owned_camera is None


def test_no_publisher_starts_owned_headless_camera_then_accepts_frame():
    clock = _Clock()
    node = _CameraHarness()
    process = _FakeCameraProcess()

    def frame_after_start(_now):
        if process.start_count:
            node.publishers = ("/xycar_cam",)
            node.fresh = True

    node.prepare_camera(
        spin_once=_spin(clock, frame_after_start),
        monotonic=clock.monotonic,
        process_factory=lambda: process,
    )

    assert process.start_count == 1
    assert node._owned_camera is process


def test_existing_publisher_without_frame_never_starts_duplicate():
    clock = _Clock()
    sensors = SensorConfig(
        camera_discovery_timeout_sec=0.1,
        camera_start_timeout_sec=1.0,
    )
    node = _CameraHarness(sensors)
    node.publishers = ("/xycar_cam",)
    factories = []

    with pytest.raises(CameraStartupError, match="publisher exists"):
        node.prepare_camera(
            spin_once=_spin(clock),
            monotonic=clock.monotonic,
            process_factory=lambda: factories.append(object()),
        )

    assert factories == []
    assert node._owned_camera is None


def test_decode_failure_is_reported_at_startup_timeout():
    clock = _Clock()
    sensors = SensorConfig(
        camera_discovery_timeout_sec=0.1,
        camera_start_timeout_sec=1.0,
    )
    node = _CameraHarness(sensors)
    node.publishers = ("/xycar_cam",)
    node._last_camera_decode_error = "unsupported encoding"

    with pytest.raises(CameraStartupError, match="decode failed"):
        node.prepare_camera(
            spin_once=_spin(clock),
            monotonic=clock.monotonic,
        )


def test_owned_camera_early_exit_is_reported():
    clock = _Clock()
    sensors = SensorConfig(
        camera_discovery_timeout_sec=0.1,
        camera_start_timeout_sec=1.0,
    )
    node = _CameraHarness(sensors)
    process = _FakeCameraProcess(returncode=3)

    with pytest.raises(CameraStartupError, match=r"exited.*code 3"):
        node.prepare_camera(
            spin_once=_spin(clock),
            monotonic=clock.monotonic,
            process_factory=lambda: process,
        )

    assert process.start_count == 1
    assert node._owned_camera is process


def test_runtime_owned_camera_exit_immediately_enters_failure_path():
    reasons = []
    node = SimpleNamespace(
        _owned_camera=_FakeCameraProcess(returncode=7),
        exit_requested=False,
        _fail=lambda reason: reasons.append(reason),
    )

    TeleopRecorderNode._on_control_timer(node)

    assert len(reasons) == 1
    assert "exited with code 7" in reasons[0]


def test_fatal_failure_requests_incomplete_session_close():
    exits = []
    errors = []
    node = SimpleNamespace(
        _fatal_reason=None,
        exit_code=0,
        get_logger=lambda: SimpleNamespace(
            error=lambda message: errors.append(message)
        ),
        request_exit=lambda reason, *, complete: exits.append(
            (reason, complete)
        ),
    )

    TeleopRecorderNode._fail(node, "owned camera exited")

    assert node.exit_code == 1
    assert exits == [("owned camera exited", False)]
    assert errors == ["Teleop recorder failure: owned camera exited"]


def test_camera_runtime_diagnostic_is_throttled_by_status():
    logger = _Logger()
    node = SimpleNamespace(
        camera_publishers=lambda: (),
        tuning=SimpleNamespace(
            topics=SimpleNamespace(camera_topic="/image_raw")
        ),
        _last_camera_decode_error=None,
        _camera=None,
        _diagnostic_times={},
        get_logger=lambda: logger,
    )

    TeleopRecorderNode._report_camera_unavailable(
        node,
        1.0,
        prefix="Safety stop",
    )
    TeleopRecorderNode._report_camera_unavailable(
        node,
        2.0,
        prefix="Safety stop",
    )
    TeleopRecorderNode._report_camera_unavailable(
        node,
        3.0,
        prefix="Safety stop",
    )

    assert len(logger.messages) == 2
    assert all(
        "no publisher is present" in message
        for _level, message in logger.messages
    )


def test_auto_start_disabled_reports_missing_publisher():
    clock = _Clock()
    sensors = SensorConfig(
        camera_auto_start=False,
        camera_discovery_timeout_sec=0.1,
    )
    node = _CameraHarness(sensors)

    with pytest.raises(CameraStartupError, match="camera_auto_start is false"):
        node.prepare_camera(
            spin_once=_spin(clock),
            monotonic=clock.monotonic,
        )


def test_headless_process_uses_exact_command_and_new_process_group():
    calls = []
    process = SimpleNamespace(poll=lambda: None)

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return process

    camera = HeadlessCameraProcess(popen=popen)
    camera.start()

    assert calls == [
        (
            ["ros2", "launch", "xycar_cam", "xycar_cam.launch.py"],
            {"start_new_session": True},
        )
    ]


def test_owned_process_shutdown_uses_sigint_then_sigterm():
    signals = []

    class Process:
        pid = 4321

        def __init__(self):
            self.returncode = None
            self.wait_calls = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired("camera", timeout)
            self.returncode = 0
            return 0

    process = Process()
    camera = HeadlessCameraProcess(
        popen=lambda *_args, **_kwargs: process,
        killpg=lambda pid, signum: signals.append((pid, signum)),
    )
    camera.start()
    camera.stop(0.1)

    assert signals == [
        (4321, signal.SIGINT),
        (4321, signal.SIGTERM),
    ]


def test_unowned_external_camera_has_no_process_to_stop():
    signals = []
    camera = HeadlessCameraProcess(
        popen=lambda *_args, **_kwargs: pytest.fail("must not start"),
        killpg=lambda pid, signum: signals.append((pid, signum)),
    )

    camera.stop(0.1)

    assert signals == []
