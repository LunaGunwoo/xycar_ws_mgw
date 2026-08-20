# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import hashlib
import json
import csv
from collections import deque
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from sensor_msgs.msg import Joy
from xycar_data.session_writer import AsyncSessionWriter
from xycar_ai_drive.artifact import (
    ArtifactContractError,
    RoadWarpParameters,
    load_policy_artifact,
)
from xycar_ai_drive.collection_profile import validate_collection_profile
from xycar_ai_drive.control import (
    DriveCommand,
    HoldDriveGate,
    ToggleAction,
    ToggleDriveGate,
    command_history_token_ids,
    decode_class_ids,
    decode_compact_output_ids,
    is_fresh,
)
from xycar_ai_drive.guided_policy_collector import (
    FusedCommand,
    GuideInput,
    GuidedPolicyCollectorNode,
    GuidedPrediction,
    _collection_profile_metadata,
    _validate_control_indices,
    fuse_guided_command,
    trigger_depth,
)
from xycar_ai_drive.policy_runtime import (
    PolicyRuntimeError,
    TorchScriptPolicy,
    preprocess_rgb_frame,
)
from xycar_ai_drive.road_warp import (
    load_road_warp_config,
    save_road_warp_config,
    warp_road_image,
)
from xycar_ai_drive.steering_contract import (
    NORMALIZED_STEERING_CONTRACT,
    parse_steering_contract,
    steering_contract_mapping,
)


def test_motor_relay_default_is_explicit_in_vehicle_config():
    config_path = (
        Path(__file__).parents[1]
        / 'config'
        / 'front_cam_policy.yaml'
    )
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    parameters = config['front_cam_policy']['ros__parameters']
    assert parameters['allowed_motor_relay_nodes'] == ['/ros_bridge']
    assert parameters['a_release_grace_sec'] == pytest.approx(0.12)


def test_a_button_toggle_requires_release_and_fault_rearming():
    gate = ToggleDriveGate()
    assert gate.observe(pressed=True, can_enable=True) == ToggleAction.NONE
    assert not gate.enabled
    assert gate.observe(pressed=False, can_enable=True) == ToggleAction.NONE
    assert (
        gate.observe(pressed=True, can_enable=False) == ToggleAction.REJECTED
    )
    assert not gate.enabled
    assert gate.observe(pressed=True, can_enable=True) == ToggleAction.NONE
    assert gate.observe(pressed=False, can_enable=True) == ToggleAction.NONE
    assert gate.observe(pressed=True, can_enable=True) == ToggleAction.ENABLED
    assert gate.enabled
    assert gate.observe(pressed=False, can_enable=True) == ToggleAction.NONE
    assert gate.observe(pressed=True, can_enable=True) == ToggleAction.DISABLED
    assert not gate.enabled

    gate.observe(pressed=False, can_enable=True)
    gate.observe(pressed=True, can_enable=True)
    assert gate.fault()
    assert not gate.enabled
    assert gate.observe(pressed=True, can_enable=True) == ToggleAction.NONE
    assert gate.observe(pressed=False, can_enable=True) == ToggleAction.NONE
    assert gate.observe(pressed=True, can_enable=True) == ToggleAction.ENABLED


def test_a_button_hold_masks_short_false_pulses_and_stops_on_release():
    gate = HoldDriveGate(release_grace_sec=0.12)
    assert (
        gate.observe(
            pressed=True,
            can_enable=True,
            now_monotonic=1.0,
        )
        == ToggleAction.ENABLED
    )
    assert gate.enabled

    assert (
        gate.observe(
            pressed=False,
            can_enable=True,
            now_monotonic=1.05,
        )
        == ToggleAction.NONE
    )
    assert gate.enabled
    assert (
        gate.observe(
            pressed=True,
            can_enable=True,
            now_monotonic=1.09,
        )
        == ToggleAction.NONE
    )
    assert (
        gate.observe(
            pressed=False,
            can_enable=True,
            now_monotonic=1.20,
        )
        == ToggleAction.NONE
    )
    assert gate.enabled
    assert (
        gate.observe(
            pressed=False,
            can_enable=True,
            now_monotonic=1.22,
        )
        == ToggleAction.DISABLED
    )
    assert not gate.enabled


def test_a_button_hold_requires_release_after_rejection_or_fault():
    gate = HoldDriveGate(release_grace_sec=0.12)
    assert (
        gate.observe(
            pressed=True,
            can_enable=False,
            now_monotonic=1.0,
        )
        == ToggleAction.REJECTED
    )
    assert (
        gate.observe(
            pressed=True,
            can_enable=True,
            now_monotonic=1.05,
        )
        == ToggleAction.NONE
    )
    gate.observe(
        pressed=False,
        can_enable=True,
        now_monotonic=1.18,
    )
    assert (
        gate.observe(
            pressed=True,
            can_enable=True,
            now_monotonic=1.19,
        )
        == ToggleAction.ENABLED
    )
    assert gate.fault()
    assert (
        gate.observe(
            pressed=True,
            can_enable=True,
            now_monotonic=1.20,
        )
        == ToggleAction.NONE
    )
    gate.observe(
        pressed=False,
        can_enable=True,
        now_monotonic=1.33,
    )
    assert (
        gate.observe(
            pressed=True,
            can_enable=True,
            now_monotonic=1.34,
        )
        == ToggleAction.ENABLED
    )


@pytest.mark.parametrize('release_grace_sec', [0.0, -0.1, float('nan')])
def test_a_button_hold_rejects_invalid_release_grace(release_grace_sec):
    with pytest.raises(ValueError, match='finite and positive'):
        HoldDriveGate(release_grace_sec=release_grace_sec)


def test_class_decode_blocks_reverse_without_positive_speed_cap():
    assert decode_class_ids(0, 0).angle == -100.0
    assert decode_class_ids(0, 0).speed == 0.0
    assert decode_class_ids(200, 125).angle == 100.0
    assert decode_class_ids(200, 125).speed == 25.0
    assert decode_class_ids(100, 200).speed == 100.0
    with pytest.raises(ValueError):
        decode_class_ids(-1, 100)
    with pytest.raises(ValueError):
        decode_class_ids(100, 201)


def test_compact_decode_and_executed_history_endpoint_mapping():
    assert decode_compact_output_ids(0, 0) == DriveCommand(-100.0, 0.0)
    assert decode_compact_output_ids(100, 30) == DriveCommand(100.0, 30.0)
    assert command_history_token_ids(DriveCommand(-100.0, 0.0)) == (0, 50)
    assert command_history_token_ids(DriveCommand(100.0, 30.0)) == (100, 80)
    with pytest.raises(ValueError, match=r'\[0, 100\]'):
        decode_compact_output_ids(101, 0)
    with pytest.raises(ValueError, match=r'\[0, 30\]'):
        decode_compact_output_ids(40, 31)


def test_guided_command_uses_rb_takeover_and_preserves_speed_trim():
    fused = fuse_guided_command(
        DriveCommand(angle=20.0, speed=6.0),
        GuideInput(
            steering_axis=0.5,
            steering_takeover=True,
            rt_depth=1.0,
        ),
        max_steering_angle=100.0,
        invert_steering=True,
        rt_speed_increment=2.0,
        lt_speed_decrement=5.0,
        speed_cap=7.0,
        correction_deadzone=0.05,
    )
    assert fused.executed == DriveCommand(angle=-50.0, speed=7.0)
    assert fused.steering_residual == -70.0
    assert fused.speed_delta == 2.0
    assert fused.human_correction

    clamped = fuse_guided_command(
        DriveCommand(angle=-90.0, speed=1.0),
        GuideInput(
            steering_axis=2.0,
            steering_takeover=True,
            lt_depth=1.0,
        ),
        max_steering_angle=100.0,
        invert_steering=False,
        rt_speed_increment=2.0,
        lt_speed_decrement=5.0,
        speed_cap=7.0,
        correction_deadzone=0.05,
    )
    assert clamped.executed == DriveCommand(angle=100.0, speed=0.0)
    assert clamped.steering_residual == 190.0

    neutral = fuse_guided_command(
        DriveCommand(angle=-35.0, speed=6.0),
        GuideInput(),
        max_steering_angle=100.0,
        invert_steering=True,
        rt_speed_increment=2.0,
        lt_speed_decrement=5.0,
        speed_cap=30.0,
        correction_deadzone=0.05,
    )
    assert neutral.executed == DriveCommand(angle=-35.0, speed=6.0)
    assert neutral.steering_residual == 0.0
    assert not neutral.human_correction

    model_out_of_range = fuse_guided_command(
        DriveCommand(angle=120.0, speed=6.0),
        GuideInput(),
        max_steering_angle=100.0,
        invert_steering=True,
        rt_speed_increment=2.0,
        lt_speed_decrement=5.0,
        speed_cap=30.0,
        correction_deadzone=0.05,
    )
    assert model_out_of_range.executed.angle == 100.0

    ignored_stick_with_speed_trim = fuse_guided_command(
        DriveCommand(angle=80.0, speed=6.0),
        GuideInput(steering_axis=0.75, rt_depth=1.0),
        max_steering_angle=100.0,
        invert_steering=True,
        rt_speed_increment=2.0,
        lt_speed_decrement=5.0,
        speed_cap=30.0,
        correction_deadzone=0.05,
    )
    assert ignored_stick_with_speed_trim.executed == DriveCommand(
        angle=80.0,
        speed=8.0,
    )
    assert ignored_stick_with_speed_trim.steering_residual == 0.0
    assert ignored_stick_with_speed_trim.speed_delta == 2.0
    assert ignored_stick_with_speed_trim.human_correction

    ignored_stick = fuse_guided_command(
        DriveCommand(angle=80.0, speed=6.0),
        GuideInput(steering_axis=-1.0),
        max_steering_angle=100.0,
        invert_steering=True,
        rt_speed_increment=2.0,
        lt_speed_decrement=5.0,
        speed_cap=30.0,
        correction_deadzone=0.05,
    )
    assert ignored_stick.executed == DriveCommand(angle=80.0, speed=6.0)
    assert ignored_stick.steering_residual == 0.0
    assert not ignored_stick.human_correction

    neutral_takeover = fuse_guided_command(
        DriveCommand(angle=80.0, speed=6.0),
        GuideInput(steering_takeover=True),
        max_steering_angle=100.0,
        invert_steering=True,
        rt_speed_increment=2.0,
        lt_speed_decrement=5.0,
        speed_cap=30.0,
        correction_deadzone=0.05,
    )
    assert neutral_takeover.executed == DriveCommand(angle=0.0, speed=6.0)
    assert neutral_takeover.steering_residual == -80.0
    assert neutral_takeover.human_correction
    assert trigger_depth(-1.0, 'negative') == 1.0
    assert trigger_depth(1.0, 'signed') == 0.0


def test_guided_drive_rearm_ignores_stick_but_requires_released_rb():
    fake = SimpleNamespace(
        _guide=GuideInput(steering_axis=1.0),
        correction_deadzone=0.05,
        allow_motion=True,
        _unsafe_reason_locked=lambda _now: None,
    )
    assert GuidedPolicyCollectorNode._can_enable_locked(fake, 1.0)

    fake._guide = GuideInput(
        steering_axis=1.0,
        steering_takeover=True,
    )
    assert not GuidedPolicyCollectorNode._can_enable_locked(fake, 1.0)

    fake._guide = GuideInput(steering_axis=1.0, rt_depth=0.1)
    assert not GuidedPolicyCollectorNode._can_enable_locked(fake, 1.0)


def test_guided_control_indices_and_joy_length_include_rb():
    _validate_control_indices((0, 4, 5), (0, 1, 2, 3, 10))
    with pytest.raises(ValueError, match='A/B/X/Y/RB'):
        _validate_control_indices((0, 4, 5), (0, 1, 2, 3, 3))

    failures = []
    fake = SimpleNamespace(
        steering_axis=0,
        lt_axis=4,
        rt_axis=5,
        record_start_button=0,
        record_stop_button=1,
        record_discard_button=2,
        drive_toggle_button=3,
        steering_takeover_button=10,
        _force_off=failures.append,
    )
    message = Joy(axes=[0.0] * 6, buttons=[0] * 10)
    GuidedPolicyCollectorNode._on_joy(fake, message)
    assert failures == ['Joy axis or button array is too short']

    fake._lock = RLock()
    fake._last_buttons = []
    fake._drive_gate = ToggleDriveGate()
    fake._can_enable_locked = lambda _now: False
    fake.trigger_axis_mode = 'negative'
    valid_message = Joy(axes=[0.75, 0.0, 0.0, 0.0, 0.0, 0.0])
    valid_message.buttons = [0] * 11
    valid_message.buttons[10] = 1
    GuidedPolicyCollectorNode._on_joy(fake, valid_message)
    assert fake._guide == GuideInput(
        steering_axis=0.75,
        steering_takeover=True,
    )


def test_stateless_guided_record_uses_executed_label_and_profile_hash(tmp_path):
    profile = tmp_path / 'guided.yaml'
    profile.write_text(
        'guided_policy_collector:\n'
        '  ros__parameters:\n'
        '    max_steering_angle: 100.0\n'
        '    steering_takeover_button: 10\n'
        '    steering_contract: normalized_percent_v2\n',
        encoding='utf-8',
    )
    validate_collection_profile(str(profile))
    assert GuidedPolicyCollectorNode._initial_history(
        SimpleNamespace(artifact=SimpleNamespace(history=None))
    ) is None
    profile_metadata = _collection_profile_metadata(str(profile))
    assert profile_metadata['path'] == str(profile)
    assert profile_metadata['sha256'] == hashlib.sha256(
        profile.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match='existing absolute file'):
        validate_collection_profile(str(tmp_path / 'missing.yaml'))
    legacy_profile = tmp_path / 'legacy.yaml'
    legacy_profile.write_text(
        'guided_policy_collector:\n'
        '  ros__parameters:\n'
        '    residual_gain: 200.0\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='legacy residual_gain'):
        validate_collection_profile(str(legacy_profile))
    missing_angle_profile = tmp_path / 'missing_angle.yaml'
    missing_angle_profile.write_text(
        'guided_policy_collector:\n'
        '  ros__parameters:\n'
        '    speed_cap: 30.0\n'
        '    steering_contract: normalized_percent_v2\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='explicitly set max_steering_angle'):
        validate_collection_profile(str(missing_angle_profile))
    missing_takeover_profile = tmp_path / 'missing_takeover.yaml'
    missing_takeover_profile.write_text(
        'guided_policy_collector:\n'
        '  ros__parameters:\n'
        '    max_steering_angle: 100.0\n'
        '    steering_contract: normalized_percent_v2\n',
        encoding='utf-8',
    )
    with pytest.raises(
        ValueError,
        match='explicitly set steering_takeover_button',
    ):
        validate_collection_profile(str(missing_takeover_profile))

    writer = AsyncSessionWriter(
        tmp_path / 'sessions',
        png_compression=0,
        queue_size=16,
        min_free_space_mb=0,
    )
    token = writer.start_session({'control_mode': 'guided_policy'})
    assert token is not None
    fake = SimpleNamespace(
        _recording_tail=deque(),
        recording_queue_size=128,
        tail_discard_frames=0,
        _session_token=token,
        writer=writer,
    )
    fake._flush_recording_prefix = lambda: (
        GuidedPolicyCollectorNode._flush_recording_prefix(fake)
    )
    prediction = GuidedPrediction(
        sequence=4,
        command=DriveCommand(angle=20.0, speed=6.0),
        source_monotonic=1.0,
        completed_monotonic=1.01,
        inference_ms=3.0,
        image_bgr=np.zeros((4, 6, 3), dtype=np.uint8),
        stamp_sec=1,
        stamp_nanosec=2,
        received_wall_time_ns=3,
    )
    guide = GuideInput(
        steering_axis=0.5,
        steering_takeover=True,
        rt_depth=1.0,
    )
    fused = FusedCommand(
        executed=DriveCommand(angle=-50.0, speed=8.0),
        steering_residual=-70.0,
        speed_delta=2.0,
        human_correction=True,
    )

    GuidedPolicyCollectorNode._record_prediction(
        fake, prediction, guide, fused
    )

    assert not fake._recording_tail
    assert writer.finish(token, 'test_complete')
    assert writer.shutdown()
    results = writer.poll_results()
    assert len(results) == 1
    with (results[0].path / 'samples.csv').open(
        encoding='utf-8', newline=''
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert [(row['angle'], row['speed']) for row in rows] == [
        ('-50.000000', '8.000000')
    ]
    assert [(row['model_angle'], row['model_speed']) for row in rows] == [
        ('20.000000', '6.000000')
    ]
    assert rows[0]['human_correction'] == 'true'


def test_x_discards_guided_session_before_forcing_drive_off():
    events = []
    fake = SimpleNamespace(
        _discard_active_session=lambda **kwargs: events.append(
            ('discard', kwargs)
        ),
        _force_off=lambda reason: events.append(('force_off', reason)),
    )

    GuidedPolicyCollectorNode._discard_session_and_stop(fake)

    assert events == [
        ('discard', {'reason': 'x_button'}),
        ('force_off', 'X button emergency stop'),
    ]


def test_external_collection_templates_and_launch_contract():
    package_root = Path(__file__).parents[1]
    profile_paths = (
        package_root
        / 'config'
        / 'guided_stateless_collection_normalized_v2.yaml',
        package_root
        / 'config'
        / 'guided_policy_collection_normalized_v2.yaml',
    )
    for path in profile_paths:
        validate_collection_profile(str(path))
    retired_v1 = (
        package_root
        / 'config'
        / 'guided_stateless_collection_normalized_v1.yaml'
    )
    with pytest.raises(ValueError, match='normalized_percent_v2'):
        validate_collection_profile(str(retired_v1))
    profiles = [
        yaml.safe_load(path.read_text(encoding='utf-8'))[
            'guided_policy_collector'
        ]['ros__parameters']
        for path in profile_paths
    ]
    profile = profiles[0]
    ar_profile = profiles[1]
    launch_text = (
        package_root / 'launch' / 'jetson_guided_collection.launch.py'
    ).read_text(encoding='utf-8')
    generic_launch_text = (
        package_root / 'launch' / 'guided_policy_collection.launch.py'
    ).read_text(encoding='utf-8')
    collector_text = (
        package_root
        / 'xycar_ai_drive'
        / 'guided_policy_collector.py'
    ).read_text(encoding='utf-8')

    assert profile['recording_root_dir'].endswith('/stateless_guided')
    assert ar_profile['recording_root_dir'].endswith('/teleop_15')
    assert profile['speed_cap'] == 30.0
    assert "DeclareLaunchArgument('speed_cap', default_value='30.0')" in (
        generic_launch_text
    )
    for configured_profile in profiles:
        assert configured_profile['max_steering_angle'] == 100.0
        assert (
            configured_profile['steering_contract']
            == 'normalized_percent_v2'
        )
        assert configured_profile['steering_takeover_button'] == 10
        assert 'residual_gain' not in configured_profile
    assert profile['rt_speed_increment'] == 2.0
    assert profile['lt_speed_decrement'] == 5.0
    assert profile['curriculum_generation'] == 1
    assert profile['allow_motion'] is False
    assert profile['recording_image_format'] == 'jpeg'
    assert profile['recording_jpeg_quality'] == 95
    assert "['params_file:=', params_file]" in launch_text
    assert 'guided_policy_collection_normalized_v2.yaml' in launch_text
    assert 'OpaqueFunction(function=_require_params_file)' in launch_text
    assert 'validate_collection_profile(str(configured))' in launch_text
    assert launch_text.index('OpaqueFunction(function=_require_params_file)') < (
        launch_text.index('ExecuteProcess(')
    )
    assert 'OpaqueFunction(function=_require_params_file)' in (
        generic_launch_text
    )
    assert 'validate_collection_profile(str(configured))' in (
        generic_launch_text
    )
    assert generic_launch_text.index(
        'OpaqueFunction(function=_require_params_file)'
    ) < generic_launch_text.index('IncludeLaunchDescription(')
    for metadata_group in (
        "'steering_contract':",
        "'steering_mode':",
        "'steering_takeover_button':",
        "'collection_profile':",
        "'runtime_safety':",
        "'inference_runtime':",
        "'recording':",
    ):
        assert metadata_group in collector_text
    for required in (
        "DeclareLaunchArgument(\n                'artifact_id'",
        "DeclareLaunchArgument(\n                'allow_motion'",
        "DeclareLaunchArgument(\n                'curriculum_generation'",
        "DeclareLaunchArgument(\n                'speed_cap'",
    ):
        assert required in launch_text


def test_freshness_and_rgb_preprocessing_contract():
    assert is_fresh(1.20, 1.00, 0.25)
    assert not is_fresh(1.26, 1.00, 0.25)
    assert not is_fresh(0.99, 1.00, 0.25)
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    output = preprocess_rgb_frame(
        frame,
        image_size=2,
        mean=np.asarray([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3),
        std=np.asarray([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3),
    )
    assert output.shape == (3, 2, 2)
    assert np.allclose(output[0], 1.0)
    assert np.allclose(output[1:], -1.0)
    with pytest.raises(PolicyRuntimeError):
        preprocess_rgb_frame(
            frame.astype(np.float32),
            image_size=2,
            mean=np.zeros((1, 1, 3), dtype=np.float32),
            std=np.ones((1, 1, 3), dtype=np.float32),
        )


def test_artifact_contract_and_checksum_tampering(tmp_path):
    artifact = tmp_path / 'fixture-policy'
    artifact.mkdir()
    (artifact / 'model.ts').write_bytes(b'model')
    manifest = {
        'schema_version': 1,
        'artifact_id': artifact.name,
        'model': {
            'format': 'torchscript',
            'file': 'model.ts',
            'input': {
                'color_space': 'RGB',
                'dtype': 'float32',
                'shape': [1, 3, 224, 224],
            },
            'output': {
                'kind': 'tuple',
                'order': ['angle_logits', 'speed_logits'],
                'shapes': [[1, 201], [1, 201]],
            },
        },
        'preprocessing': {
            'geometry': 'full_frame_bicubic_resize',
            'image_size': 224,
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
        },
        'label_contract': {
            'num_classes': 201,
            'decode_mapping': 'class_id - 100',
        },
    }
    (artifact / 'manifest.yaml').write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding='utf-8',
    )
    checksum_lines = []
    for name in ('model.ts', 'manifest.yaml'):
        digest = hashlib.sha256((artifact / name).read_bytes()).hexdigest()
        checksum_lines.append(f'{digest}  {name}\n')
    (artifact / 'SHA256SUMS').write_text(
        ''.join(checksum_lines),
        encoding='utf-8',
    )

    contract = load_policy_artifact(artifact)
    assert contract.image_size == 224
    assert contract.model_path == artifact / 'model.ts'
    assert contract.history is None
    assert contract.steering_contract is None
    assert (
        parse_steering_contract(
            steering_contract_mapping(),
            context='fixture.steering_contract',
        )
        == NORMALIZED_STEERING_CONTRACT
    )

    alias = tmp_path / 'fixture-policy-alias'
    alias.symlink_to(artifact, target_is_directory=True)
    with pytest.raises(ArtifactContractError, match='must not be a symlink'):
        load_policy_artifact(alias)

    (artifact / 'model.ts').write_bytes(b'tampered')
    with pytest.raises(ArtifactContractError, match='checksum mismatch'):
        load_policy_artifact(artifact)


def test_v2_artifact_history_contract_and_runtime_queue(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')

    class HistoryEchoPolicy(torch.nn.Module):
        def forward(self, images, history_class_ids):
            del images
            next_ids = torch.clamp(history_class_ids[:, -1] + 1, max=200)
            angle = torch.nn.functional.one_hot(next_ids[:, 0], 201).to(
                torch.float32
            )
            speed = torch.nn.functional.one_hot(next_ids[:, 1], 201).to(
                torch.float32
            )
            return angle, speed

    artifact = tmp_path / 'fixture-ar-policy'
    artifact.mkdir()
    sample_image = torch.zeros(1, 3, 4, 4)
    sample_history = torch.tensor([[[100, 125]] * 4], dtype=torch.long)
    traced = torch.jit.trace(
        HistoryEchoPolicy(),
        (sample_image, sample_history),
        strict=True,
    )
    traced.save(str(artifact / 'model.ts'))
    manifest = {
        'schema_version': 2,
        'artifact_id': artifact.name,
        'model': {
            'format': 'torchscript',
            'file': 'model.ts',
            'architecture': 'ar_control_tokens',
            'input': {
                'kind': 'tuple',
                'order': ['images', 'history_class_ids'],
                'images': {
                    'color_space': 'RGB',
                    'dtype': 'float32',
                    'shape': [1, 3, 4, 4],
                },
                'history_class_ids': {
                    'dtype': 'int64',
                    'shape': [1, 4, 2],
                },
            },
            'output': {
                'kind': 'tuple',
                'order': ['angle_logits', 'speed_logits'],
                'shapes': [[1, 201], [1, 201]],
            },
        },
        'preprocessing': {
            'geometry': 'full_frame_bicubic_resize',
            'image_size': 4,
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
        },
        'label_contract': {
            'num_classes': 201,
            'decode_mapping': 'class_id - 100',
        },
        'history': {
            'frames': 4,
            'pair_order': ['angle_class_id', 'speed_class_id'],
            'time_order': 'oldest_to_newest',
            'initial_command': [0, 25],
            'initial_class_ids': [100, 125],
            'update': 'predicted_argmax',
        },
    }
    (artifact / 'manifest.yaml').write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding='utf-8',
    )
    _write_checksums(artifact)

    contract = load_policy_artifact(artifact)
    assert contract.history is not None
    assert contract.history.frames == 4
    assert contract.history.initial_class_ids == (100, 125)
    runtime = TorchScriptPolicy(
        artifact_dir=str(artifact),
        torch_num_threads=1,
        warmup_count=0,
        history_reset_timeout_sec=0.25,
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert runtime.history_class_ids == ((100, 125),) * 4
    first = runtime.infer(frame)
    assert first.command.angle == 1.0
    assert first.command.speed == 26.0
    assert runtime.history_class_ids[-1] == (101, 126)
    second = runtime.infer(frame)
    assert second.command.angle == 2.0
    assert second.command.speed == 27.0

    runtime.reset_history()
    assert runtime.history_class_ids == ((100, 125),) * 4
    runtime._last_successful_inference_monotonic = 0.0
    monkeypatch.setattr(
        'xycar_ai_drive.policy_runtime.time.monotonic', lambda: 1.0
    )
    after_stale_gap = runtime.infer(frame)
    assert after_stale_gap.command.angle == 1.0
    assert runtime.history_class_ids[-1] == (101, 126)

    class NonFinitePolicy:
        def __call__(self, *_args):
            return (
                torch.full((1, 201), float('nan')),
                torch.zeros(1, 201),
            )

    runtime._model = NonFinitePolicy()
    with pytest.raises(PolicyRuntimeError, match='non-finite'):
        runtime.infer(frame)
    assert runtime.history_class_ids == ((100, 125),) * 4


@pytest.mark.parametrize(
    ('initialization', 'initial_ids', 'initial_command'),
    [
        ('learned_unknown_tokens', [101, 102], None),
        ('canonical_initial_command', [50, 65], [0, 15]),
    ],
)
def test_schema_v5_runtime_uses_compact_initial_and_external_tokens(
    tmp_path,
    initialization,
    initial_ids,
    initial_command,
):
    torch = pytest.importorskip('torch')

    class CompactPolicy(torch.nn.Module):
        def forward(self, images, history_token_ids):
            batch_size = images.shape[0]
            history_signal = history_token_ids[:, :1, :1].to(images.dtype) * 0
            angle = torch.full(
                (batch_size, 101), -10.0, dtype=images.dtype
            ) + history_signal
            speed = torch.full(
                (batch_size, 31), -10.0, dtype=images.dtype
            ) + history_signal
            angle[:, 0] = 10.0
            speed[:, 30] = 10.0
            return angle, speed

    artifact = tmp_path / 'fixture-compact-ar-policy'
    artifact.mkdir()
    sample_image = torch.zeros(1, 3, 4, 4)
    sample_history = torch.tensor([[initial_ids] * 4], dtype=torch.long)
    traced = torch.jit.trace(
        CompactPolicy(),
        (sample_image, sample_history),
        strict=True,
    )
    traced.save(str(artifact / 'model.ts'))
    manifest = {
        'schema_version': 5,
        'artifact_id': artifact.name,
        'model': {
            'format': 'torchscript',
            'file': 'model.ts',
            'architecture': 'ar_control_tokens',
            'control_encoding': 'driver_compact_v2',
            'input': {
                'kind': 'tuple',
                'order': ['images', 'history_token_ids'],
                'images': {
                    'color_space': 'RGB',
                    'dtype': 'float32',
                    'shape': [1, 3, 4, 4],
                },
                'history_token_ids': {
                    'dtype': 'int64',
                    'shape': [1, 4, 2],
                },
            },
            'output': {
                'kind': 'tuple',
                'order': ['angle_logits', 'speed_logits'],
                'shapes': [[1, 101], [1, 31]],
            },
        },
        'preprocessing': {
            'geometry': 'full_frame_bicubic_resize',
            'image_size': 4,
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
        },
        'label_contract': {
            'control_encoding': 'driver_compact_v2',
            'output_shapes': {
                'angle_logits': [1, 101],
                'speed_logits': [1, 31],
            },
            'angle': {'num_classes': 101, 'driver_range': [-50, 50]},
            'speed': {'num_classes': 31, 'command_range': [0, 30]},
            'shared_numeric_vocabulary': {
                'numeric_range': [-50, 50],
                'unknown_angle_token_id': 101,
                'unknown_speed_token_id': 102,
                'angle_query_token_id': 103,
                'speed_query_token_id': 104,
                'vocabulary_size': 105,
            },
        },
        'history': {
            'frames': 4,
            'pair_order': ['angle_token_id', 'speed_token_id'],
            'time_order': 'oldest_to_newest',
            'initialization': initialization,
            'initial_token_ids': initial_ids,
            'actual_angle_token_range': [0, 100],
            'actual_speed_token_range': [50, 80],
            'update': 'externally_executed_commands',
        },
        'steering_contract': steering_contract_mapping(),
    }
    if initial_command is not None:
        manifest['history']['initial_command'] = initial_command
    (artifact / 'manifest.yaml').write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding='utf-8',
    )
    _write_checksums(artifact)

    contract = load_policy_artifact(artifact)
    assert contract.output_shapes == ((1, 101), (1, 31))
    assert contract.history is not None
    assert contract.history.initial_class_ids == tuple(initial_ids)
    runtime = TorchScriptPolicy(
        artifact_dir=str(artifact),
        torch_num_threads=1,
        warmup_count=0,
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    prediction = runtime.infer(frame, [initial_ids] * 4)
    assert prediction.command == DriveCommand(-100.0, 30.0)
    prediction = runtime.infer(
        frame,
        [[0, 50], [25, 55], [50, 65], [100, 80]],
    )
    assert prediction.command == DriveCommand(-100.0, 30.0)
    with pytest.raises(PolicyRuntimeError, match='executed history'):
        runtime.infer(frame, [[101, 50]] * 4)


def test_road_warp_artifact_and_runtime_preprocessing(tmp_path):
    artifact = tmp_path / 'warp-policy'
    artifact.mkdir()
    (artifact / 'model.ts').write_bytes(b'model')
    road_warp = {
        'schema_version': 1,
        'parameters': {
            'top_y': 0.0,
            'bottom_y': 1.0,
            'top_left_x': 0.0,
            'top_right_x': 1.0,
            'bottom_left_x': 0.0,
            'bottom_right_x': 1.0,
            'bev_width': 80,
            'bev_height': 60,
            'dst_left_x': 0.0,
            'dst_right_x': 1.0,
        },
        'sha256': '',
        'source_point_order': [
            'bottom_left',
            'top_left',
            'top_right',
            'bottom_right',
        ],
        'coordinate_space': 'normalized_input_frame',
        'interpolation': 'bilinear',
    }
    canonical = json.dumps(
        {'schema_version': 1, 'warp': road_warp['parameters']},
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    road_warp['sha256'] = hashlib.sha256(canonical).hexdigest()
    manifest = {
        'schema_version': 1,
        'artifact_id': artifact.name,
        'model': {
            'format': 'torchscript',
            'file': 'model.ts',
            'input': {
                'color_space': 'RGB',
                'dtype': 'float32',
                'shape': [1, 3, 20, 20],
            },
            'output': {
                'kind': 'tuple',
                'order': ['angle_logits', 'speed_logits'],
                'shapes': [[1, 201], [1, 201]],
            },
        },
        'preprocessing': {
            'geometry': 'perspective_road_warp_then_bicubic_resize',
            'image_size': 20,
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'road_warp': road_warp,
        },
        'label_contract': {
            'num_classes': 201,
            'decode_mapping': 'class_id - 100',
        },
    }
    (artifact / 'manifest.yaml').write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding='utf-8',
    )
    _write_checksums(artifact)

    contract = load_policy_artifact(artifact)
    assert contract.road_warp == RoadWarpParameters(**road_warp['parameters'])
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frame[:, :40, 0] = 255
    mean = np.asarray([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
    warped = preprocess_rgb_frame(
        frame,
        image_size=20,
        mean=mean,
        std=std,
        road_warp=contract.road_warp,
    )
    full_frame = preprocess_rgb_frame(
        frame,
        image_size=20,
        mean=mean,
        std=std,
    )
    assert np.allclose(warped, full_frame)

    manifest['preprocessing']['road_warp']['parameters']['top_y'] = 0.99
    (artifact / 'manifest.yaml').write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding='utf-8',
    )
    _write_checksums(artifact)
    with pytest.raises(ArtifactContractError, match='sha256 mismatch'):
        load_policy_artifact(artifact)


def test_live_warp_tuning_yaml_round_trip_and_runtime_geometry(tmp_path):
    config = RoadWarpParameters(
        top_y=0.0,
        bottom_y=1.0,
        top_left_x=0.0,
        top_right_x=1.0,
        bottom_left_x=0.0,
        bottom_right_x=1.0,
        bev_width=80,
        bev_height=60,
        dst_left_x=0.0,
        dst_right_x=1.0,
    )
    output_path = tmp_path / 'nested' / 'warp.yaml'
    assert save_road_warp_config(output_path, config) == output_path.resolve()
    assert load_road_warp_config(output_path) == config

    payload = yaml.safe_load(output_path.read_text(encoding='utf-8'))
    assert payload['schema_version'] == 1
    assert tuple(payload['warp']) == (
        'top_y',
        'bottom_y',
        'top_left_x',
        'top_right_x',
        'bottom_left_x',
        'bottom_right_x',
        'bev_width',
        'bev_height',
        'dst_left_x',
        'dst_right_x',
    )
    frame = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
    assert np.array_equal(warp_road_image(frame, config), frame)


def test_live_warp_tuning_yaml_rejects_invalid_geometry(tmp_path):
    output_path = tmp_path / 'warp.yaml'
    output_path.write_text(
        yaml.safe_dump(
            {
                'schema_version': 1,
                'warp': {
                    'top_y': 0.99,
                    'bottom_y': 1.0,
                    'top_left_x': 0.34,
                    'top_right_x': 0.66,
                    'bottom_left_x': 0.0,
                    'bottom_right_x': 1.0,
                    'bev_width': 224,
                    'bev_height': 200,
                    'dst_left_x': 0.0,
                    'dst_right_x': 1.0,
                },
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='at least 0.02 below'):
        load_road_warp_config(output_path)


def _write_checksums(artifact):
    lines = []
    for name in ('model.ts', 'manifest.yaml'):
        digest = hashlib.sha256((artifact / name).read_bytes()).hexdigest()
        lines.append(f'{digest}  {name}\n')
    (artifact / 'SHA256SUMS').write_text(''.join(lines), encoding='utf-8')
