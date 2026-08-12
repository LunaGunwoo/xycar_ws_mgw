# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from xycar_ai_drive.artifact import (
    ArtifactContractError,
    RoadWarpParameters,
    load_policy_artifact,
)
from xycar_ai_drive.control import (
    ToggleAction,
    ToggleDriveGate,
    decode_class_ids,
    is_fresh,
)
from xycar_ai_drive.policy_runtime import (
    PolicyRuntimeError,
    TorchScriptPolicy,
    preprocess_rgb_frame,
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


def _write_checksums(artifact):
    lines = []
    for name in ('model.ts', 'manifest.yaml'):
        digest = hashlib.sha256((artifact / name).read_bytes()).hexdigest()
        lines.append(f'{digest}  {name}\n')
    (artifact / 'SHA256SUMS').write_text(''.join(lines), encoding='utf-8')
