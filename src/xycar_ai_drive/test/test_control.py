# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import hashlib

import numpy as np
import pytest
import yaml
from xycar_ai_drive.artifact import (
    ArtifactContractError,
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
    preprocess_rgb_frame,
)


def test_a_button_toggle_requires_release_and_fault_rearming():
    gate = ToggleDriveGate()
    assert gate.observe(pressed=True, can_enable=True) == ToggleAction.NONE
    assert not gate.enabled
    assert gate.observe(pressed=False, can_enable=True) == ToggleAction.NONE
    assert gate.observe(pressed=True, can_enable=False) == ToggleAction.REJECTED
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

    alias = tmp_path / 'fixture-policy-alias'
    alias.symlink_to(artifact, target_is_directory=True)
    with pytest.raises(ArtifactContractError, match='must not be a symlink'):
        load_policy_artifact(alias)

    (artifact / 'model.ts').write_bytes(b'tampered')
    with pytest.raises(ArtifactContractError, match='checksum mismatch'):
        load_policy_artifact(artifact)
