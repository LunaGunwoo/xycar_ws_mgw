import hashlib
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from xycar_ai_drive.control import DriveCommand
from xycar_ai_drive.competition_artifact import load_competition_bundle
from xycar_ai_drive.competition_gpu_runtime import (
    CompetitionInference,
    SignalObservation,
    ShortcutObservation,
)
from xycar_ai_drive.competition_ipc import (
    CompetitionIpcClient,
    CompetitionIpcServer,
)
from xycar_ai_drive.policy_ipc import PolicyIpcServer, UnixSocketPolicyClient
from xycar_ai_drive.policy_runtime import InferenceResult, PolicyRuntimeError


class _FakePolicy:
    def __init__(self, artifact, *, device='cuda', delay_sec=0.0):
        self.artifact = SimpleNamespace(root=artifact)
        self.device_name = device
        self.delay_sec = delay_sec
        self.reset_count = 0
        self.last_history = None

    def infer(self, _image, history=None):
        self.last_history = history
        if self.delay_sec:
            time.sleep(self.delay_sec)
        return InferenceResult(
            command=DriveCommand(angle=-18.0, speed=25.0),
            inference_ms=2.5,
        )

    def reset_history(self):
        self.reset_count += 1


def _artifact(tmp_path, *, external_history=False):
    root = tmp_path / 'fixture-policy'
    root.mkdir()
    (root / 'model.ts').write_bytes(b'model')
    manifest = {
        'schema_version': 3 if external_history else 1,
        'artifact_id': root.name,
        'model': {
            'format': 'torchscript',
            'file': 'model.ts',
            'input': {
                'color_space': 'RGB',
                'dtype': 'float32',
                'shape': [1, 3, 4, 4],
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
    }
    if external_history:
        manifest['model']['architecture'] = 'ar_control_tokens'
        manifest['model']['input'] = {
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
        }
        manifest['history'] = {
            'frames': 4,
            'pair_order': ['angle_class_id', 'speed_class_id'],
            'time_order': 'oldest_to_newest',
            'initial_command': [0, 25],
            'initial_class_ids': [100, 125],
            'update': 'externally_executed_commands',
        }
    (root / 'manifest.yaml').write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding='utf-8',
    )
    lines = []
    for name in ('model.ts', 'manifest.yaml'):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        lines.append(f'{digest}  {name}\n')
    (root / 'SHA256SUMS').write_text(''.join(lines), encoding='utf-8')
    return root


def _start_server(socket_path, policy):
    server = PolicyIpcServer(socket_path=str(socket_path), policy=policy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    deadline = time.monotonic() + 2.0
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert socket_path.exists()
    return server, thread


def _stop_server(client, server, thread):
    client.close()
    server.stop()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_ipc_handshake_inference_and_reset(tmp_path):
    artifact = _artifact(tmp_path)
    policy = _FakePolicy(artifact)
    socket_path = tmp_path / 'policy.sock'
    server, thread = _start_server(socket_path, policy)
    client = UnixSocketPolicyClient(
        artifact_dir=str(artifact),
        socket_path=str(socket_path),
        timeout_sec=0.2,
        required_device='cuda',
    )
    result = client.infer(np.zeros((8, 12, 3), dtype=np.uint8))
    assert result.command == DriveCommand(angle=-18.0, speed=25.0)
    assert result.inference_ms == pytest.approx(2.5)
    client.reset_history()
    assert policy.reset_count == 1
    _stop_server(client, server, thread)


def test_v3_ipc_transports_executed_history(tmp_path):
    artifact = _artifact(tmp_path, external_history=True)
    policy = _FakePolicy(artifact)
    socket_path = tmp_path / 'policy.sock'
    server, thread = _start_server(socket_path, policy)
    client = UnixSocketPolicyClient(
        artifact_dir=str(artifact),
        socket_path=str(socket_path),
        timeout_sec=0.2,
        required_device='cuda',
    )
    history = ((90, 120), (91, 121), (92, 122), (93, 123))

    client.infer(np.zeros((8, 12, 3), dtype=np.uint8), history)

    assert policy.last_history == [list(pair) for pair in history]
    with pytest.raises(PolicyRuntimeError, match='requires executed'):
        client.infer(np.zeros((8, 12, 3), dtype=np.uint8))
    _stop_server(client, server, thread)


def test_ipc_rejects_device_mismatch(tmp_path):
    artifact = _artifact(tmp_path)
    socket_path = tmp_path / 'policy.sock'
    server, thread = _start_server(
        socket_path,
        _FakePolicy(artifact, device='cpu'),
    )
    with pytest.raises(PolicyRuntimeError, match='identity mismatch'):
        UnixSocketPolicyClient(
            artifact_dir=str(artifact),
            socket_path=str(socket_path),
            timeout_sec=0.2,
            required_device='cuda',
        )
    server.stop()
    thread.join(timeout=2.0)


def test_ipc_timeout_closes_client_fail_closed(tmp_path):
    artifact = _artifact(tmp_path)
    policy = _FakePolicy(artifact, delay_sec=0.20)
    socket_path = tmp_path / 'policy.sock'
    server, thread = _start_server(socket_path, policy)
    client = UnixSocketPolicyClient(
        artifact_dir=str(artifact),
        socket_path=str(socket_path),
        timeout_sec=0.05,
        required_device='cuda',
    )
    with pytest.raises(PolicyRuntimeError, match='failed closed'):
        client.infer(np.zeros((8, 12, 3), dtype=np.uint8))
    assert client._socket is None
    server.stop()
    thread.join(timeout=2.0)


def test_ipc_server_disconnect_closes_client_fail_closed(tmp_path):
    artifact = _artifact(tmp_path)
    socket_path = tmp_path / 'policy.sock'
    server, thread = _start_server(socket_path, _FakePolicy(artifact))
    client = UnixSocketPolicyClient(
        artifact_dir=str(artifact),
        socket_path=str(socket_path),
        timeout_sec=0.2,
        required_device='cuda',
    )
    server.stop()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    with pytest.raises(PolicyRuntimeError, match='failed closed'):
        client.infer(np.zeros((8, 12, 3), dtype=np.uint8))
    assert client._socket is None


class _FakeCompetitionRuntime:
    def __init__(self, artifact):
        self.artifact = load_competition_bundle(artifact)
        self.device_name = 'cuda'
        self.reset_all_count = 0
        self.reset_shortcut_count = 0

    def infer(self, _image, *, mode, previous_command):
        assert mode == 'normal'
        assert previous_command == DriveCommand(1.0, 2.0)
        return CompetitionInference(
            base_command=DriveCommand(-3.0, 15.0),
            base_confidence=0.8,
            signal=SignalObservation(
                approach=0.9,
                visible=0.8,
                readable=0.7,
                red=0.1,
                yellow=0.2,
                left=0.95,
                green=0.6,
                bbox=(0.1, 0.2, 0.3, 0.4),
                progress=0.5,
            ),
            shortcut=ShortcutObservation(
                command=DriveCommand(4.0, 14.0),
                phase=2,
                handoff_probability=0.25,
            ),
            inference_ms=4.5,
        )

    def reset_all(self):
        self.reset_all_count += 1

    def reset_shortcut(self):
        self.reset_shortcut_count += 1


def _competition_artifact(tmp_path):
    root = tmp_path / 'competition-fixture'
    root.mkdir()
    for name in ('base_model.ts', 'signal_model.ts', 'shortcut_model.ts'):
        (root / name).write_bytes(b'model')
    manifest = {
        'schema_version': 1,
        'artifact_kind': 'competition_bundle',
        'artifact_id': root.name,
        'steering_contract': {
            'schema_version': 1,
            'name': 'normalized_percent_v2',
            'command_min': -100.0,
            'command_max': 100.0,
            'driver_min': -50.0,
            'driver_max': 50.0,
            'mapping': 'linear_scale_0.5',
        },
        'models': {
            'base': {
                'file': 'base_model.ts',
                'model': {'input': {'shape': [1, 3, 16, 16]}},
                'preprocessing': {
                    'geometry': 'full_frame_bicubic_resize',
                    'mean': [0.5, 0.5, 0.5],
                    'std': [0.5, 0.5, 0.5],
                },
            },
            'signal': {
                'file': 'signal_model.ts',
                'config': {'hidden_size': 8},
                'preprocessing': {
                    'geometry': 'upper_two_thirds_bicubic_resize',
                    'input_shape': [1, 3, 8, 12],
                    'mean': [0.5, 0.5, 0.5],
                    'std': [0.5, 0.5, 0.5],
                },
            },
            'shortcut': {
                'file': 'shortcut_model.ts',
                'config': {'hidden_size': 8, 'horizon_steps': 3},
                'preprocessing': {
                    'geometry': 'full_frame_bicubic_resize_unwarped',
                    'input_shape': [1, 3, 16, 16],
                    'mean': [0.5, 0.5, 0.5],
                    'std': [0.5, 0.5, 0.5],
                },
            },
        },
        'mission': {
            'signal_probability_threshold': 0.5,
            'stop_votes': {'required': 2, 'window': 3},
            'go_votes': {'required': 4, 'window': 5},
            'decision_progress_deadline': 0.9,
            'handoff_probability_threshold': 0.9,
            'handoff_consecutive_frames': 5,
            'handoff_max_angle_difference': 25.0,
            'shortcut_timeout_sec': 12.0,
            'action_priority': ['STOP', 'LEFT', 'STRAIGHT'],
        },
        'runtime': {'maximum_forward_speed': 15.0},
    }
    (root / 'manifest.yaml').write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding='utf-8'
    )
    files = ('base_model.ts', 'manifest.yaml', 'shortcut_model.ts', 'signal_model.ts')
    lines = []
    for name in files:
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        lines.append(f'{digest}  {name}\n')
    (root / 'SHA256SUMS').write_text(''.join(lines), encoding='utf-8')
    return root


def test_competition_ipc_transports_all_policy_outputs(tmp_path):
    artifact = _competition_artifact(tmp_path)
    runtime = _FakeCompetitionRuntime(artifact)
    socket_path = tmp_path / 'competition.sock'
    server = CompetitionIpcServer(socket_path=str(socket_path), runtime=runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    deadline = time.monotonic() + 2.0
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert socket_path.exists()
    client = CompetitionIpcClient(
        artifact_dir=str(artifact),
        socket_path=str(socket_path),
        timeout_sec=0.2,
        required_device='cuda',
    )

    result = client.infer(
        np.zeros((8, 12, 3), dtype=np.uint8),
        mode='normal',
        previous_command=DriveCommand(1.0, 2.0),
    )

    assert result.base_command == DriveCommand(-3.0, 15.0)
    assert result.signal.left == pytest.approx(0.95)
    assert result.shortcut.phase == 2
    client.reset_shortcut()
    assert runtime.reset_shortcut_count == 1
    _stop_server(client, server, thread)
