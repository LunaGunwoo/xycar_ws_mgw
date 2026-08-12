import hashlib
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from xycar_ai_drive.control import DriveCommand
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
