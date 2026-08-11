"""Fail-closed Unix-socket transport for isolated policy inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import stat
import struct
import threading
import time
from pathlib import Path

import numpy as np

from xycar_ai_drive.artifact import load_policy_artifact
from xycar_ai_drive.control import DriveCommand
from xycar_ai_drive.gpu_runtime import DeviceTorchScriptPolicy
from xycar_ai_drive.policy_runtime import InferenceResult, PolicyRuntimeError

_MAGIC = b'XPAI'
_VERSION = 1
_HEADER = struct.Struct('!4sBBHII')
_IMAGE_HEADER = struct.Struct('!III')
_HELLO = 1
_INFER = 2
_RESET = 3
_ERROR = 255
_MAX_PAYLOAD = 16 * 1024 * 1024


def _artifact_identity(artifact_dir: str | Path) -> tuple[str, str]:
    artifact = load_policy_artifact(artifact_dir)
    checksum_path = artifact.root / 'SHA256SUMS'
    digest = hashlib.sha256(checksum_path.read_bytes()).hexdigest()
    return artifact.root.name, digest


class UnixSocketPolicyClient:
    """Policy interface that refuses stale, malformed, or mismatched servers."""

    def __init__(
        self,
        *,
        artifact_dir: str,
        socket_path: str,
        timeout_sec: float,
        required_device: str,
    ) -> None:
        if not socket_path:
            raise ValueError('socket_path must not be empty')
        if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
            raise ValueError('timeout_sec must be finite and positive')
        if required_device not in {'cpu', 'cuda'}:
            raise ValueError('required_device must be cpu or cuda')
        self._socket_path = socket_path
        self._timeout_sec = float(timeout_sec)
        self._required_device = required_device
        self._artifact_id, self._artifact_digest = _artifact_identity(
            artifact_dir
        )
        self._request_id = 0
        self._lock = threading.RLock()
        self._socket: socket.socket | None = None
        self._connect_and_verify()

    def infer(self, rgb_frame: np.ndarray) -> InferenceResult:
        if (
            not isinstance(rgb_frame, np.ndarray)
            or rgb_frame.dtype != np.uint8
            or rgb_frame.ndim != 3
            or rgb_frame.shape[2] != 3
        ):
            raise PolicyRuntimeError(
                'IPC camera frame must be a uint8 RGB image'
            )
        frame = np.ascontiguousarray(rgb_frame)
        payload = _IMAGE_HEADER.pack(*frame.shape) + frame.tobytes()
        response = self._request(_INFER, payload)
        try:
            result = json.loads(response.decode('utf-8'))
            angle = float(result['angle'])
            speed = float(result['speed'])
            inference_ms = float(result['inference_ms'])
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            self.close()
            raise PolicyRuntimeError(
                'GPU server returned a malformed inference result'
            ) from exc
        if not all(math.isfinite(value) for value in (angle, speed, inference_ms)):
            self.close()
            raise PolicyRuntimeError(
                'GPU server returned a non-finite inference result'
            )
        return InferenceResult(
            command=DriveCommand(angle=angle, speed=speed),
            inference_ms=inference_ms,
        )

    def reset_history(self) -> None:
        self._request(_RESET, b'')

    def close(self) -> None:
        with self._lock:
            if self._socket is not None:
                try:
                    self._socket.close()
                finally:
                    self._socket = None

    def _connect_and_verify(self) -> None:
        self.close()
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self._timeout_sec)
        try:
            connection.connect(self._socket_path)
        except OSError as exc:
            connection.close()
            raise PolicyRuntimeError(
                f'could not connect to GPU policy socket: {exc}'
            ) from exc
        self._socket = connection
        response = self._request(_HELLO, b'', reconnect=False)
        try:
            identity = json.loads(response.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.close()
            raise PolicyRuntimeError('GPU server handshake is malformed') from exc
        expected = {
            'artifact_id': self._artifact_id,
            'artifact_digest': self._artifact_digest,
            'device': self._required_device,
        }
        if any(identity.get(key) != value for key, value in expected.items()):
            self.close()
            raise PolicyRuntimeError(
                f'GPU server identity mismatch: expected {expected}, got {identity}'
            )

    def _request(
        self,
        opcode: int,
        payload: bytes,
        *,
        reconnect: bool = True,
    ) -> bytes:
        if len(payload) > _MAX_PAYLOAD:
            raise PolicyRuntimeError('IPC request exceeds the payload limit')
        with self._lock:
            if self._socket is None:
                if not reconnect:
                    raise PolicyRuntimeError('GPU policy socket is closed')
                self._connect_and_verify()
            assert self._socket is not None
            self._request_id = (self._request_id + 1) & 0xFFFFFFFF
            request_id = self._request_id
            deadline = time.monotonic() + self._timeout_sec
            try:
                self._set_remaining_timeout(deadline)
                self._socket.sendall(
                    _HEADER.pack(
                        _MAGIC,
                        _VERSION,
                        opcode,
                        0,
                        request_id,
                        len(payload),
                    )
                    + payload
                )
                header = self._recv_exact(_HEADER.size, deadline)
                magic, version, status, _reserved, response_id, size = (
                    _HEADER.unpack(header)
                )
                if (
                    magic != _MAGIC
                    or version != _VERSION
                    or response_id != request_id
                    or size > _MAX_PAYLOAD
                ):
                    raise PolicyRuntimeError('GPU server response header is invalid')
                body = self._recv_exact(size, deadline)
                if status == _ERROR:
                    detail = body.decode('utf-8', errors='replace')
                    raise PolicyRuntimeError(f'GPU server rejected request: {detail}')
                if status != opcode:
                    raise PolicyRuntimeError('GPU server response opcode is invalid')
                return body
            except (OSError, TimeoutError, struct.error) as exc:
                self.close()
                raise PolicyRuntimeError(
                    f'GPU policy IPC failed closed: {exc}'
                ) from exc
            except PolicyRuntimeError:
                self.close()
                raise

    def _recv_exact(self, size: int, deadline: float) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            self._set_remaining_timeout(deadline)
            assert self._socket is not None
            chunk = self._socket.recv(size - len(chunks))
            if not chunk:
                raise OSError('GPU policy socket closed during a response')
            chunks.extend(chunk)
        return bytes(chunks)

    def _set_remaining_timeout(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError('GPU policy request timed out')
        assert self._socket is not None
        self._socket.settimeout(remaining)


class PolicyIpcServer:
    def __init__(self, *, socket_path: str, policy: object) -> None:
        self.socket_path = Path(socket_path)
        self.policy = policy
        self.artifact_id, self.artifact_digest = _artifact_identity(
            policy.artifact.root
        )
        self.device = str(policy.device_name)
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._connection: socket.socket | None = None

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        if self._connection is not None:
            try:
                self._connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def serve_forever(self) -> None:
        self.socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.socket_path.exists() or self.socket_path.is_symlink():
            metadata = self.socket_path.lstat()
            if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != os.getuid():
                raise PolicyRuntimeError(
                    f'refusing unsafe existing socket path: {self.socket_path}'
                )
            self.socket_path.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener = listener
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            listener.listen(1)
            listener.settimeout(0.25)
            while not self._stop.is_set():
                try:
                    connection, _address = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                with connection:
                    self._connection = connection
                    try:
                        self._verify_peer(connection)
                        self._serve_connection(connection)
                    except (EOFError, OSError, PolicyRuntimeError):
                        if self._stop.is_set():
                            break
                    finally:
                        self._connection = None
        finally:
            listener.close()
            self._listener = None
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass

    def _verify_peer(self, connection: socket.socket) -> None:
        if not hasattr(socket, 'SO_PEERCRED'):
            return
        credentials = connection.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            struct.calcsize('3i'),
        )
        _pid, uid, _gid = struct.unpack('3i', credentials)
        if uid != os.getuid():
            raise PolicyRuntimeError('GPU policy client uid is not authorized')

    def _serve_connection(self, connection: socket.socket) -> None:
        while not self._stop.is_set():
            try:
                header = _recv_exact_socket(connection, _HEADER.size)
            except EOFError:
                return
            magic, version, opcode, _reserved, request_id, size = _HEADER.unpack(
                header
            )
            if magic != _MAGIC or version != _VERSION or size > _MAX_PAYLOAD:
                return
            payload = _recv_exact_socket(connection, size)
            try:
                response = self._handle(opcode, payload)
                status = opcode
            except Exception as exc:
                response = str(exc).encode('utf-8')[:4096]
                status = _ERROR
                self.policy.reset_history()
            try:
                connection.sendall(
                    _HEADER.pack(
                        _MAGIC,
                        _VERSION,
                        status,
                        0,
                        request_id,
                        len(response),
                    )
                    + response
                )
            except OSError:
                return

    def _handle(self, opcode: int, payload: bytes) -> bytes:
        if opcode == _HELLO:
            if payload:
                raise PolicyRuntimeError('hello payload must be empty')
            return json.dumps(
                {
                    'artifact_id': self.artifact_id,
                    'artifact_digest': self.artifact_digest,
                    'device': self.device,
                },
                sort_keys=True,
            ).encode('utf-8')
        if opcode == _RESET:
            if payload:
                raise PolicyRuntimeError('reset payload must be empty')
            self.policy.reset_history()
            return b''
        if opcode != _INFER or len(payload) < _IMAGE_HEADER.size:
            raise PolicyRuntimeError('unsupported GPU policy request')
        height, width, channels = _IMAGE_HEADER.unpack(
            payload[: _IMAGE_HEADER.size]
        )
        expected_size = height * width * channels
        image_bytes = payload[_IMAGE_HEADER.size :]
        if (
            height < 1
            or width < 1
            or channels != 3
            or expected_size != len(image_bytes)
        ):
            raise PolicyRuntimeError('IPC image shape or payload is invalid')
        image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
            height,
            width,
            channels,
        )
        result = self.policy.infer(image)
        return json.dumps(
            {
                'angle': float(result.command.angle),
                'speed': float(result.command.speed),
                'inference_ms': float(result.inference_ms),
            },
            separators=(',', ':'),
        ).encode('utf-8')


def _recv_exact_socket(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            if chunks:
                raise OSError('socket closed in the middle of a request')
            raise EOFError
        chunks.extend(chunk)
    return bytes(chunks)


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-dir', required=True)
    parser.add_argument('--socket-path', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--torch-num-threads', type=int, default=4)
    parser.add_argument('--warmup-count', type=int, default=3)
    parser.add_argument('--history-reset-timeout-sec', type=float, default=0.25)
    options = parser.parse_args(args)
    policy = DeviceTorchScriptPolicy(
        artifact_dir=options.artifact_dir,
        device=options.device,
        torch_num_threads=options.torch_num_threads,
        warmup_count=options.warmup_count,
        history_reset_timeout_sec=options.history_reset_timeout_sec,
    )
    server = PolicyIpcServer(socket_path=options.socket_path, policy=policy)
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_shutdown(_signum, _frame) -> None:
        server.stop()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    try:
        server.serve_forever()
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == '__main__':
    main()
