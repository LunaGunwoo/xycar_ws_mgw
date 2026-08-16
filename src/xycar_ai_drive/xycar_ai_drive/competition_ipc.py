"""Fail-closed Unix IPC for the preloaded competition bundle."""

from __future__ import annotations

import argparse
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

from xycar_ai_drive.competition_artifact import load_competition_bundle
from xycar_ai_drive.competition_gpu_runtime import (
    CompetitionGpuRuntime,
    CompetitionInference,
    SignalObservation,
    ShortcutObservation,
)
from xycar_ai_drive.control import DriveCommand
from xycar_ai_drive.policy_runtime import PolicyRuntimeError


_MAGIC = b"XCMP"
_VERSION = 1
_HEADER = struct.Struct("!4sBBHII")
_INFER_HEADER = struct.Struct("!BffIII")
_HELLO = 1
_INFER = 2
_RESET_ALL = 3
_RESET_SHORTCUT = 4
_ERROR = 255
_MAX_PAYLOAD = 16 * 1024 * 1024
_MODE_TO_ID = {
    "signal_only": 1,
    "normal": 2,
    "signal_stop": 3,
    "shortcut": 4,
    "handoff_verify": 5,
}
_ID_TO_MODE = {value: key for key, value in _MODE_TO_ID.items()}


class CompetitionIpcClient:
    def __init__(
        self,
        *,
        artifact_dir: str,
        socket_path: str,
        timeout_sec: float,
        required_device: str,
    ) -> None:
        if not socket_path:
            raise ValueError("socket_path must not be empty")
        if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be finite and positive")
        if required_device not in {"cpu", "cuda"}:
            raise ValueError("required_device must be cpu or cuda")
        self.artifact = load_competition_bundle(artifact_dir)
        self._socket_path = socket_path
        self._timeout_sec = timeout_sec
        self._required_device = required_device
        self._request_id = 0
        self._socket: socket.socket | None = None
        self._lock = threading.RLock()
        self._connect_and_verify()

    def infer(
        self,
        rgb_frame: np.ndarray,
        *,
        mode: str,
        previous_command: DriveCommand,
    ) -> CompetitionInference:
        if mode not in _MODE_TO_ID:
            raise PolicyRuntimeError(f"unsupported competition IPC mode: {mode}")
        if (
            not isinstance(rgb_frame, np.ndarray)
            or rgb_frame.dtype != np.uint8
            or rgb_frame.ndim != 3
            or rgb_frame.shape[2] != 3
        ):
            raise PolicyRuntimeError("IPC frame must be uint8 RGB")
        frame = np.ascontiguousarray(rgb_frame)
        payload = _INFER_HEADER.pack(
            _MODE_TO_ID[mode],
            float(previous_command.angle),
            float(previous_command.speed),
            *frame.shape,
        ) + frame.tobytes()
        response = self._request(_INFER, payload)
        try:
            raw = json.loads(response.decode("utf-8"))
            base = raw.get("base")
            base_command = (
                None
                if base is None
                else DriveCommand(float(base["angle"]), float(base["speed"]))
            )
            base_confidence = None if base is None else float(base["confidence"])
            signal_raw = raw.get("signal")
            signal_value = (
                None
                if signal_raw is None
                else SignalObservation(
                    approach=float(signal_raw["approach"]),
                    visible=float(signal_raw["visible"]),
                    readable=float(signal_raw["readable"]),
                    red=float(signal_raw["red"]),
                    yellow=float(signal_raw["yellow"]),
                    left=float(signal_raw["left"]),
                    green=float(signal_raw["green"]),
                    bbox=tuple(float(value) for value in signal_raw["bbox"]),
                    progress=float(signal_raw["progress"]),
                )
            )
            shortcut_raw = raw.get("shortcut")
            shortcut_value = (
                None
                if shortcut_raw is None
                else ShortcutObservation(
                    command=DriveCommand(
                        float(shortcut_raw["angle"]),
                        float(shortcut_raw["speed"]),
                    ),
                    phase=int(shortcut_raw["phase"]),
                    handoff_probability=float(shortcut_raw["handoff"]),
                )
            )
            result = CompetitionInference(
                base_command=base_command,
                base_confidence=base_confidence,
                signal=signal_value,
                shortcut=shortcut_value,
                inference_ms=float(raw["inference_ms"]),
            )
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            self.close()
            raise PolicyRuntimeError("malformed competition IPC response") from exc
        _validate_result_finite(result)
        return result

    def reset_all(self) -> None:
        self._request(_RESET_ALL, b"")

    def reset_shortcut(self) -> None:
        self._request(_RESET_SHORTCUT, b"")

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
                f"could not connect to competition GPU socket: {exc}"
            ) from exc
        self._socket = connection
        response = self._request(_HELLO, b"", reconnect=False)
        try:
            identity = json.loads(response.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.close()
            raise PolicyRuntimeError("competition handshake is malformed") from exc
        expected = {
            "artifact_id": self.artifact.artifact_id,
            "artifact_digest": self.artifact.digest,
            "device": self._required_device,
        }
        if any(identity.get(key) != value for key, value in expected.items()):
            self.close()
            raise PolicyRuntimeError(
                f"competition GPU identity mismatch: expected {expected}, got {identity}"
            )

    def _request(
        self,
        opcode: int,
        payload: bytes,
        *,
        reconnect: bool = True,
    ) -> bytes:
        if len(payload) > _MAX_PAYLOAD:
            raise PolicyRuntimeError("competition IPC payload is too large")
        with self._lock:
            if self._socket is None:
                if not reconnect:
                    raise PolicyRuntimeError("competition IPC socket is closed")
                self._connect_and_verify()
            assert self._socket is not None
            self._request_id = (self._request_id + 1) & 0xFFFFFFFF
            request_id = self._request_id
            deadline = time.monotonic() + self._timeout_sec
            try:
                self._set_timeout(deadline)
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
                    raise PolicyRuntimeError("competition IPC header is invalid")
                body = self._recv_exact(size, deadline)
                if status == _ERROR:
                    raise PolicyRuntimeError(
                        "competition GPU rejected request: "
                        + body.decode("utf-8", errors="replace")
                    )
                if status != opcode:
                    raise PolicyRuntimeError("competition IPC opcode mismatch")
                return body
            except (OSError, TimeoutError, struct.error) as exc:
                self.close()
                raise PolicyRuntimeError(
                    f"competition IPC failed closed: {exc}"
                ) from exc
            except PolicyRuntimeError:
                self.close()
                raise

    def _recv_exact(self, size: int, deadline: float) -> bytes:
        result = bytearray()
        while len(result) < size:
            self._set_timeout(deadline)
            assert self._socket is not None
            chunk = self._socket.recv(size - len(result))
            if not chunk:
                raise OSError("competition IPC socket closed")
            result.extend(chunk)
        return bytes(result)

    def _set_timeout(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError("competition IPC timed out")
        assert self._socket is not None
        self._socket.settimeout(remaining)


class CompetitionIpcServer:
    def __init__(self, *, socket_path: str, runtime: CompetitionGpuRuntime) -> None:
        self.socket_path = Path(socket_path)
        self.runtime = runtime
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
                    f"refusing unsafe socket path: {self.socket_path}"
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
                        self.runtime.reset_all()
        finally:
            listener.close()
            self._listener = None
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass

    def _verify_peer(self, connection: socket.socket) -> None:
        if not hasattr(socket, "SO_PEERCRED"):
            return
        credentials = connection.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            struct.calcsize("3i"),
        )
        _pid, uid, _gid = struct.unpack("3i", credentials)
        if uid != os.getuid():
            raise PolicyRuntimeError("competition IPC peer uid is unauthorized")

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
                self.runtime.reset_all()
                response = str(exc).encode("utf-8")[:4096]
                status = _ERROR
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
                raise PolicyRuntimeError("hello payload must be empty")
            return json.dumps(
                {
                    "artifact_id": self.runtime.artifact.artifact_id,
                    "artifact_digest": self.runtime.artifact.digest,
                    "device": self.runtime.device_name,
                },
                sort_keys=True,
            ).encode("utf-8")
        if opcode == _RESET_ALL:
            if payload:
                raise PolicyRuntimeError("reset payload must be empty")
            self.runtime.reset_all()
            return b""
        if opcode == _RESET_SHORTCUT:
            if payload:
                raise PolicyRuntimeError("shortcut reset payload must be empty")
            self.runtime.reset_shortcut()
            return b""
        if opcode != _INFER or len(payload) < _INFER_HEADER.size:
            raise PolicyRuntimeError("unsupported competition IPC request")
        mode_id, angle, speed, height, width, channels = _INFER_HEADER.unpack(
            payload[: _INFER_HEADER.size]
        )
        mode = _ID_TO_MODE.get(mode_id)
        image_bytes = payload[_INFER_HEADER.size :]
        expected_size = height * width * channels
        if (
            mode is None
            or channels != 3
            or height < 1
            or width < 1
            or len(image_bytes) != expected_size
        ):
            raise PolicyRuntimeError("competition IPC inference payload is invalid")
        frame = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
            height,
            width,
            channels,
        )
        result = self.runtime.infer(
            frame,
            mode=mode,
            previous_command=DriveCommand(angle=float(angle), speed=float(speed)),
        )
        return json.dumps(
            _result_payload(result),
            separators=(",", ":"),
        ).encode("utf-8")


def _result_payload(result: CompetitionInference) -> dict[str, object]:
    base = None
    if result.base_command is not None:
        base = {
            "angle": result.base_command.angle,
            "speed": result.base_command.speed,
            "confidence": result.base_confidence,
        }
    signal_payload = None
    if result.signal is not None:
        signal_payload = {
            "approach": result.signal.approach,
            "visible": result.signal.visible,
            "readable": result.signal.readable,
            "red": result.signal.red,
            "yellow": result.signal.yellow,
            "left": result.signal.left,
            "green": result.signal.green,
            "bbox": list(result.signal.bbox),
            "progress": result.signal.progress,
        }
    shortcut = None
    if result.shortcut is not None:
        shortcut = {
            "angle": result.shortcut.command.angle,
            "speed": result.shortcut.command.speed,
            "phase": result.shortcut.phase,
            "handoff": result.shortcut.handoff_probability,
        }
    return {
        "base": base,
        "signal": signal_payload,
        "shortcut": shortcut,
        "inference_ms": result.inference_ms,
    }


def _validate_result_finite(result: CompetitionInference) -> None:
    values: list[float] = [result.inference_ms]
    if result.base_command is not None:
        values.extend((result.base_command.angle, result.base_command.speed))
        if result.base_confidence is not None:
            values.append(result.base_confidence)
    if result.signal is not None:
        values.extend(
            (
                result.signal.approach,
                result.signal.visible,
                result.signal.readable,
                result.signal.red,
                result.signal.yellow,
                result.signal.left,
                result.signal.green,
                *result.signal.bbox,
                result.signal.progress,
            )
        )
    if result.shortcut is not None:
        values.extend(
            (
                result.shortcut.command.angle,
                result.shortcut.command.speed,
                result.shortcut.handoff_probability,
            )
        )
    if not all(math.isfinite(value) for value in values):
        raise PolicyRuntimeError("competition IPC result contains non-finite values")


def _recv_exact_socket(connection: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = connection.recv(size - len(result))
        if not chunk:
            if result:
                raise OSError("socket closed mid-request")
            raise EOFError
        result.extend(chunk)
    return bytes(result)


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--warmup-count", type=int, default=3)
    options = parser.parse_args(args)
    runtime = CompetitionGpuRuntime(
        artifact_dir=options.artifact_dir,
        device=options.device,
        torch_num_threads=options.torch_num_threads,
        warmup_count=options.warmup_count,
    )
    server = CompetitionIpcServer(
        socket_path=options.socket_path,
        runtime=runtime,
    )
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


if __name__ == "__main__":
    main()
