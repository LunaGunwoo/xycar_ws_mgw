"""Preload both mission policies and serve them behind one CUDA lock."""

from __future__ import annotations

import argparse
import signal
import threading

from xycar_ai_drive.gpu_runtime import DeviceTorchScriptPolicy
from xycar_ai_drive.policy_ipc import PolicyIpcServer
from xycar_ai_drive.policy_runtime import PolicyRuntimeError
from xycar_ai_drive.traffic_shortcut_artifact import (
    load_traffic_shortcut_bundle,
)


class _LockedPolicy:
    def __init__(self, policy: object, lock: threading.RLock) -> None:
        self._policy = policy
        self._lock = lock
        self.artifact = policy.artifact
        self.device_name = policy.device_name

    def infer(self, *args, **kwargs):
        with self._lock:
            return self._policy.infer(*args, **kwargs)

    def reset_history(self) -> None:
        with self._lock:
            self._policy.reset_history()


class DualPolicyIpcService:
    """Own two IPC servers whose CUDA work is strictly serialized."""

    def __init__(
        self,
        *,
        bundle_dir: str,
        base_socket_path: str,
        shortcut_socket_path: str,
        device: str,
        torch_num_threads: int,
        warmup_count: int,
        history_reset_timeout_sec: float,
    ) -> None:
        if base_socket_path == shortcut_socket_path:
            raise ValueError('base and shortcut socket paths must differ')
        self.bundle = load_traffic_shortcut_bundle(bundle_dir)
        shared_lock = threading.RLock()
        # Both constructors complete model load and warm-up before either
        # server socket is opened.
        base_policy = DeviceTorchScriptPolicy(
            artifact_dir=str(self.bundle.base.root),
            device=device,
            torch_num_threads=torch_num_threads,
            warmup_count=warmup_count,
            history_reset_timeout_sec=history_reset_timeout_sec,
        )
        shortcut_policy = DeviceTorchScriptPolicy(
            artifact_dir=str(self.bundle.shortcut.root),
            device=device,
            torch_num_threads=torch_num_threads,
            warmup_count=warmup_count,
            history_reset_timeout_sec=history_reset_timeout_sec,
        )
        self.base_server = PolicyIpcServer(
            socket_path=base_socket_path,
            policy=_LockedPolicy(base_policy, shared_lock),
        )
        self.shortcut_server = PolicyIpcServer(
            socket_path=shortcut_socket_path,
            policy=_LockedPolicy(shortcut_policy, shared_lock),
        )
        self._servers = (self.base_server, self.shortcut_server)
        self._stop = threading.Event()
        self._failure: BaseException | None = None
        self._failure_lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def serve_forever(self) -> None:
        for name, server in (
            ('base-policy-ipc', self.base_server),
            ('shortcut-policy-ipc', self.shortcut_server),
        ):
            thread = threading.Thread(
                target=self._serve_one,
                args=(server,),
                name=name,
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()
        self._stop.wait()
        self.stop()
        for thread in self._threads:
            thread.join(timeout=2.0)
        if self._failure is not None:
            raise PolicyRuntimeError(
                f'dual policy IPC server failed: {self._failure}'
            ) from self._failure

    def stop(self) -> None:
        self._stop.set()
        for server in self._servers:
            server.stop()

    def _serve_one(self, server: PolicyIpcServer) -> None:
        try:
            server.serve_forever()
        except BaseException as exc:  # noqa: BLE001 - thread boundary
            with self._failure_lock:
                if self._failure is None:
                    self._failure = exc
            self.stop()


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--base-socket-path', required=True)
    parser.add_argument('--shortcut-socket-path', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--torch-num-threads', type=int, default=4)
    parser.add_argument('--warmup-count', type=int, default=3)
    parser.add_argument('--history-reset-timeout-sec', type=float, default=0.25)
    options = parser.parse_args(args)
    service = DualPolicyIpcService(
        bundle_dir=options.bundle_dir,
        base_socket_path=options.base_socket_path,
        shortcut_socket_path=options.shortcut_socket_path,
        device=options.device,
        torch_num_threads=options.torch_num_threads,
        warmup_count=options.warmup_count,
        history_reset_timeout_sec=options.history_reset_timeout_sec,
    )
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_shutdown(_signum, _frame) -> None:
        service.stop()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    try:
        service.serve_forever()
    finally:
        service.stop()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == '__main__':
    main()
