"""Device-aware TorchScript runtime used by the Jetson inference server."""

from __future__ import annotations

import threading
import time

import numpy as np

from xycar_ai_drive.artifact import load_policy_artifact
from xycar_ai_drive.control import decode_class_ids
from xycar_ai_drive.policy_runtime import (
    InferenceResult,
    PolicyRuntimeError,
    preprocess_rgb_frame,
)


class DeviceTorchScriptPolicy:
    """Run a policy on an explicitly requested CPU or CUDA device."""

    def __init__(
        self,
        *,
        artifact_dir: str,
        device: str,
        torch_num_threads: int,
        warmup_count: int,
        history_reset_timeout_sec: float = 0.25,
    ) -> None:
        if device not in {'cpu', 'cuda'}:
            raise ValueError('device must be cpu or cuda')
        if torch_num_threads < 1:
            raise ValueError('torch_num_threads must be positive')
        if warmup_count < 0:
            raise ValueError('warmup_count must be non-negative')
        if (
            not np.isfinite(history_reset_timeout_sec)
            or history_reset_timeout_sec <= 0.0
        ):
            raise ValueError(
                'history_reset_timeout_sec must be finite and positive'
            )

        try:
            import torch
        except ImportError as exc:
            raise PolicyRuntimeError(
                'PyTorch is unavailable in the inference environment'
            ) from exc

        if device == 'cuda' and not torch.cuda.is_available():
            raise PolicyRuntimeError(
                'CUDA was requested but torch.cuda.is_available() is false'
            )

        self.artifact = load_policy_artifact(artifact_dir)
        self.device_name = device
        self._torch = torch
        self._device = torch.device(device)
        self._history_lock = threading.RLock()
        self._history_reset_timeout_sec = float(history_reset_timeout_sec)
        self._last_successful_inference_monotonic: float | None = None
        self._history_class_ids: list[list[int]] | None = None
        self._reset_history_locked()

        if device == 'cpu':
            torch.set_num_threads(torch_num_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

        try:
            self._model = torch.jit.load(
                str(self.artifact.model_path),
                map_location=self._device,
            ).eval()
        except Exception as exc:
            raise PolicyRuntimeError(
                f'could not load TorchScript model on {device}: {exc}'
            ) from exc

        self._mean = np.asarray(
            self.artifact.mean,
            dtype=np.float32,
        ).reshape(1, 1, 3)
        self._std = np.asarray(
            self.artifact.std,
            dtype=np.float32,
        ).reshape(1, 1, 3)
        sample = torch.zeros(
            1,
            3,
            self.artifact.image_size,
            self.artifact.image_size,
            dtype=torch.float32,
            device=self._device,
        )
        with torch.inference_mode():
            for _ in range(warmup_count):
                self._validate_outputs(self._forward(sample))
        self._synchronize()

    def reset_history(self) -> None:
        with self._history_lock:
            self._reset_history_locked()

    def _reset_history_locked(self) -> None:
        history = getattr(self.artifact, 'history', None)
        if history is None:
            self._history_class_ids = None
        else:
            pair = list(history.initial_class_ids)
            self._history_class_ids = [
                pair.copy() for _ in range(history.frames)
            ]
        self._last_successful_inference_monotonic = None

    def infer(self, rgb_frame: np.ndarray) -> InferenceResult:
        chw = preprocess_rgb_frame(
            rgb_frame,
            image_size=self.artifact.image_size,
            mean=self._mean,
            std=self._std,
            road_warp=self.artifact.road_warp,
        )
        tensor = (
            self._torch.from_numpy(chw)
            .unsqueeze(0)
            .to(self._device, non_blocking=False)
        )
        now = time.monotonic()
        try:
            with self._history_lock:
                if (
                    self._last_successful_inference_monotonic is not None
                    and now - self._last_successful_inference_monotonic
                    >= self._history_reset_timeout_sec
                ):
                    self._reset_history_locked()

                self._synchronize()
                started = time.perf_counter()
                with self._torch.inference_mode():
                    outputs = self._forward(tensor)
                self._synchronize()
                inference_ms = (time.perf_counter() - started) * 1000.0
                angle_logits, speed_logits = self._validate_outputs(outputs)
                angle_class_id = int(
                    self._torch.argmax(angle_logits, dim=1).item()
                )
                speed_class_id = int(
                    self._torch.argmax(speed_logits, dim=1).item()
                )
                if self._history_class_ids is not None:
                    self._history_class_ids = [
                        *self._history_class_ids[1:],
                        [angle_class_id, speed_class_id],
                    ]
                self._last_successful_inference_monotonic = time.monotonic()
        except PolicyRuntimeError:
            self.reset_history()
            raise
        except Exception as exc:
            self.reset_history()
            raise PolicyRuntimeError(
                f'TorchScript inference failed on {self.device_name}: {exc}'
            ) from exc

        return InferenceResult(
            command=decode_class_ids(angle_class_id, speed_class_id),
            inference_ms=inference_ms,
        )

    def _forward(self, tensor):
        if self._history_class_ids is None:
            return self._model(tensor)
        history = self._torch.tensor(
            [self._history_class_ids],
            dtype=self._torch.long,
            device=self._device,
        )
        return self._model(tensor, history)

    def _synchronize(self) -> None:
        if self.device_name == 'cuda':
            self._torch.cuda.synchronize(self._device)

    def _validate_outputs(self, outputs: object):
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise PolicyRuntimeError('model output must be a two-tensor tuple')
        angle_logits, speed_logits = outputs
        for name, logits in (
            ('angle_logits', angle_logits),
            ('speed_logits', speed_logits),
        ):
            if not isinstance(logits, self._torch.Tensor):
                raise PolicyRuntimeError(f'{name} must be a tensor')
            if tuple(logits.shape) != (1, 201):
                raise PolicyRuntimeError(
                    f'{name} shape must be [1,201], got {tuple(logits.shape)}'
                )
            if not bool(self._torch.isfinite(logits).all()):
                raise PolicyRuntimeError(f'{name} contains a non-finite value')
        return angle_logits, speed_logits
