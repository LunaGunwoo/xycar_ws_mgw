"""Device-aware TorchScript runtime used by the Jetson inference server."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence

import numpy as np

from xycar_ai_drive.artifact import (
    ANGLE_REGRESSION_FIXED_SPEED_PREDICTION_MODE,
    CATEGORICAL_PREDICTION_MODE,
    COMPACT_CONTROL_ENCODING,
    CONTINUOUS_REGRESSION_PREDICTION_MODE,
    load_policy_artifact,
)
from xycar_ai_drive.control import (
    DriveCommand,
    decode_class_ids,
    decode_compact_output_ids,
    decode_regression_outputs,
)
from xycar_ai_drive.policy_runtime import (
    InferenceResult,
    PolicyRuntimeError,
    normalize_rgb_geometry,
    prepare_rgb_geometry,
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
                supplied_history = (
                    self._history_class_ids
                    if self.artifact.history is not None
                    and self.artifact.history.update
                    == 'externally_executed_commands'
                    else None
                )
                self._validate_outputs(
                    self._forward(
                        sample,
                        history_class_ids=supplied_history,
                    )
                )
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

    def infer(
        self,
        rgb_frame: np.ndarray,
        history_class_ids: Sequence[Sequence[int]] | None = None,
    ) -> InferenceResult:
        tensor = self.prepare_tensor(rgb_frame)
        return self.infer_preprocessed(
            tensor,
            history_class_ids=history_class_ids,
        )

    def prepare_tensor(self, rgb_frame: np.ndarray):
        geometry = self.prepare_geometry(rgb_frame)
        return self.prepare_tensor_from_geometry(geometry)

    def prepare_geometry(self, rgb_frame: np.ndarray) -> np.ndarray:
        return prepare_rgb_geometry(
            rgb_frame,
            image_size=self.artifact.image_size,
            road_warp=self.artifact.road_warp,
        )

    def prepare_tensor_from_geometry(self, geometry: np.ndarray):
        chw = normalize_rgb_geometry(
            geometry,
            image_size=self.artifact.image_size,
            mean=self._mean,
            std=self._std,
        )
        return (
            self._torch.from_numpy(chw)
            .unsqueeze(0)
            .to(self._device, non_blocking=False)
        )

    def infer_preprocessed(
        self,
        tensor,
        history_class_ids: Sequence[Sequence[int]] | None = None,
    ) -> InferenceResult:
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
                    outputs = self._forward(
                        tensor,
                        history_class_ids=history_class_ids,
                    )
                self._synchronize()
                inference_ms = (time.perf_counter() - started) * 1000.0
                angle_logits, speed_logits = self._validate_outputs(outputs)
                if (
                    self.artifact.prediction_mode
                    == ANGLE_REGRESSION_FIXED_SPEED_PREDICTION_MODE
                ):
                    assert self.artifact.fixed_speed is not None
                    command = DriveCommand(
                        angle=float(angle_logits.item()) * 100.0,
                        speed=self.artifact.fixed_speed,
                    )
                    angle_class_id = speed_class_id = None
                elif (
                    self.artifact.prediction_mode
                    == CONTINUOUS_REGRESSION_PREDICTION_MODE
                ):
                    command = decode_regression_outputs(
                        float(angle_logits.item()),
                        float(speed_logits.item()),
                        speed_max=self.artifact.speed_output_max,
                    )
                    angle_class_id = speed_class_id = None
                else:
                    angle_class_id = int(
                        self._torch.argmax(angle_logits, dim=1).item()
                    )
                    speed_class_id = int(
                        self._torch.argmax(speed_logits, dim=1).item()
                    )
                if (
                    self._history_class_ids is not None
                    and self.artifact.history is not None
                    and self.artifact.history.update == 'predicted_argmax'
                ):
                    if angle_class_id is None or speed_class_id is None:
                        raise PolicyRuntimeError(
                            'regression artifacts require external history'
                        )
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

        if self.artifact.prediction_mode == CATEGORICAL_PREDICTION_MODE:
            decoder = (
                decode_compact_output_ids
                if self.artifact.control_encoding == COMPACT_CONTROL_ENCODING
                else decode_class_ids
            )
            command = decoder(angle_class_id, speed_class_id)
        return InferenceResult(
            command=command,
            inference_ms=inference_ms,
        )

    def _forward(self, tensor, *, history_class_ids=None):
        if (
            self.artifact.prediction_mode
            == ANGLE_REGRESSION_FIXED_SPEED_PREDICTION_MODE
        ):
            if history_class_ids is not None:
                raise PolicyRuntimeError(
                    'fixed-speed policy does not accept executed history'
                )
            fixed_speed = self.artifact.fixed_speed
            divisor = self.artifact.speed_normalization_divisor
            if fixed_speed is None or divisor is None:
                raise PolicyRuntimeError(
                    'fixed-speed artifact runtime values are missing'
                )
            speed = self._torch.tensor(
                [[fixed_speed / divisor]],
                dtype=self._torch.float32,
                device=self._device,
            )
            return self._model(tensor, speed)
        history_contract = self.artifact.history
        if history_contract is None:
            if history_class_ids is not None:
                raise PolicyRuntimeError(
                    'stateless policy does not accept executed history'
                )
            return self._model(tensor)
        if history_contract.update == 'predicted_argmax':
            if history_class_ids is not None:
                raise PolicyRuntimeError(
                    'schema v2 policy owns its predicted history'
                )
            values = self._history_class_ids
        else:
            values = self._validate_external_history(history_class_ids)
        history = self._torch.tensor(
            [values],
            dtype=self._torch.long,
            device=self._device,
        )
        return self._model(tensor, history)

    def _validate_external_history(self, supplied):
        history = self.artifact.history
        if history is None or supplied is None:
            raise PolicyRuntimeError(
                'schema v3 policy requires executed command history'
            )
        values = [list(pair) for pair in supplied]
        if len(values) != history.frames or any(
            not history.valid_pair(pair) for pair in values
        ):
            raise PolicyRuntimeError(
                'executed history must contain four [angle,speed] class pairs'
            )
        return values

    def _synchronize(self) -> None:
        if self.device_name == 'cuda':
            self._torch.cuda.synchronize(self._device)

    def _validate_outputs(self, outputs: object):
        if (
            self.artifact.prediction_mode
            == ANGLE_REGRESSION_FIXED_SPEED_PREDICTION_MODE
        ):
            if not isinstance(outputs, self._torch.Tensor):
                raise PolicyRuntimeError('angle output must be a tensor')
            expected_shape = self.artifact.output_shapes[0]
            if tuple(outputs.shape) != expected_shape:
                raise PolicyRuntimeError(
                    f'angle output shape must be {list(expected_shape)}, '
                    f'got {tuple(outputs.shape)}'
                )
            if not bool(self._torch.isfinite(outputs).all()):
                raise PolicyRuntimeError(
                    'angle output contains a non-finite value'
                )
            value = float(outputs.item())
            if not -1.0 <= value <= 1.0:
                raise PolicyRuntimeError(
                    'normalized angle output must be in [-1,1]'
                )
            return outputs, None
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise PolicyRuntimeError('model output must be a two-tensor tuple')
        angle_logits, speed_logits = outputs
        for name, logits, expected_shape in (
            ('angle_logits', angle_logits, self.artifact.output_shapes[0]),
            ('speed_logits', speed_logits, self.artifact.output_shapes[1]),
        ):
            if not isinstance(logits, self._torch.Tensor):
                raise PolicyRuntimeError(f'{name} must be a tensor')
            if tuple(logits.shape) != expected_shape:
                raise PolicyRuntimeError(
                    f'{name} shape must be {list(expected_shape)}, '
                    f'got {tuple(logits.shape)}'
                )
            if not bool(self._torch.isfinite(logits).all()):
                raise PolicyRuntimeError(f'{name} contains a non-finite value')
        if (
            self.artifact.prediction_mode
            == CONTINUOUS_REGRESSION_PREDICTION_MODE
        ):
            try:
                decode_regression_outputs(
                    float(angle_logits.item()),
                    float(speed_logits.item()),
                    speed_max=self.artifact.speed_output_max,
                )
            except ValueError as exc:
                raise PolicyRuntimeError(str(exc)) from exc
        return angle_logits, speed_logits
