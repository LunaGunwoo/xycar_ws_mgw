"""TorchScript runtime for the front-camera driving policy."""

from __future__ import annotations

import time
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from xycar_ai_drive.artifact import (
    ANGLE_REGRESSION_FIXED_SPEED_PREDICTION_MODE,
    CATEGORICAL_PREDICTION_MODE,
    COMPACT_CONTROL_ENCODING,
    CONTINUOUS_REGRESSION_PREDICTION_MODE,
    RoadWarpParameters,
    load_policy_artifact,
)
from xycar_ai_drive.control import (
    DriveCommand,
    decode_class_ids,
    decode_compact_output_ids,
    decode_regression_outputs,
)
from xycar_ai_drive.road_warp import warp_road_image


class PolicyRuntimeError(RuntimeError):
    """Raised when preprocessing or TorchScript inference is unsafe."""


@dataclass(frozen=True)
class InferenceResult:
    command: DriveCommand
    inference_ms: float


class TorchScriptPolicy:
    def __init__(
        self,
        *,
        artifact_dir: str,
        torch_num_threads: int,
        warmup_count: int,
        history_reset_timeout_sec: float = 0.25,
    ) -> None:
        if torch_num_threads < 1:
            raise ValueError('torch_num_threads must be positive')
        if warmup_count < 0:
            raise ValueError('warmup_count must be non-negative')
        if (
            not np.isfinite(history_reset_timeout_sec)
            or history_reset_timeout_sec <= 0
        ):
            raise ValueError(
                'history_reset_timeout_sec must be finite and positive'
            )
        self.artifact = load_policy_artifact(artifact_dir)
        try:
            import torch
        except ImportError as exc:
            raise PolicyRuntimeError(
                'PyTorch is unavailable in the vehicle Python environment'
            ) from exc
        self._torch = torch
        self._history_lock = threading.RLock()
        self._history_reset_timeout_sec = float(history_reset_timeout_sec)
        self._last_successful_inference_monotonic: float | None = None
        self._history_class_ids: list[list[int]] | None = None
        self._reset_history_locked()
        torch.set_num_threads(torch_num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        try:
            self._model = torch.jit.load(
                str(self.artifact.model_path),
                map_location='cpu',
            ).eval()
        except Exception as exc:
            raise PolicyRuntimeError(
                f'could not load TorchScript model: {exc}'
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
        )
        with torch.inference_mode():
            for _ in range(warmup_count):
                if (
                    self.artifact.prediction_mode
                    == ANGLE_REGRESSION_FIXED_SPEED_PREDICTION_MODE
                ):
                    outputs = self._model(sample, self._fixed_speed_tensor())
                elif self._history_class_ids is None:
                    outputs = self._model(sample)
                else:
                    history = torch.tensor(
                        [self._history_class_ids],
                        dtype=torch.long,
                    )
                    outputs = self._model(sample, history)
                self._validate_outputs(outputs)

    @property
    def history_class_ids(self) -> tuple[tuple[int, int], ...] | None:
        with self._history_lock:
            if self._history_class_ids is None:
                return None
            return tuple(tuple(pair) for pair in self._history_class_ids)

    def reset_history(self) -> None:
        with self._history_lock:
            self._reset_history_locked()

    def _reset_history_locked(self) -> None:
        history = getattr(self, 'artifact', None)
        history = history.history if history is not None else None
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
        chw = preprocess_rgb_frame(
            rgb_frame,
            image_size=self.artifact.image_size,
            mean=self._mean,
            std=self._std,
            road_warp=self.artifact.road_warp,
        )
        tensor = self._torch.from_numpy(chw).unsqueeze(0)
        now = time.monotonic()
        try:
            with self._history_lock:
                if (
                    self._last_successful_inference_monotonic is not None
                    and now - self._last_successful_inference_monotonic
                    >= self._history_reset_timeout_sec
                ):
                    self._reset_history_locked()
                started = time.perf_counter()
                with self._torch.inference_mode():
                    inference_history = self._inference_history(
                        history_class_ids
                    )
                    if (
                        self.artifact.prediction_mode
                        == ANGLE_REGRESSION_FIXED_SPEED_PREDICTION_MODE
                    ):
                        outputs = self._model(
                            tensor,
                            self._fixed_speed_tensor(),
                        )
                    elif inference_history is None:
                        outputs = self._model(tensor)
                    else:
                        history = self._torch.tensor(
                            [inference_history],
                            dtype=self._torch.long,
                        )
                        outputs = self._model(tensor, history)
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
                f'TorchScript inference failed: {exc}'
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

    def _inference_history(
        self,
        supplied: Sequence[Sequence[int]] | None,
    ) -> list[list[int]] | None:
        history = self.artifact.history
        if history is None:
            if supplied is not None:
                raise PolicyRuntimeError(
                    'stateless policy does not accept executed history'
                )
            return None
        if history.update == 'predicted_argmax':
            if supplied is not None:
                raise PolicyRuntimeError(
                    'schema v2 policy owns its predicted history'
                )
            return self._history_class_ids
        if supplied is None:
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

    def _fixed_speed_tensor(self):
        fixed_speed = self.artifact.fixed_speed
        divisor = self.artifact.speed_normalization_divisor
        if fixed_speed is None or divisor is None:
            raise PolicyRuntimeError(
                'fixed-speed artifact runtime values are missing'
            )
        return self._torch.tensor(
            [[fixed_speed / divisor]],
            dtype=self._torch.float32,
        )

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
                )
            except ValueError as exc:
                raise PolicyRuntimeError(str(exc)) from exc
        return angle_logits, speed_logits


def preprocess_rgb_frame(
    rgb_frame: np.ndarray,
    *,
    image_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    road_warp: RoadWarpParameters | None = None,
) -> np.ndarray:
    if (
        not isinstance(rgb_frame, np.ndarray)
        or rgb_frame.dtype != np.uint8
        or rgb_frame.ndim != 3
        or rgb_frame.shape[2] != 3
        or rgb_frame.shape[0] < 1
        or rgb_frame.shape[1] < 1
    ):
        raise PolicyRuntimeError(
            'camera frame must be a non-empty uint8 RGB image'
        )
    if image_size < 1:
        raise PolicyRuntimeError('image_size must be positive')
    if road_warp is not None and (
        rgb_frame.shape[0] < 2 or rgb_frame.shape[1] < 2
    ):
        raise PolicyRuntimeError('road warp requires a frame of at least 2x2')
    if mean.shape != (1, 1, 3) or std.shape != (1, 1, 3):
        raise PolicyRuntimeError(
            'normalization arrays must have shape [1,1,3]'
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise PolicyRuntimeError('normalization arrays must be finite')
    if np.any(std <= 0.0):
        raise PolicyRuntimeError('normalization std must be positive')
    geometry_image = (
        _warp_road_image(rgb_frame, road_warp)
        if road_warp is not None
        else rgb_frame
    )
    resized = cv2.resize(
        geometry_image,
        (image_size, image_size),
        interpolation=cv2.INTER_CUBIC,
    )
    normalized = (resized.astype(np.float32) / 255.0 - mean) / std
    chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    if chw.shape != (3, image_size, image_size) or not np.isfinite(chw).all():
        raise PolicyRuntimeError('preprocessed image is invalid')
    return chw


def _warp_road_image(
    rgb_frame: np.ndarray,
    config: RoadWarpParameters,
) -> np.ndarray:
    try:
        return warp_road_image(rgb_frame, config)
    except ValueError as exc:
        raise PolicyRuntimeError(f'invalid road warp: {exc}') from exc
