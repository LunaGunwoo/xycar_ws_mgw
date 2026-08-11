"""TorchScript runtime for the front-camera driving policy."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from xycar_ai_drive.artifact import RoadWarpParameters, load_policy_artifact
from xycar_ai_drive.control import DriveCommand, decode_class_ids


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
    ) -> None:
        if torch_num_threads < 1:
            raise ValueError('torch_num_threads must be positive')
        if warmup_count < 0:
            raise ValueError('warmup_count must be non-negative')
        self.artifact = load_policy_artifact(artifact_dir)
        try:
            import torch
        except ImportError as exc:
            raise PolicyRuntimeError(
                'PyTorch is unavailable in the vehicle Python environment'
            ) from exc
        self._torch = torch
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
                self._validate_outputs(self._model(sample))

    def infer(self, rgb_frame: np.ndarray) -> InferenceResult:
        chw = preprocess_rgb_frame(
            rgb_frame,
            image_size=self.artifact.image_size,
            mean=self._mean,
            std=self._std,
            road_warp=self.artifact.road_warp,
        )
        tensor = self._torch.from_numpy(chw).unsqueeze(0)
        started = time.perf_counter()
        try:
            with self._torch.inference_mode():
                outputs = self._model(tensor)
        except Exception as exc:
            raise PolicyRuntimeError(f'TorchScript inference failed: {exc}') from exc
        inference_ms = (time.perf_counter() - started) * 1000.0
        angle_logits, speed_logits = self._validate_outputs(outputs)
        angle_class_id = int(self._torch.argmax(angle_logits, dim=1).item())
        speed_class_id = int(self._torch.argmax(speed_logits, dim=1).item())
        return InferenceResult(
            command=decode_class_ids(angle_class_id, speed_class_id),
            inference_ms=inference_ms,
        )

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
        raise PolicyRuntimeError('camera frame must be a non-empty uint8 RGB image')
    if image_size < 1:
        raise PolicyRuntimeError('image_size must be positive')
    if road_warp is not None and (
        rgb_frame.shape[0] < 2 or rgb_frame.shape[1] < 2
    ):
        raise PolicyRuntimeError('road warp requires a frame of at least 2x2')
    if mean.shape != (1, 1, 3) or std.shape != (1, 1, 3):
        raise PolicyRuntimeError('normalization arrays must have shape [1,1,3]')
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
    height, width = rgb_frame.shape[:2]
    max_x = float(width - 1)
    max_y = float(height - 1)
    source = np.asarray(
        [
            [config.bottom_left_x * max_x, config.bottom_y * max_y],
            [config.top_left_x * max_x, config.top_y * max_y],
            [config.top_right_x * max_x, config.top_y * max_y],
            [config.bottom_right_x * max_x, config.bottom_y * max_y],
        ],
        dtype=np.float32,
    )
    output_max_x = float(config.bev_width - 1)
    output_max_y = float(config.bev_height - 1)
    left = config.dst_left_x * output_max_x
    right = config.dst_right_x * output_max_x
    destination = np.asarray(
        [
            [left, output_max_y],
            [left, 0.0],
            [right, 0.0],
            [right, output_max_y],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        rgb_frame,
        transform,
        (config.bev_width, config.bev_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
