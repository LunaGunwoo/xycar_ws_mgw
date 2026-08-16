"""Preloaded CUDA runtime for base, signal, and shortcut policies."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from xycar_ai_drive.competition_artifact import (
    CompetitionBundle,
    ImageContract,
    load_competition_bundle,
)
from xycar_ai_drive.control import DriveCommand, decode_class_ids
from xycar_ai_drive.policy_runtime import PolicyRuntimeError, preprocess_rgb_frame


INFERENCE_MODES = {
    "signal_only",
    "normal",
    "signal_stop",
    "shortcut",
    "handoff_verify",
}


@dataclass(frozen=True)
class SignalObservation:
    approach: float
    visible: float
    readable: float
    red: float
    yellow: float
    left: float
    green: float
    bbox: tuple[float, float, float, float]
    progress: float


@dataclass(frozen=True)
class ShortcutObservation:
    command: DriveCommand
    phase: int
    handoff_probability: float


@dataclass(frozen=True)
class CompetitionInference:
    base_command: DriveCommand | None
    base_confidence: float | None
    signal: SignalObservation | None
    shortcut: ShortcutObservation | None
    inference_ms: float


class CompetitionGpuRuntime:
    """Keep every race model resident and run only the requested branches."""

    def __init__(
        self,
        *,
        artifact_dir: str,
        device: str,
        torch_num_threads: int,
        warmup_count: int,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if torch_num_threads < 1 or warmup_count < 0:
            raise ValueError("invalid runtime warmup/thread count")
        try:
            import torch
        except ImportError as exc:
            raise PolicyRuntimeError("PyTorch is unavailable") from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise PolicyRuntimeError("CUDA was requested but is unavailable")
        self.artifact: CompetitionBundle = load_competition_bundle(artifact_dir)
        self.device_name = device
        self._torch = torch
        self._device = torch.device(device)
        self._lock = threading.RLock()
        if device == "cpu":
            torch.set_num_threads(torch_num_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        try:
            self._base_model = torch.jit.load(
                str(self.artifact.base.model_path),
                map_location=self._device,
            ).eval()
            self._signal_model = torch.jit.load(
                str(self.artifact.signal.model_path),
                map_location=self._device,
            ).eval()
            self._shortcut_model = torch.jit.load(
                str(self.artifact.shortcut.model_path),
                map_location=self._device,
            ).eval()
        except Exception as exc:
            raise PolicyRuntimeError(
                f"could not preload competition models on {device}: {exc}"
            ) from exc
        self._base_mean = _normalization_array(self.artifact.base.mean)
        self._base_std = _normalization_array(self.artifact.base.std)
        self._signal_mean = _normalization_array(self.artifact.signal.mean)
        self._signal_std = _normalization_array(self.artifact.signal.std)
        self._shortcut_mean = _normalization_array(self.artifact.shortcut.mean)
        self._shortcut_std = _normalization_array(self.artifact.shortcut.std)
        self.reset_all()
        self._warmup(warmup_count)

    def reset_all(self) -> None:
        with self._lock:
            self._signal_hidden = self._torch.zeros(
                1,
                1,
                self.artifact.signal.hidden_size,
                dtype=self._torch.float32,
                device=self._device,
            )
            self._shortcut_hidden = self._torch.zeros(
                1,
                1,
                self.artifact.shortcut.hidden_size,
                dtype=self._torch.float32,
                device=self._device,
            )

    def reset_shortcut(self) -> None:
        with self._lock:
            self._shortcut_hidden.zero_()

    def infer(
        self,
        rgb_frame: np.ndarray,
        *,
        mode: str,
        previous_command: DriveCommand,
    ) -> CompetitionInference:
        if mode not in INFERENCE_MODES:
            raise PolicyRuntimeError(f"unsupported competition mode: {mode}")
        if not all(
            math.isfinite(value)
            for value in (previous_command.angle, previous_command.speed)
        ):
            raise PolicyRuntimeError("previous command must be finite")
        _validate_rgb_frame(rgb_frame)
        with self._lock:
            try:
                self._synchronize()
                started = time.perf_counter()
                base_command = None
                base_confidence = None
                signal = None
                shortcut = None
                with self._torch.inference_mode():
                    if mode in {"normal", "signal_stop", "handoff_verify"}:
                        base_command, base_confidence = self._infer_base(rgb_frame)
                    if mode in {"signal_only", "normal", "signal_stop"}:
                        signal = self._infer_signal(rgb_frame)
                    if mode in {"shortcut", "handoff_verify"}:
                        shortcut = self._infer_shortcut(
                            rgb_frame,
                            previous_command,
                        )
                self._synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
            except PolicyRuntimeError:
                self.reset_all()
                raise
            except Exception as exc:
                self.reset_all()
                raise PolicyRuntimeError(
                    f"competition inference failed on {self.device_name}: {exc}"
                ) from exc
        return CompetitionInference(
            base_command=base_command,
            base_confidence=base_confidence,
            signal=signal,
            shortcut=shortcut,
            inference_ms=elapsed_ms,
        )

    def _infer_base(self, frame: np.ndarray) -> tuple[DriveCommand, float]:
        contract = self.artifact.base
        chw = preprocess_rgb_frame(
            frame,
            image_size=contract.width,
            mean=self._base_mean,
            std=self._base_std,
            road_warp=contract.road_warp,
        )
        tensor = self._tensor(chw)
        outputs = self._base_model(tensor)
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise PolicyRuntimeError("base model returned an invalid tuple")
        angle_logits, speed_logits = outputs
        _validate_logits(angle_logits, (1, 201), "base angle", self._torch)
        _validate_logits(speed_logits, (1, 201), "base speed", self._torch)
        angle_id = int(self._torch.argmax(angle_logits, dim=1).item())
        speed_id = int(self._torch.argmax(speed_logits, dim=1).item())
        command = decode_class_ids(angle_id, speed_id)
        command = DriveCommand(
            angle=command.angle,
            speed=min(command.speed, self.artifact.mission.maximum_forward_speed),
        )
        angle_confidence = float(
            self._torch.softmax(angle_logits, dim=1).max().item()
        )
        speed_confidence = float(
            self._torch.softmax(speed_logits, dim=1).max().item()
        )
        return command, min(angle_confidence, speed_confidence)

    def _infer_signal(self, frame: np.ndarray) -> SignalObservation:
        chw = _preprocess_temporal_frame(
            frame,
            self.artifact.signal,
            self._signal_mean,
            self._signal_std,
            upper_two_thirds=True,
        )
        outputs = self._signal_model(self._tensor(chw), self._signal_hidden)
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
            raise PolicyRuntimeError("signal model returned an invalid tuple")
        status_logits, bbox, progress, next_hidden = outputs
        _validate_logits(status_logits, (1, 7), "signal status", self._torch)
        _validate_logits(bbox, (1, 4), "signal bbox", self._torch)
        _validate_logits(progress, (1,), "signal progress", self._torch)
        _validate_logits(
            next_hidden,
            (1, 1, self.artifact.signal.hidden_size),
            "signal hidden",
            self._torch,
        )
        self._signal_hidden = next_hidden
        probabilities = self._torch.sigmoid(status_logits)[0].tolist()
        bbox_values = tuple(float(value) for value in bbox[0].tolist())
        return SignalObservation(
            approach=float(probabilities[0]),
            visible=float(probabilities[1]),
            readable=float(probabilities[2]),
            red=float(probabilities[3]),
            yellow=float(probabilities[4]),
            left=float(probabilities[5]),
            green=float(probabilities[6]),
            bbox=bbox_values,
            progress=float(progress[0].item()),
        )

    def _infer_shortcut(
        self,
        frame: np.ndarray,
        previous_command: DriveCommand,
    ) -> ShortcutObservation:
        chw = _preprocess_temporal_frame(
            frame,
            self.artifact.shortcut,
            self._shortcut_mean,
            self._shortcut_std,
            upper_two_thirds=False,
        )
        command_tensor = self._torch.tensor(
            [[previous_command.angle, previous_command.speed]],
            dtype=self._torch.float32,
            device=self._device,
        )
        outputs = self._shortcut_model(
            self._tensor(chw),
            command_tensor,
            self._shortcut_hidden,
        )
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
            raise PolicyRuntimeError("shortcut model returned an invalid tuple")
        angle_logits, speed_logits, phase_logits, handoff_logits, next_hidden = (
            outputs
        )
        horizon = self.artifact.shortcut.horizon_steps
        _validate_logits(
            angle_logits,
            (1, horizon, 201),
            "shortcut angle",
            self._torch,
        )
        _validate_logits(
            speed_logits,
            (1, horizon, 201),
            "shortcut speed",
            self._torch,
        )
        _validate_logits(phase_logits, (1, 6), "shortcut phase", self._torch)
        _validate_logits(handoff_logits, (1,), "shortcut handoff", self._torch)
        _validate_logits(
            next_hidden,
            (1, 1, self.artifact.shortcut.hidden_size),
            "shortcut hidden",
            self._torch,
        )
        self._shortcut_hidden = next_hidden
        angle_id = int(self._torch.argmax(angle_logits[:, 0], dim=1).item())
        speed_id = int(self._torch.argmax(speed_logits[:, 0], dim=1).item())
        command = decode_class_ids(angle_id, speed_id)
        command = DriveCommand(
            angle=command.angle,
            speed=min(command.speed, self.artifact.mission.maximum_forward_speed),
        )
        return ShortcutObservation(
            command=command,
            phase=int(self._torch.argmax(phase_logits, dim=1).item()),
            handoff_probability=float(
                self._torch.sigmoid(handoff_logits)[0].item()
            ),
        )

    def _tensor(self, chw: np.ndarray):
        return self._torch.from_numpy(chw).unsqueeze(0).to(self._device)

    def _warmup(self, count: int) -> None:
        base = self._torch.zeros(
            1,
            3,
            self.artifact.base.height,
            self.artifact.base.width,
            device=self._device,
        )
        signal = self._torch.zeros(
            1,
            3,
            self.artifact.signal.height,
            self.artifact.signal.width,
            device=self._device,
        )
        shortcut = self._torch.zeros(
            1,
            3,
            self.artifact.shortcut.height,
            self.artifact.shortcut.width,
            device=self._device,
        )
        previous = self._torch.zeros(1, 2, device=self._device)
        with self._torch.inference_mode():
            for _ in range(count):
                base_outputs = self._base_model(base)
                signal_outputs = self._signal_model(signal, self._signal_hidden)
                shortcut_outputs = self._shortcut_model(
                    shortcut,
                    previous,
                    self._shortcut_hidden,
                )
                if len(base_outputs) != 2 or len(signal_outputs) != 4:
                    raise PolicyRuntimeError("warmup model output mismatch")
                if len(shortcut_outputs) != 5:
                    raise PolicyRuntimeError("shortcut warmup output mismatch")
        self._synchronize()
        self.reset_all()

    def _synchronize(self) -> None:
        if self.device_name == "cuda":
            self._torch.cuda.synchronize(self._device)


def _preprocess_temporal_frame(
    frame: np.ndarray,
    contract: ImageContract,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    upper_two_thirds: bool,
) -> np.ndarray:
    _validate_rgb_frame(frame)
    geometry = frame[: max(1, round(frame.shape[0] * 2.0 / 3.0))]
    if not upper_two_thirds:
        geometry = frame
    resized = cv2.resize(
        geometry,
        (contract.width, contract.height),
        interpolation=cv2.INTER_CUBIC,
    )
    normalized = (resized.astype(np.float32) / 255.0 - mean) / std
    chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    if not np.isfinite(chw).all():
        raise PolicyRuntimeError("temporal preprocessing produced non-finite data")
    return chw


def _validate_rgb_frame(frame: np.ndarray) -> None:
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[0] < 2
        or frame.shape[1] < 2
    ):
        raise PolicyRuntimeError("camera frame must be a non-empty uint8 RGB image")


def _normalization_array(values: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(1, 1, 3)


def _validate_logits(tensor, shape: tuple[int, ...], label: str, torch) -> None:
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
        raise PolicyRuntimeError(f"{label} shape must be {shape}")
    if not bool(torch.isfinite(tensor).all()):
        raise PolicyRuntimeError(f"{label} contains non-finite values")
