"""PyTorch datasets for temporal signal and shortcut policies."""

from __future__ import annotations

import bisect
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from xycar_ai.competition_data import (
    MissionSession,
    discover_sessions,
    load_split_manifest,
    materialize_shortcut_labels,
    materialize_signal_labels,
)
from xycar_ai.competition_models import SIGNAL_STATUS_NAMES


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


@dataclass(frozen=True)
class SequenceWindow:
    session_index: int
    sample_positions: tuple[int, ...]


def select_split_sessions(
    dataset_root: str | Path,
    split_manifest: str | Path,
    split: str,
    *,
    capture_kind: str | None = None,
) -> tuple[MissionSession, ...]:
    manifests = load_split_manifest(split_manifest)
    wanted = set(manifests[split])
    all_sessions = discover_sessions(dataset_root, require_approved=True)
    by_id = {session.session_id: session for session in all_sessions}
    missing = sorted(wanted.difference(by_id))
    if missing:
        raise ValueError(f"split references missing sessions: {missing}")
    sessions = tuple(
        by_id[session_id]
        for session_id in manifests[split]
        if capture_kind is None or by_id[session_id].capture_kind == capture_kind
    )
    if not sessions:
        raise ValueError(f"{split} has no {capture_kind or 'mission'} sessions")
    return sessions


class SignalSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        sessions: Sequence[MissionSession],
        *,
        sequence_length: int,
        frame_stride: int,
        window_step: int,
        input_height: int,
        input_width: int,
        augment: bool,
    ) -> None:
        if min(sequence_length, frame_stride, window_step) < 1:
            raise ValueError("sequence dimensions must be positive")
        self.sessions = tuple(sessions)
        self.sequence_length = sequence_length
        self.frame_stride = frame_stride
        self.input_height = input_height
        self.input_width = input_width
        self.augment = augment
        self.labels = tuple(
            materialize_signal_labels(session) for session in self.sessions
        )
        self.windows: list[SequenceWindow] = []
        extent = (sequence_length - 1) * frame_stride + 1
        for session_index, session in enumerate(self.sessions):
            last_start = max(0, len(session.samples) - extent)
            starts = list(range(0, last_start + 1, window_step))
            if not starts or starts[-1] != last_start:
                starts.append(last_start)
            for start in starts:
                positions = tuple(
                    min(len(session.samples) - 1, start + step * frame_stride)
                    for step in range(sequence_length)
                )
                self.windows.append(SequenceWindow(session_index, positions))

    def __len__(self) -> int:
        return len(self.windows)

    def sampling_weights(self) -> torch.Tensor:
        """Give each mission state equal sampling mass, regardless of duration."""
        categories = [self._window_category(window) for window in self.windows]
        counts = Counter(categories)
        return torch.tensor(
            [1.0 / counts[category] for category in categories],
            dtype=torch.double,
        )

    def sampling_category_counts(self) -> dict[str, int]:
        return dict(
            sorted(Counter(self._window_category(window) for window in self.windows).items())
        )

    def _window_category(self, window: SequenceWindow) -> str:
        labels = self.labels[window.session_index]
        selected = [labels[position] for position in window.sample_positions]
        # Left has priority over a simultaneously illuminated straight lamp,
        # matching the competition FSM. Red and yellow stay separate so each
        # stop appearance receives its own sampling mass.
        priorities = (
            ("red", "red"),
            ("yellow", "yellow"),
            ("left", "left_arrow"),
            ("green", "straight_green"),
        )
        for category, label_name in priorities:
            if any(label[label_name] > 0.5 for label in selected):
                return category
        if any(label["approach"] > 0.5 for label in selected):
            return "approach_unknown"
        return "background"

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        session = self.sessions[window.session_index]
        augment_seed = random.randrange(0, 2**31) if self.augment else None
        images = torch.stack(
            [
                _load_signal_image(
                    session.samples[position].image_path,
                    height=self.input_height,
                    width=self.input_width,
                    augment_seed=augment_seed,
                )
                for position in window.sample_positions
            ]
        )
        raw_labels = [
            self.labels[window.session_index][position]
            for position in window.sample_positions
        ]
        status = torch.tensor(
            [
                [float(label[name]) for name in SIGNAL_STATUS_NAMES]
                for label in raw_labels
            ],
            dtype=torch.float32,
        )
        bbox_values: list[tuple[float, float, float, float]] = []
        for label in raw_labels:
            x1, y1, x2, y2 = label["bbox"]
            if label["bbox_valid"]:
                crop_fraction = 2.0 / 3.0
                if y2 > crop_fraction + 1e-6:
                    raise ValueError(
                        f"signal bbox leaves upper 2/3 ROI in {session.session_id}"
                    )
                y1 /= crop_fraction
                y2 /= crop_fraction
            bbox_values.append((x1, y1, x2, y2))
        return {
            "images": images,
            "status": status,
            "bbox": torch.tensor(bbox_values, dtype=torch.float32),
            "bbox_valid": torch.tensor(
                [float(label["bbox_valid"]) for label in raw_labels],
                dtype=torch.float32,
            ),
            "progress": torch.tensor(
                [float(label["progress"]) for label in raw_labels],
                dtype=torch.float32,
            ),
        }


class ShortcutSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        sessions: Sequence[MissionSession],
        *,
        sequence_length: int,
        horizon_steps: int,
        sample_rate_hz: float,
        window_step: int,
        image_size: int,
        augment: bool,
    ) -> None:
        if min(sequence_length, horizon_steps, window_step) < 1:
            raise ValueError("shortcut sequence dimensions must be positive")
        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        self.sessions = tuple(sessions)
        self.sequence_length = sequence_length
        self.horizon_steps = horizon_steps
        self.sample_rate_hz = sample_rate_hz
        self.image_size = image_size
        self.augment = augment
        self.labels = tuple(
            materialize_shortcut_labels(session) for session in self.sessions
        )
        self.resampled_positions: list[tuple[int, ...]] = []
        self.windows: list[SequenceWindow] = []
        for session_index, session in enumerate(self.sessions):
            positions = _resample_positions(session, sample_rate_hz)
            positions = tuple(
                position
                for position in positions
                if self.labels[session_index][position]["active"] > 0.5
            )
            if not positions:
                continue
            minimum_positions = sequence_length + horizon_steps
            if len(positions) < minimum_positions:
                raise ValueError(
                    f"shortcut session {session.session_id} has "
                    f"{len(positions)} active resampled frames; "
                    f"at least {minimum_positions} are required"
                )
            self.resampled_positions.append(positions)
            last_start = max(0, len(positions) - sequence_length)
            starts = list(range(0, last_start + 1, window_step))
            if not starts or starts[-1] != last_start:
                starts.append(last_start)
            for start in starts:
                selected = tuple(
                    positions[min(len(positions) - 1, start + offset)]
                    for offset in range(sequence_length)
                )
                self.windows.append(SequenceWindow(session_index, selected))
        if not self.windows:
            raise ValueError("shortcut dataset has no active sequence windows")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        session = self.sessions[window.session_index]
        augment_seed = random.randrange(0, 2**31) if self.augment else None
        images = torch.stack(
            [
                _load_full_image(
                    session.samples[position].image_path,
                    image_size=self.image_size,
                    augment_seed=augment_seed,
                )
                for position in window.sample_positions
            ]
        )
        previous_commands: list[tuple[float, float]] = []
        angle_targets: list[list[int]] = []
        speed_targets: list[list[int]] = []
        positions = tuple(
            position
            for position in _resample_positions(session, self.sample_rate_hz)
            if self.labels[window.session_index][position]["active"] > 0.5
        )
        for position in window.sample_positions:
            location = bisect.bisect_left(positions, position)
            location = min(len(positions) - 1, location)
            previous_position = positions[max(0, location - 1)]
            previous = session.samples[previous_position]
            previous_commands.append((previous.angle, previous.speed))
            current_targets: list[Any] = []
            for horizon in range(self.horizon_steps):
                target_location = min(len(positions) - 1, location + horizon)
                current_targets.append(session.samples[positions[target_location]])
            angle_targets.append(
                [_command_class_id(sample.angle) for sample in current_targets]
            )
            speed_targets.append(
                [_command_class_id(sample.speed) for sample in current_targets]
            )
        shortcut_labels = [
            self.labels[window.session_index][position]
            for position in window.sample_positions
        ]
        return {
            "images": images,
            "previous_commands": torch.tensor(
                previous_commands,
                dtype=torch.float32,
            ),
            "angle_targets": torch.tensor(angle_targets, dtype=torch.long),
            "speed_targets": torch.tensor(speed_targets, dtype=torch.long),
            "phase_targets": torch.tensor(
                [int(label["phase"]) for label in shortcut_labels],
                dtype=torch.long,
            ),
            "handoff_targets": torch.tensor(
                [float(label["handoff_ready"]) for label in shortcut_labels],
                dtype=torch.float32,
            ),
        }


def _resample_positions(
    session: MissionSession,
    sample_rate_hz: float,
) -> tuple[int, ...]:
    timestamps = [sample.timestamp_sec for sample in session.samples]
    if len(timestamps) == 1:
        return (0,)
    interval = 1.0 / sample_rate_hz
    target = timestamps[0]
    end = timestamps[-1]
    positions: list[int] = []
    cursor = 0
    while target <= end + interval * 0.25:
        while (
            cursor + 1 < len(timestamps)
            and abs(timestamps[cursor + 1] - target)
            <= abs(timestamps[cursor] - target)
        ):
            cursor += 1
        if not positions or positions[-1] != cursor:
            positions.append(cursor)
        target += interval
    if positions[-1] != len(timestamps) - 1:
        positions.append(len(timestamps) - 1)
    return tuple(positions)


def _command_class_id(value: float) -> int:
    return int(round(max(-100.0, min(100.0, value)))) + 100


def _load_signal_image(
    path: Path,
    *,
    height: int,
    width: int,
    augment_seed: int | None,
) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
    crop_height = max(1, round(image.height * 2.0 / 3.0))
    image = image.crop((0, 0, image.width, crop_height))
    image = _augment(image, augment_seed)
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    return _to_normalized_tensor(image)


def _load_full_image(
    path: Path,
    *,
    image_size: int,
    augment_seed: int | None,
) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image = _augment(image, augment_seed)
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return _to_normalized_tensor(image)


def _augment(image: Image.Image, seed: int | None) -> Image.Image:
    if seed is None:
        return image
    generator = random.Random(seed)
    image = ImageEnhance.Brightness(image).enhance(generator.uniform(0.75, 1.25))
    image = ImageEnhance.Contrast(image).enhance(generator.uniform(0.8, 1.2))
    image = ImageEnhance.Color(image).enhance(generator.uniform(0.8, 1.2))
    if generator.random() < 0.2:
        image = image.filter(ImageFilter.GaussianBlur(generator.uniform(0.0, 1.2)))
    return image


def _to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD
