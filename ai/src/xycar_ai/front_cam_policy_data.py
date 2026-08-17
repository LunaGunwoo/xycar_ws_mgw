from __future__ import annotations

import csv
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from xycar_ai.config import AugmentationConfig, DataConfig, DataSourceConfig
from xycar_ai.front_cam_policy_warp import RoadWarpConfig, warp_pil_image
from xycar_ai.steering_contract import metadata_has_required_steering_contract

COMMAND_MIN = -100
COMMAND_MAX = 100
COMMAND_OFFSET = 100
NUM_COMMAND_CLASSES = 201
SPLIT_NAMES = ("train", "val", "test")
SESSION_NAME_RE = re.compile(r"^\d{8}_\d{6}_\d{3}_session(?:_\d+)?$")
REQUIRED_CSV_FIELDS = {
    "sample_index",
    "image",
    "angle",
    "speed",
    "input_key",
    "camera_sequence",
    "camera_stamp_sec",
    "camera_stamp_nanosec",
    "camera_received_wall_time_ns",
}


class PolicyDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class PolicySample:
    session_id: str
    image_path: Path
    relative_image: str
    angle_raw: float
    speed_raw: float
    angle: int
    speed: int
    angle_class_id: int
    speed_class_id: int
    history_class_ids: tuple[tuple[int, int], ...] | None = None
    generation: int = 0
    source_id: str | None = None


@dataclass(frozen=True)
class PolicySession:
    session_id: str
    path: Path
    metadata: Mapping[str, object]
    samples: tuple[PolicySample, ...]
    generation: int = 0
    source_id: str | None = None


@dataclass(frozen=True)
class PolicyDataSplits:
    dataset_snapshot: str
    train_sessions: tuple[PolicySession, ...]
    val_sessions: tuple[PolicySession, ...]
    test_sessions: tuple[PolicySession, ...]

    @property
    def train_samples(self) -> tuple[PolicySample, ...]:
        return _flatten_samples(self.train_sessions)

    @property
    def val_samples(self) -> tuple[PolicySample, ...]:
        return _flatten_samples(self.val_sessions)

    @property
    def test_samples(self) -> tuple[PolicySample, ...]:
        return _flatten_samples(self.test_sessions)

    def manifest(self, *, include_generation: bool = False) -> dict[str, object]:
        split_payload: dict[str, object] = {}
        for name, sessions in self._session_groups().items():
            details: dict[str, object] = {
                "sessions": [session.session_id for session in sessions],
                "session_count": len(sessions),
                "sample_count": sum(len(session.samples) for session in sessions),
            }
            if include_generation:
                details.update(
                    {
                        "generation_session_counts": _counter_dict(
                            Counter(session.generation for session in sessions)
                        ),
                        "generation_sample_counts": _counter_dict(
                            Counter(
                                sample.generation
                                for session in sessions
                                for sample in session.samples
                            )
                        ),
                    }
                )
            split_payload[name] = details
        return {
            "schema_version": (
                2
                if any(
                    session.source_id is not None
                    for sessions in self._session_groups().values()
                    for session in sessions
                )
                else 1
            ),
            "dataset_snapshot": self.dataset_snapshot,
            "splits": split_payload,
        }

    def _session_groups(self) -> dict[str, tuple[PolicySession, ...]]:
        return {
            "train": self.train_sessions,
            "val": self.val_sessions,
            "test": self.test_sessions,
        }


def smooth_training_angle_targets(
    splits: PolicyDataSplits,
    window_size: int,
) -> PolicyDataSplits:
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("angle mean window must be a positive odd integer")
    if window_size == 1:
        return splits
    return replace(
        splits,
        train_sessions=tuple(
            _smooth_session_angle_targets(session, window_size)
            for session in splits.train_sessions
        ),
    )


def _smooth_session_angle_targets(
    session: PolicySession,
    window_size: int,
) -> PolicySession:
    radius = window_size // 2
    last_index = len(session.samples) - 1
    smoothed: list[PolicySample] = []
    for index, sample in enumerate(session.samples):
        angle_raw = (
            sum(
                session.samples[min(max(index + offset, 0), last_index)].angle_raw
                for offset in range(-radius, radius + 1)
            )
            / window_size
        )
        angle = quantize_command(angle_raw)
        smoothed.append(
            replace(
                sample,
                angle_raw=angle_raw,
                angle=angle,
                angle_class_id=angle + COMMAND_OFFSET,
            )
        )
    return replace(session, samples=tuple(smoothed))


def attach_training_teacher_forced_history(
    splits: PolicyDataSplits,
    history_frames: int,
) -> PolicyDataSplits:
    if history_frames <= 0:
        raise ValueError("history_frames must be positive")
    return replace(
        splits,
        train_sessions=tuple(
            _attach_session_teacher_forced_history(session, history_frames)
            for session in splits.train_sessions
        ),
    )


def attach_executed_command_history(
    splits: PolicyDataSplits,
    history_frames: int,
) -> PolicyDataSplits:
    """Attach the actual previously executed commands to every split."""
    if history_frames <= 0:
        raise ValueError("history_frames must be positive")
    groups = {
        name: tuple(
            _attach_session_executed_history(session, history_frames)
            for session in sessions
        )
        for name, sessions in splits._session_groups().items()
    }
    return replace(
        splits,
        train_sessions=groups["train"],
        val_sessions=groups["val"],
        test_sessions=groups["test"],
    )


def _attach_session_executed_history(
    session: PolicySession,
    history_frames: int,
) -> PolicySession:
    initial = _session_initial_history(session, history_frames)
    history = list(initial)
    samples: list[PolicySample] = []
    for sample in session.samples:
        samples.append(replace(sample, history_class_ids=tuple(history)))
        history = history[1:] + [(sample.angle_class_id, sample.speed_class_id)]
    return replace(session, samples=tuple(samples))


def _session_initial_history(
    session: PolicySession,
    history_frames: int,
) -> tuple[tuple[int, int], ...]:
    curriculum = session.metadata.get("curriculum")
    configured = (
        curriculum.get("initial_history_class_ids")
        if isinstance(curriculum, dict)
        else None
    )
    if configured is None:
        return ((100, 125),) * history_frames
    if not isinstance(configured, list) or len(configured) != history_frames:
        raise PolicyDatasetError(
            f"session {session.session_id} has invalid initial history length"
        )
    history: list[tuple[int, int]] = []
    for pair in configured:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in pair
            )
            or any(not 0 <= value < NUM_COMMAND_CLASSES for value in pair)
        ):
            raise PolicyDatasetError(
                f"session {session.session_id} has invalid initial history classes"
            )
        history.append((pair[0], pair[1]))
    return tuple(history)


def _attach_session_teacher_forced_history(
    session: PolicySession,
    history_frames: int,
) -> PolicySession:
    samples: list[PolicySample] = []
    for index, sample in enumerate(session.samples):
        history = tuple(
            (
                session.samples[max(index - history_frames + offset, 0)].angle_class_id,
                session.samples[max(index - history_frames + offset, 0)].speed_class_id,
            )
            for offset in range(history_frames)
        )
        samples.append(replace(sample, history_class_ids=history))
    return replace(session, samples=tuple(samples))


def validate_session_initial_classes(
    splits: PolicyDataSplits,
    *,
    angle_class_id: int,
    speed_class_id: int,
) -> None:
    for split_name, sessions in splits._session_groups().items():
        for session in sessions:
            first = session.samples[0]
            actual = (first.angle_class_id, first.speed_class_id)
            expected = (angle_class_id, speed_class_id)
            if actual != expected:
                raise PolicyDatasetError(
                    f"{split_name} session {session.session_id} initial class "
                    f"pair differs from the history contract: {actual} != {expected}"
                )


def quantize_command(value: float) -> int:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise PolicyDatasetError(f"command must be finite, got {value!r}")
    clamped = max(float(COMMAND_MIN), min(float(COMMAND_MAX), numeric))
    return round(clamped)


def command_class_id(value: float) -> int:
    return quantize_command(value) + COMMAND_OFFSET


def discover_policy_sessions(config: DataConfig) -> tuple[PolicySession, ...]:
    sessions: list[PolicySession] = []
    if config.sources:
        for source in config.sources:
            sessions.extend(
                _discover_source_sessions(
                    source,
                    required_steering_contract=(
                        config.required_steering_contract
                    ),
                )
            )
        if not sessions:
            roots = ", ".join(str(source.root) for source in config.sources)
            raise PolicyDatasetError(
                f"no completed policy sessions match configured sources under {roots}"
            )
        return tuple(sessions)

    root = config.root
    if root is None or not root.is_dir():
        raise PolicyDatasetError(f"dataset root does not exist: {root}")
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not SESSION_NAME_RE.fullmatch(path.name):
            continue
        metadata = _load_metadata(path / "metadata.yaml")
        if not _matches_filter(metadata, config):
            continue
        sessions.append(
            _load_policy_session(
                path,
                root,
                metadata,
                legacy_generation=config.legacy_generation,
            )
        )

    if not sessions:
        raise PolicyDatasetError(
            "no completed policy sessions match "
            f"{_forward_speed_filter_description(config)} under {root}"
        )
    return tuple(sessions)


def _discover_source_sessions(
    source: DataSourceConfig,
    *,
    required_steering_contract: str | None,
) -> list[PolicySession]:
    if not source.root.is_dir():
        raise PolicyDatasetError(
            f"dataset source root does not exist: {source.source_id}={source.root}"
        )
    sessions: list[PolicySession] = []
    for path in sorted(source.root.iterdir()):
        if not path.is_dir() or not SESSION_NAME_RE.fullmatch(path.name):
            continue
        metadata = _load_metadata(path / "metadata.yaml")
        if not _matches_source_filter(
            metadata,
            source,
            required_steering_contract=required_steering_contract,
        ):
            continue
        sessions.append(
            _load_policy_session(
                path,
                source.root,
                metadata,
                source_id=source.source_id,
                fixed_generation=source.fixed_generation,
                require_curriculum_generation=(
                    source.require_curriculum_generation
                ),
            )
        )
    return sessions


def build_policy_data_splits(
    config: DataConfig,
) -> PolicyDataSplits:
    sessions = discover_policy_sessions(config)
    payload = _load_yaml_mapping(config.split_manifest)
    if set(payload) != {"schema_version", "dataset_snapshot", "splits"}:
        raise PolicyDatasetError(
            f"unexpected split manifest keys: {config.split_manifest}"
        )
    manifest_schema_version = _manifest_integer(payload, "schema_version")
    expected_schema_version = 2 if config.sources else 1
    if manifest_schema_version != expected_schema_version:
        raise PolicyDatasetError(
            "split manifest schema_version must be "
            f"{expected_schema_version} for this data configuration"
        )
    dataset_snapshot = _manifest_string(payload, "dataset_snapshot")
    split_payload = payload.get("splits")
    if not isinstance(split_payload, dict) or set(split_payload) != set(SPLIT_NAMES):
        raise PolicyDatasetError("split manifest must define train, val, and test")

    by_id = {session.session_id: session for session in sessions}
    listed_ids: list[str] = []
    split_sessions: dict[str, tuple[PolicySession, ...]] = {}
    for split_name in SPLIT_NAMES:
        values = split_payload.get(split_name)
        if not isinstance(values, list) or not values:
            raise PolicyDatasetError(f"split {split_name} must be a non-empty list")
        if not all(isinstance(value, str) and value for value in values):
            raise PolicyDatasetError(
                f"split {split_name} contains an invalid session id"
            )
        ids = [str(value) for value in values]
        listed_ids.extend(ids)
        missing = sorted(set(ids) - set(by_id))
        if missing:
            raise PolicyDatasetError(
                f"split {split_name} references missing or filtered sessions: {missing}"
            )
        split_sessions[split_name] = tuple(by_id[session_id] for session_id in ids)

    duplicates = _duplicates(listed_ids)
    if duplicates:
        raise PolicyDatasetError(
            f"sessions occur in more than one split: {sorted(duplicates)}"
        )
    if config.require_all_matching_sessions:
        unlisted = sorted(set(by_id) - set(listed_ids))
        if unlisted:
            raise PolicyDatasetError(
                f"matching sessions are absent from the split manifest: {unlisted}"
            )

    result = PolicyDataSplits(
        dataset_snapshot=dataset_snapshot,
        train_sessions=split_sessions["train"],
        val_sessions=split_sessions["val"],
        test_sessions=split_sessions["test"],
    )
    _validate_manual_anchor_split(result, config)
    _validate_current_generation_session_counts(result, config)
    _validate_minimum_dataset_size(result, config)
    if config.ema_sampling:
        future_generations = sorted(
            {
                sample.generation
                for sessions in result._session_groups().values()
                for session in sessions
                for sample in session.samples
                if sample.generation > config.current_generation
            }
        )
        if future_generations:
            raise PolicyDatasetError(
                "split contains generation(s) newer than current_generation: "
                f"{future_generations}"
            )
    return result


def _validate_manual_anchor_split(
    splits: PolicyDataSplits,
    config: DataConfig,
) -> None:
    path = config.manual_anchor_split_manifest
    if path is None:
        return
    payload = _load_yaml_mapping(path)
    if set(payload) != {"schema_version", "dataset_snapshot", "splits"}:
        raise PolicyDatasetError(f"unexpected Manual anchor manifest keys: {path}")
    if _manifest_integer(payload, "schema_version") != 1:
        raise PolicyDatasetError("Manual anchor split manifest must use schema_version 1")
    split_payload = payload.get("splits")
    if not isinstance(split_payload, dict) or set(split_payload) != set(SPLIT_NAMES):
        raise PolicyDatasetError("Manual anchor split must define train, val, and test")
    for split_name, sessions in splits._session_groups().items():
        raw_expected = split_payload.get(split_name)
        if not isinstance(raw_expected, list) or not all(
            isinstance(session_id, str) and session_id and "/" not in session_id
            for session_id in raw_expected
        ):
            raise PolicyDatasetError(
                f"Manual anchor {split_name} must contain plain session ids"
            )
        expected = {f"manual/{session_id}" for session_id in raw_expected}
        observed = {
            session.session_id
            for session in sessions
            if session.source_id == "manual"
        }
        if observed != expected:
            raise PolicyDatasetError(
                f"Manual anchor {split_name} sessions differ from {path}; "
                f"missing={sorted(expected - observed)}, "
                f"unexpected={sorted(observed - expected)}"
            )


def _validate_current_generation_session_counts(
    splits: PolicyDataSplits,
    config: DataConfig,
) -> None:
    if not config.current_generation_session_counts:
        return
    failures = []
    for split_name, sessions in splits._session_groups().items():
        actual = sum(
            session.source_id == "guided"
            and session.generation == config.current_generation
            for session in sessions
        )
        expected = config.current_generation_session_counts[split_name]
        if actual != expected:
            failures.append(f"{split_name} {actual} != {expected}")
    if failures:
        raise PolicyDatasetError(
            "current Guided generation session counts differ from config: "
            + "; ".join(failures)
        )


def _validate_minimum_dataset_size(
    splits: PolicyDataSplits,
    config: DataConfig,
) -> None:
    groups = splits._session_groups()
    all_sessions = tuple(
        session for sessions in groups.values() for session in sessions
    )
    requirements = {
        "total samples": (
            sum(len(session.samples) for session in all_sessions),
            config.minimum_total_samples,
        ),
        "total sessions": (
            len(all_sessions),
            config.minimum_total_sessions,
        ),
        "train sessions": (
            len(groups["train"]),
            config.minimum_train_sessions,
        ),
        "val sessions": (
            len(groups["val"]),
            config.minimum_val_sessions,
        ),
        "test sessions": (
            len(groups["test"]),
            config.minimum_test_sessions,
        ),
    }
    failures = [
        f"{label} {actual} < {minimum}"
        for label, (actual, minimum) in requirements.items()
        if actual < minimum
    ]
    if failures:
        raise PolicyDatasetError(
            "dataset is below configured minimum: " + "; ".join(failures)
        )


def policy_dataset_stats(
    splits: PolicyDataSplits, *, include_generation: bool = False
) -> dict[str, object]:
    groups = {
        "train": splits.train_samples,
        "val": splits.val_samples,
        "test": splits.test_samples,
    }
    all_samples = tuple(sample for samples in groups.values() for sample in samples)
    return {
        "schema_version": 1,
        "dataset_snapshot": splits.dataset_snapshot,
        "all": _sample_stats(all_samples, include_generation=include_generation),
        "splits": {
            name: _sample_stats(samples, include_generation=include_generation)
            for name, samples in groups.items()
        },
    }


def make_policy_transform(
    *,
    train: bool,
    image_size: int,
    model_data_config: Mapping[str, object],
    augmentation: AugmentationConfig,
) -> transforms.Compose:
    mean = _normalization_tuple(model_data_config, "mean")
    std = _normalization_tuple(model_data_config, "std")
    steps: list[object] = [
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    ]
    if train:
        steps.append(
            transforms.ColorJitter(
                brightness=augmentation.brightness,
                contrast=augmentation.contrast,
                saturation=augmentation.saturation,
                hue=augmentation.hue,
            )
        )
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(steps)


class FrontCamPolicyDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[PolicySample],
        *,
        transform: object,
        horizontal_flip_probability: float = 0.0,
        road_warp: RoadWarpConfig | None = None,
    ) -> None:
        if not 0 <= horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        self.samples = tuple(samples)
        self.transform = transform
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.road_warp = road_warp

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        # Always consume one RNG draw, including for the no-flip baseline, so
        # same-seed A/B runs receive the same subsequent ColorJitter draws.
        horizontal_flipped = bool(
            torch.rand(()).item() < self.horizontal_flip_probability
        )
        with Image.open(sample.image_path) as image:
            rgb_image = image.convert("RGB")
            if self.road_warp is not None:
                rgb_image = warp_pil_image(rgb_image, self.road_warp)
            if horizontal_flipped:
                rgb_image = transform_functional.hflip(rgb_image)
            image_tensor = self.transform(rgb_image)
        angle_raw = -sample.angle_raw if horizontal_flipped else sample.angle_raw
        angle = -sample.angle if horizontal_flipped else sample.angle
        angle_class_id = (
            NUM_COMMAND_CLASSES - 1 - sample.angle_class_id
            if horizontal_flipped
            else sample.angle_class_id
        )
        item: dict[str, object] = {
            "image_tensor": image_tensor,
            "angle": angle,
            "speed": sample.speed,
            "angle_raw": angle_raw,
            "speed_raw": sample.speed_raw,
            "angle_class_id": angle_class_id,
            "speed_class_id": sample.speed_class_id,
            "horizontal_flipped": horizontal_flipped,
            "session_id": sample.session_id,
            "relative_image": sample.relative_image,
            "generation": sample.generation,
        }
        if sample.source_id is not None:
            item["source_id"] = sample.source_id
        if sample.history_class_ids is not None:
            history_class_ids = torch.tensor(
                sample.history_class_ids,
                dtype=torch.long,
            )
            if horizontal_flipped:
                history_class_ids[:, 0] = (
                    NUM_COMMAND_CLASSES - 1 - history_class_ids[:, 0]
                )
            item["history_class_ids"] = history_class_ids
        return item


def compute_sqrt_inverse_frequency_weights(
    samples: Sequence[PolicySample],
    *,
    field: str,
    min_weight: float,
    max_weight: float,
    mirror_probability: float = 0.0,
    sample_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    if field not in {"angle_class_id", "speed_class_id"}:
        raise ValueError(f"unsupported class field: {field}")
    if not 0 <= mirror_probability <= 1:
        raise ValueError("mirror_probability must be in [0, 1]")
    if mirror_probability > 0 and field != "angle_class_id":
        raise ValueError("mirror_probability is only valid for angle_class_id")
    if sample_weights is not None and len(sample_weights) != len(samples):
        raise ValueError("sample_weights length must match samples")
    effective_weights = sample_weights or [1.0] * len(samples)
    raw_counts: Counter[int] = Counter()
    for sample, sample_weight in zip(samples, effective_weights, strict=True):
        if not math.isfinite(sample_weight) or sample_weight <= 0:
            raise ValueError("sample_weights must be finite and positive")
        raw_counts[int(getattr(sample, field))] += float(sample_weight)
    if not raw_counts:
        raise PolicyDatasetError("cannot compute class weights for no samples")
    counts = [
        (1 - mirror_probability) * raw_counts.get(class_id, 0)
        + mirror_probability * raw_counts.get(NUM_COMMAND_CLASSES - 1 - class_id, 0)
        for class_id in range(NUM_COMMAND_CLASSES)
    ]
    nonzero_counts = [count for count in counts if count > 0]
    mean_count = sum(nonzero_counts) / len(nonzero_counts)
    weights = []
    for count in counts:
        if count == 0:
            weight = max_weight
        else:
            weight = math.sqrt(mean_count / count)
            weight = min(max(weight, min_weight), max_weight)
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)


def generation_sampling_weights(
    samples: Sequence[PolicySample],
    *,
    current_generation: int,
    generation_decay: float,
    source_sampling_masses: Mapping[str, float] | None = None,
) -> tuple[float, ...]:
    """Assign generation mass, optionally anchored to fixed source totals."""
    if not samples:
        raise PolicyDatasetError("cannot build generation weights for no samples")
    if not 0 < generation_decay <= 1:
        raise ValueError("generation_decay must be in (0, 1]")
    if source_sampling_masses:
        pair_masses = source_generation_sampling_masses(
            samples,
            current_generation=current_generation,
            generation_decay=generation_decay,
            source_sampling_masses=source_sampling_masses,
        )
        counts = Counter((sample.source_id, sample.generation) for sample in samples)
        return tuple(
            pair_masses[(str(sample.source_id), sample.generation)]
            / counts[(sample.source_id, sample.generation)]
            for sample in samples
        )
    counts = Counter(sample.generation for sample in samples)
    future = sorted(
        generation for generation in counts if generation > current_generation
    )
    if future:
        raise PolicyDatasetError(
            f"dataset contains generation(s) newer than current_generation: {future}"
        )
    if counts.get(current_generation, 0) == 0:
        raise PolicyDatasetError(
            f"training split has no samples for current_generation={current_generation}"
        )
    masses = {
        generation: generation_decay ** (current_generation - generation)
        for generation in counts
    }
    return tuple(
        masses[sample.generation] / counts[sample.generation] for sample in samples
    )


def generation_epoch_sample_count(
    samples: Sequence[PolicySample],
    *,
    current_generation: int,
    generation_decay: float,
    source_sampling_masses: Mapping[str, float] | None = None,
) -> int:
    if source_sampling_masses:
        pair_masses = source_generation_sampling_masses(
            samples,
            current_generation=current_generation,
            generation_decay=generation_decay,
            source_sampling_masses=source_sampling_masses,
        )
        current_count = sum(
            sample.generation == current_generation for sample in samples
        )
        current_mass = sum(
            mass
            for (_, generation), mass in pair_masses.items()
            if generation == current_generation
        )
        if current_count == 0 or current_mass <= 0:
            raise PolicyDatasetError(
                "training split has no samples for "
                f"current_generation={current_generation}"
            )
        return max(1, math.ceil(current_count / current_mass))
    counts = Counter(sample.generation for sample in samples)
    if counts.get(current_generation, 0) == 0:
        raise PolicyDatasetError(
            f"training split has no samples for current_generation={current_generation}"
        )
    total_mass = sum(
        generation_decay ** (current_generation - generation)
        for generation in counts
        if generation <= current_generation
    )
    return max(1, round(counts[current_generation] * total_mass))


def generation_sampling_summary(
    samples: Sequence[PolicySample],
    *,
    current_generation: int,
    generation_decay: float,
    source_sampling_masses: Mapping[str, float] | None = None,
) -> dict[str, object]:
    if source_sampling_masses:
        pair_masses = source_generation_sampling_masses(
            samples,
            current_generation=current_generation,
            generation_decay=generation_decay,
            source_sampling_masses=source_sampling_masses,
        )
        pair_counts = Counter(
            (str(sample.source_id), sample.generation) for sample in samples
        )
        generations = sorted({sample.generation for sample in samples})
        return {
            "mode": "source_anchored_generation_decay",
            "current_generation": current_generation,
            "generation_decay": generation_decay,
            "source_sampling_masses": dict(source_sampling_masses),
            "epoch_sample_count": generation_epoch_sample_count(
                samples,
                current_generation=current_generation,
                generation_decay=generation_decay,
                source_sampling_masses=source_sampling_masses,
            ),
            "generations": {
                str(generation): {
                    "sample_count": sum(
                        count
                        for (source_id, observed_generation), count in pair_counts.items()
                        if observed_generation == generation
                    ),
                    "total_sampling_mass": sum(
                        mass
                        for (source_id, observed_generation), mass in pair_masses.items()
                        if observed_generation == generation
                    ),
                }
                for generation in generations
            },
            "sources": {
                source_id: {
                    "sample_count": sum(
                        count
                        for (observed_source, generation), count in pair_counts.items()
                        if observed_source == source_id
                    ),
                    "total_sampling_mass": source_sampling_masses[source_id],
                    "generations": {
                        str(generation): {
                            "sample_count": pair_counts[(source_id, generation)],
                            "total_sampling_mass": pair_masses[(source_id, generation)],
                        }
                        for observed_source, generation in sorted(pair_masses)
                        if observed_source == source_id
                    },
                }
                for source_id in source_sampling_masses
            },
        }
    counts = Counter(sample.generation for sample in samples)
    return {
        "current_generation": current_generation,
        "generation_decay": generation_decay,
        "epoch_sample_count": generation_epoch_sample_count(
            samples,
            current_generation=current_generation,
            generation_decay=generation_decay,
        ),
        "generations": {
            str(generation): {
                "sample_count": counts[generation],
                "total_sampling_mass": generation_decay
                ** (current_generation - generation),
            }
            for generation in sorted(counts)
        },
    }


def source_generation_sampling_masses(
    samples: Sequence[PolicySample],
    *,
    current_generation: int,
    generation_decay: float,
    source_sampling_masses: Mapping[str, float],
) -> dict[tuple[str, int], float]:
    """Normalize generation decay within each fixed-mass dataset source."""
    if not samples:
        raise PolicyDatasetError("cannot build source generation masses for no samples")
    if not 0 < generation_decay <= 1:
        raise ValueError("generation_decay must be in (0, 1]")
    if not source_sampling_masses:
        raise ValueError("source_sampling_masses must not be empty")
    if any(
        not math.isfinite(mass) or mass <= 0
        for mass in source_sampling_masses.values()
    ):
        raise ValueError("source_sampling_masses must be finite and positive")
    if not math.isclose(
        sum(source_sampling_masses.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("source_sampling_masses must sum to 1")
    observed_sources = {sample.source_id for sample in samples}
    if None in observed_sources or observed_sources != set(source_sampling_masses):
        raise PolicyDatasetError(
            "sample source ids must exactly match source_sampling_masses; "
            f"observed={sorted(str(value) for value in observed_sources)}, "
            f"configured={sorted(source_sampling_masses)}"
        )
    future = sorted(
        {sample.generation for sample in samples if sample.generation > current_generation}
    )
    if future:
        raise PolicyDatasetError(
            f"dataset contains generation(s) newer than current_generation: {future}"
        )
    if not any(sample.generation == current_generation for sample in samples):
        raise PolicyDatasetError(
            f"training split has no samples for current_generation={current_generation}"
        )

    generations_by_source: dict[str, set[int]] = {
        source_id: set() for source_id in source_sampling_masses
    }
    for sample in samples:
        generations_by_source[str(sample.source_id)].add(sample.generation)

    masses: dict[tuple[str, int], float] = {}
    for source_id, source_mass in source_sampling_masses.items():
        raw_masses = {
            generation: generation_decay ** (current_generation - generation)
            for generation in generations_by_source[source_id]
        }
        normalizer = sum(raw_masses.values())
        for generation, raw_mass in raw_masses.items():
            masses[(source_id, generation)] = source_mass * raw_mass / normalizer
    return masses


def _load_policy_session(
    path: Path,
    root: Path,
    metadata: Mapping[str, object],
    *,
    source_id: str | None = None,
    legacy_generation: int | None = 0,
    fixed_generation: int | None = None,
    require_curriculum_generation: bool = False,
) -> PolicySession:
    generation = _metadata_generation(
        metadata,
        legacy_generation=legacy_generation,
        fixed_generation=fixed_generation,
        require_curriculum_generation=require_curriculum_generation,
    )
    qualified_session_id = (
        f"{source_id}/{path.name}" if source_id is not None else path.name
    )
    samples_path = path / "samples.csv"
    images_path = path / "Images"
    if not samples_path.is_file():
        raise PolicyDatasetError(f"missing samples.csv: {samples_path}")
    if not images_path.is_dir():
        raise PolicyDatasetError(f"missing Images directory: {images_path}")

    samples: list[PolicySample] = []
    with samples_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fields = set(reader.fieldnames or ())
        missing_fields = sorted(REQUIRED_CSV_FIELDS - fields)
        if missing_fields:
            raise PolicyDatasetError(
                f"missing samples.csv fields in {samples_path}: {missing_fields}"
            )
        for row_number, row in enumerate(reader, start=2):
            samples.append(
                _sample_from_row(
                    row,
                    row_number,
                    path,
                    root,
                    generation=generation,
                    session_id=qualified_session_id,
                    source_id=source_id,
                )
            )

    if not samples:
        raise PolicyDatasetError(f"session contains no samples: {path}")
    expected_count = metadata.get("sample_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise PolicyDatasetError(f"metadata sample_count must be an integer: {path}")
    if expected_count != len(samples):
        raise PolicyDatasetError(
            f"metadata sample_count mismatch in {path}: {expected_count} != {len(samples)}"
        )
    return PolicySession(
        session_id=qualified_session_id,
        path=path,
        metadata=dict(metadata),
        samples=tuple(samples),
        generation=generation,
        source_id=source_id,
    )


def _sample_from_row(
    row: Mapping[str, str],
    row_number: int,
    session_path: Path,
    data_root: Path,
    *,
    generation: int = 0,
    session_id: str | None = None,
    source_id: str | None = None,
) -> PolicySample:
    image_value = row.get("image", "")
    relative_path = PurePosixPath(image_value)
    if (
        not image_value
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.parts[0] != "Images"
    ):
        raise PolicyDatasetError(
            f"unsafe image path at {session_path / 'samples.csv'}:{row_number}"
        )
    image_path = session_path.joinpath(*relative_path.parts)
    if not image_path.is_file():
        raise PolicyDatasetError(
            f"missing image at {session_path / 'samples.csv'}:{row_number}: {image_path}"
        )

    angle_raw = _float_field(row, "angle", row_number, session_path)
    speed_raw = _float_field(row, "speed", row_number, session_path)
    for name, value in (("angle", angle_raw), ("speed", speed_raw)):
        if not COMMAND_MIN <= value <= COMMAND_MAX:
            raise PolicyDatasetError(
                f"{name} out of range at {session_path / 'samples.csv'}:{row_number}: {value}"
            )
    angle = quantize_command(angle_raw)
    speed = quantize_command(speed_raw)
    return PolicySample(
        session_id=session_id or session_path.name,
        image_path=image_path,
        relative_image=(
            f"{source_id}/{image_path.relative_to(data_root)}"
            if source_id is not None
            else str(image_path.relative_to(data_root))
        ),
        angle_raw=angle_raw,
        speed_raw=speed_raw,
        angle=angle,
        speed=speed,
        angle_class_id=angle + COMMAND_OFFSET,
        speed_class_id=speed + COMMAND_OFFSET,
        generation=generation,
        source_id=source_id,
    )


def metadata_matches_policy_filter(
    metadata: Mapping[str, object],
    *,
    control_mode: str,
    max_forward_speed: float | None,
    min_forward_speed: float | None,
) -> bool:
    if metadata.get("format_version") != 1 or metadata.get("complete") is not True:
        return False
    if metadata.get("dataset_kind") != "camera_first_teleop_behavior_cloning":
        return False
    if metadata.get("control_mode") != control_mode:
        return False
    gamepad = metadata.get("gamepad")
    if not isinstance(gamepad, dict):
        return False
    observed_speed = gamepad.get("max_forward_speed")
    if isinstance(observed_speed, bool) or not isinstance(observed_speed, (int, float)):
        return False
    speed = float(observed_speed)
    if max_forward_speed is not None:
        return math.isclose(
            speed,
            max_forward_speed,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    if min_forward_speed is not None:
        return speed >= min_forward_speed
    raise ValueError("a forward-speed filter is required")


def _matches_filter(metadata: Mapping[str, object], config: DataConfig) -> bool:
    if metadata.get("format_version") != 1 or metadata.get("complete") is not True:
        return False
    if metadata.get("dataset_kind") != "camera_first_teleop_behavior_cloning":
        return False
    if not metadata_has_required_steering_contract(
        metadata, config.required_steering_contract
    ):
        return False
    accepted_modes = config.control_modes or (config.control_mode,)
    if metadata.get("control_mode") not in accepted_modes:
        return False
    if config.max_forward_speed is None and config.min_forward_speed is None:
        return True
    gamepad = metadata.get("gamepad")
    if not isinstance(gamepad, dict):
        return False
    observed_speed = gamepad.get("max_forward_speed")
    if isinstance(observed_speed, bool) or not isinstance(observed_speed, (int, float)):
        return False
    speed = float(observed_speed)
    if config.max_forward_speed is not None:
        return math.isclose(
            speed,
            config.max_forward_speed,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    return speed >= float(config.min_forward_speed)


def _matches_source_filter(
    metadata: Mapping[str, object],
    source: DataSourceConfig,
    *,
    required_steering_contract: str | None,
) -> bool:
    if metadata.get("format_version") != 1 or metadata.get("complete") is not True:
        return False
    if metadata.get("dataset_kind") != "camera_first_teleop_behavior_cloning":
        return False
    if not metadata_has_required_steering_contract(
        metadata, required_steering_contract
    ):
        return False
    return metadata.get("control_mode") in source.control_modes


def _forward_speed_filter_description(config: DataConfig) -> str:
    if config.max_forward_speed is not None:
        return f"max_forward_speed={config.max_forward_speed}"
    if config.min_forward_speed is not None:
        return f"min_forward_speed>={config.min_forward_speed}"
    return "configured control modes"


def _metadata_generation(
    metadata: Mapping[str, object],
    *,
    legacy_generation: int | None,
    fixed_generation: int | None = None,
    require_curriculum_generation: bool = False,
) -> int:
    if fixed_generation is not None:
        return fixed_generation
    curriculum = metadata.get("curriculum")
    if not isinstance(curriculum, dict) or "generation" not in curriculum:
        if require_curriculum_generation or legacy_generation is None:
            raise PolicyDatasetError(
                "curriculum.generation is required for this dataset source"
            )
        return legacy_generation
    generation = curriculum.get("generation")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise PolicyDatasetError("curriculum.generation must be a non-negative integer")
    return generation


def _load_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise PolicyDatasetError(f"missing metadata.yaml: {path}")
    return _load_yaml_mapping(path)


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PolicyDatasetError(f"invalid YAML: {path}") from exc
    if not isinstance(payload, dict):
        raise PolicyDatasetError(f"YAML root must be a mapping: {path}")
    return dict(payload)


def _float_field(
    row: Mapping[str, str], field: str, row_number: int, session_path: Path
) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyDatasetError(
            f"invalid {field} at {session_path / 'samples.csv'}:{row_number}"
        ) from exc
    if not math.isfinite(value):
        raise PolicyDatasetError(
            f"non-finite {field} at {session_path / 'samples.csv'}:{row_number}"
        )
    return value


def _normalization_tuple(
    data_config: Mapping[str, object], key: str
) -> tuple[float, float, float]:
    value = data_config.get(key)
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"timm model data config has invalid {key}: {value!r}")
    return tuple(float(item) for item in value)


def _sample_stats(
    samples: Sequence[PolicySample], *, include_generation: bool = False
) -> dict[str, object]:
    angle_counts = Counter(sample.angle for sample in samples)
    speed_counts = Counter(sample.speed for sample in samples)
    stats: dict[str, object] = {
        "sample_count": len(samples),
        "session_count": len({sample.session_id for sample in samples}),
        "angle_range": _range_or_none(sample.angle_raw for sample in samples),
        "speed_range": _range_or_none(sample.speed_raw for sample in samples),
        "angle_class_counts": _counter_dict(angle_counts),
        "speed_class_counts": _counter_dict(speed_counts),
        "angle_buckets": _angle_bucket_counts(samples),
    }
    if include_generation:
        stats["generation_sample_counts"] = _counter_dict(
            Counter(sample.generation for sample in samples)
        )
        source_generation_counts: dict[str, Counter[int]] = {}
        for sample in samples:
            if sample.source_id is not None:
                source_generation_counts.setdefault(sample.source_id, Counter())[
                    sample.generation
                ] += 1
        if source_generation_counts:
            stats["source_generation_sample_counts"] = {
                source_id: _counter_dict(counts)
                for source_id, counts in sorted(source_generation_counts.items())
            }
    return stats


def _angle_bucket_counts(samples: Sequence[PolicySample]) -> dict[str, int]:
    buckets = {
        "hard_left": 0,
        "left": 0,
        "near_zero": 0,
        "right": 0,
        "hard_right": 0,
    }
    for sample in samples:
        if sample.angle <= -61:
            buckets["hard_left"] += 1
        elif sample.angle <= -11:
            buckets["left"] += 1
        elif sample.angle <= 10:
            buckets["near_zero"] += 1
        elif sample.angle <= 60:
            buckets["right"] += 1
        else:
            buckets["hard_right"] += 1
    return buckets


def _range_or_none(values: Iterable[float]) -> list[float] | None:
    collected = list(values)
    if not collected:
        return None
    return [min(collected), max(collected)]


def _counter_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


def _manifest_integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PolicyDatasetError(f"split manifest {key} must be an integer")
    return value


def _manifest_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise PolicyDatasetError(f"split manifest {key} must be a string")
    return value


def _flatten_samples(
    sessions: Sequence[PolicySession],
) -> tuple[PolicySample, ...]:
    return tuple(sample for session in sessions for sample in session.samples)


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates
