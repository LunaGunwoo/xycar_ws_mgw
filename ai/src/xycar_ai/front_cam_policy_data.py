from __future__ import annotations

import csv
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from xycar_ai.config import AugmentationConfig, DataConfig

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


@dataclass(frozen=True)
class PolicySession:
    session_id: str
    path: Path
    metadata: Mapping[str, object]
    samples: tuple[PolicySample, ...]


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

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "dataset_snapshot": self.dataset_snapshot,
            "splits": {
                name: {
                    "sessions": [session.session_id for session in sessions],
                    "session_count": len(sessions),
                    "sample_count": sum(len(session.samples) for session in sessions),
                }
                for name, sessions in self._session_groups().items()
            },
        }

    def _session_groups(self) -> dict[str, tuple[PolicySession, ...]]:
        return {
            "train": self.train_sessions,
            "val": self.val_sessions,
            "test": self.test_sessions,
        }


def quantize_command(value: float) -> int:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise PolicyDatasetError(f"command must be finite, got {value!r}")
    clamped = max(float(COMMAND_MIN), min(float(COMMAND_MAX), numeric))
    return round(clamped)


def command_class_id(value: float) -> int:
    return quantize_command(value) + COMMAND_OFFSET


def discover_policy_sessions(config: DataConfig) -> tuple[PolicySession, ...]:
    root = config.root
    if not root.is_dir():
        raise PolicyDatasetError(f"dataset root does not exist: {root}")

    sessions: list[PolicySession] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not SESSION_NAME_RE.fullmatch(path.name):
            continue
        metadata = _load_metadata(path / "metadata.yaml")
        if not _matches_filter(metadata, config):
            continue
        sessions.append(_load_policy_session(path, root, metadata))

    if not sessions:
        raise PolicyDatasetError(
            "no completed gamepad sessions match "
            f"max_forward_speed={config.max_forward_speed} under {root}"
        )
    return tuple(sessions)


def build_policy_data_splits(
    config: DataConfig,
) -> PolicyDataSplits:
    sessions = discover_policy_sessions(config)
    payload = _load_yaml_mapping(config.split_manifest)
    if set(payload) != {"schema_version", "dataset_snapshot", "splits"}:
        raise PolicyDatasetError(
            f"unexpected split manifest keys: {config.split_manifest}"
        )
    if _manifest_integer(payload, "schema_version") != 1:
        raise PolicyDatasetError("unsupported split manifest schema_version")
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

    return PolicyDataSplits(
        dataset_snapshot=dataset_snapshot,
        train_sessions=split_sessions["train"],
        val_sessions=split_sessions["val"],
        test_sessions=split_sessions["test"],
    )


def policy_dataset_stats(splits: PolicyDataSplits) -> dict[str, object]:
    groups = {
        "train": splits.train_samples,
        "val": splits.val_samples,
        "test": splits.test_samples,
    }
    all_samples = tuple(sample for samples in groups.values() for sample in samples)
    return {
        "schema_version": 1,
        "dataset_snapshot": splits.dataset_snapshot,
        "all": _sample_stats(all_samples),
        "splits": {name: _sample_stats(samples) for name, samples in groups.items()},
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
    ) -> None:
        if not 0 <= horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        self.samples = tuple(samples)
        self.transform = transform
        self.horizontal_flip_probability = float(horizontal_flip_probability)

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
        return {
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
        }


def compute_sqrt_inverse_frequency_weights(
    samples: Sequence[PolicySample],
    *,
    field: str,
    min_weight: float,
    max_weight: float,
    mirror_probability: float = 0.0,
) -> torch.Tensor:
    if field not in {"angle_class_id", "speed_class_id"}:
        raise ValueError(f"unsupported class field: {field}")
    if not 0 <= mirror_probability <= 1:
        raise ValueError("mirror_probability must be in [0, 1]")
    if mirror_probability > 0 and field != "angle_class_id":
        raise ValueError("mirror_probability is only valid for angle_class_id")
    raw_counts = Counter(int(getattr(sample, field)) for sample in samples)
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


def _load_policy_session(
    path: Path,
    root: Path,
    metadata: Mapping[str, object],
) -> PolicySession:
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
            samples.append(_sample_from_row(row, row_number, path, root))

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
        session_id=path.name,
        path=path,
        metadata=dict(metadata),
        samples=tuple(samples),
    )


def _sample_from_row(
    row: Mapping[str, str],
    row_number: int,
    session_path: Path,
    data_root: Path,
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
        session_id=session_path.name,
        image_path=image_path,
        relative_image=str(image_path.relative_to(data_root)),
        angle_raw=angle_raw,
        speed_raw=speed_raw,
        angle=angle,
        speed=speed,
        angle_class_id=angle + COMMAND_OFFSET,
        speed_class_id=speed + COMMAND_OFFSET,
    )


def _matches_filter(metadata: Mapping[str, object], config: DataConfig) -> bool:
    if metadata.get("format_version") != 1 or metadata.get("complete") is not True:
        return False
    if metadata.get("dataset_kind") != "camera_first_teleop_behavior_cloning":
        return False
    if metadata.get("control_mode") != config.control_mode:
        return False
    gamepad = metadata.get("gamepad")
    if not isinstance(gamepad, dict):
        return False
    max_forward_speed = gamepad.get("max_forward_speed")
    if isinstance(max_forward_speed, bool) or not isinstance(
        max_forward_speed, (int, float)
    ):
        return False
    return math.isclose(
        float(max_forward_speed),
        config.max_forward_speed,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


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


def _sample_stats(samples: Sequence[PolicySample]) -> dict[str, object]:
    angle_counts = Counter(sample.angle for sample in samples)
    speed_counts = Counter(sample.speed for sample in samples)
    return {
        "sample_count": len(samples),
        "session_count": len({sample.session_id for sample in samples}),
        "angle_range": _range_or_none(sample.angle_raw for sample in samples),
        "speed_range": _range_or_none(sample.speed_raw for sample in samples),
        "angle_class_counts": _counter_dict(angle_counts),
        "speed_class_counts": _counter_dict(speed_counts),
        "angle_buckets": _angle_bucket_counts(samples),
    }


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
