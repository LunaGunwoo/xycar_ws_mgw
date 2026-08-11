from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_SCHEMA_VERSION = 1
CLASS_WEIGHTING_MODES = {"none", "sqrt_inverse_frequency"}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    pretrained: bool
    image_size: int


@dataclass(frozen=True)
class DataConfig:
    root: Path
    split_manifest: Path
    require_all_matching_sessions: bool
    control_mode: str
    max_forward_speed: float
    num_workers: int


@dataclass(frozen=True)
class AugmentationConfig:
    brightness: float
    contrast: float
    saturation: float
    hue: float
    horizontal_flip_probability: float


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    warmup_epochs: int


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    grad_clip: float
    seed: int
    device: str
    amp: bool
    deterministic: bool


@dataclass(frozen=True)
class LossConfig:
    angle_label_smoothing: float
    speed_label_smoothing: float
    angle_class_weighting: str
    speed_class_weighting: str
    min_class_weight: float
    max_class_weight: float
    speed_loss_weight: float
    emd_loss_weight: float


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    run_name: str


@dataclass(frozen=True)
class TrainConfig:
    config_path: Path
    project_root: Path
    model: ModelConfig
    data: DataConfig
    augmentation: AugmentationConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    loss: LossConfig
    output: OutputConfig

    def serializable(self) -> dict[str, object]:
        payload = asdict(self)
        return _paths_to_strings(payload)


def load_train_config(path: str | Path) -> TrainConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"training config does not exist: {config_path}")
    payload = _load_yaml_mapping(config_path)
    _expect_keys(
        payload,
        {
            "schema_version",
            "model",
            "data",
            "augmentation",
            "optimizer",
            "scheduler",
            "training",
            "loss",
            "output",
        },
        "training config",
    )
    if _integer(payload, "schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unsupported training config schema_version in {config_path}")

    project_root = config_path.parent.parent
    model_payload = _mapping(payload, "model")
    data_payload = _mapping(payload, "data")
    augmentation_payload = _mapping(payload, "augmentation")
    optimizer_payload = _mapping(payload, "optimizer")
    scheduler_payload = _mapping(payload, "scheduler")
    training_payload = _mapping(payload, "training")
    loss_payload = _mapping(payload, "loss")
    output_payload = _mapping(payload, "output")

    _expect_keys(model_payload, {"name", "pretrained", "image_size"}, "model")
    _expect_keys(
        data_payload,
        {
            "root",
            "split_manifest",
            "require_all_matching_sessions",
            "control_mode",
            "max_forward_speed",
            "num_workers",
        },
        "data",
    )
    _expect_keys(
        augmentation_payload,
        {
            "brightness",
            "contrast",
            "saturation",
            "hue",
            "horizontal_flip_probability",
        },
        "augmentation",
    )
    _expect_keys(
        optimizer_payload,
        {"name", "learning_rate", "weight_decay"},
        "optimizer",
    )
    _expect_keys(scheduler_payload, {"name", "warmup_epochs"}, "scheduler")
    _expect_keys(
        training_payload,
        {
            "epochs",
            "batch_size",
            "grad_clip",
            "seed",
            "device",
            "amp",
            "deterministic",
        },
        "training",
    )
    _expect_keys(
        loss_payload,
        {
            "angle_label_smoothing",
            "speed_label_smoothing",
            "angle_class_weighting",
            "speed_class_weighting",
            "min_class_weight",
            "max_class_weight",
            "speed_loss_weight",
            "emd_loss_weight",
        },
        "loss",
    )
    _expect_keys(output_payload, {"root", "run_name"}, "output")

    config = TrainConfig(
        config_path=config_path,
        project_root=project_root,
        model=ModelConfig(
            name=_string(model_payload, "name"),
            pretrained=_boolean(model_payload, "pretrained"),
            image_size=_integer(model_payload, "image_size"),
        ),
        data=DataConfig(
            root=_resolve_project_path(project_root, _string(data_payload, "root")),
            split_manifest=_resolve_project_path(
                project_root, _string(data_payload, "split_manifest")
            ),
            require_all_matching_sessions=_boolean(
                data_payload, "require_all_matching_sessions"
            ),
            control_mode=_string(data_payload, "control_mode"),
            max_forward_speed=_number(data_payload, "max_forward_speed"),
            num_workers=_integer(data_payload, "num_workers"),
        ),
        augmentation=AugmentationConfig(
            brightness=_number(augmentation_payload, "brightness"),
            contrast=_number(augmentation_payload, "contrast"),
            saturation=_number(augmentation_payload, "saturation"),
            hue=_number(augmentation_payload, "hue"),
            horizontal_flip_probability=_number(
                augmentation_payload, "horizontal_flip_probability"
            ),
        ),
        optimizer=OptimizerConfig(
            name=_string(optimizer_payload, "name"),
            learning_rate=_number(optimizer_payload, "learning_rate"),
            weight_decay=_number(optimizer_payload, "weight_decay"),
        ),
        scheduler=SchedulerConfig(
            name=_string(scheduler_payload, "name"),
            warmup_epochs=_integer(scheduler_payload, "warmup_epochs"),
        ),
        training=TrainingConfig(
            epochs=_integer(training_payload, "epochs"),
            batch_size=_integer(training_payload, "batch_size"),
            grad_clip=_number(training_payload, "grad_clip"),
            seed=_integer(training_payload, "seed"),
            device=_string(training_payload, "device"),
            amp=_boolean(training_payload, "amp"),
            deterministic=_boolean(training_payload, "deterministic"),
        ),
        loss=LossConfig(
            angle_label_smoothing=_number(loss_payload, "angle_label_smoothing"),
            speed_label_smoothing=_number(loss_payload, "speed_label_smoothing"),
            angle_class_weighting=_string(loss_payload, "angle_class_weighting"),
            speed_class_weighting=_string(loss_payload, "speed_class_weighting"),
            min_class_weight=_number(loss_payload, "min_class_weight"),
            max_class_weight=_number(loss_payload, "max_class_weight"),
            speed_loss_weight=_number(loss_payload, "speed_loss_weight"),
            emd_loss_weight=_number(loss_payload, "emd_loss_weight"),
        ),
        output=OutputConfig(
            root=_resolve_project_path(project_root, _string(output_payload, "root")),
            run_name=_optional_string(output_payload, "run_name"),
        ),
    )
    _validate(config)
    return config


def _validate(config: TrainConfig) -> None:
    if config.model.image_size <= 0:
        raise ValueError("model.image_size must be > 0")
    if config.model.pretrained and config.model.image_size != 224:
        raise ValueError("the selected pretrained ViT requires image_size 224")
    if config.data.num_workers < 0:
        raise ValueError("data.num_workers must be >= 0")
    if not -100.0 <= config.data.max_forward_speed <= 100.0:
        raise ValueError("data.max_forward_speed must be in [-100, 100]")
    for field_name, value in asdict(config.augmentation).items():
        if value < 0:
            raise ValueError(f"augmentation.{field_name} must be >= 0")
    if config.augmentation.hue > 0.5:
        raise ValueError("augmentation.hue must be <= 0.5")
    if config.augmentation.horizontal_flip_probability > 1:
        raise ValueError("augmentation.horizontal_flip_probability must be <= 1")
    optimizer = config.optimizer
    if optimizer.name != "adamw":
        raise ValueError("optimizer.name must be adamw")
    if optimizer.learning_rate <= 0:
        raise ValueError("optimizer.learning_rate must be > 0")
    if optimizer.weight_decay < 0:
        raise ValueError("optimizer.weight_decay must be >= 0")
    scheduler = config.scheduler
    if scheduler.name != "cosine":
        raise ValueError("scheduler.name must be cosine")
    training = config.training
    if training.epochs <= 0:
        raise ValueError("training.epochs must be > 0")
    if training.batch_size <= 0:
        raise ValueError("training.batch_size must be > 0")
    if training.grad_clip < 0:
        raise ValueError("training.grad_clip must be >= 0")
    if scheduler.warmup_epochs < 0 or scheduler.warmup_epochs > training.epochs:
        raise ValueError("scheduler.warmup_epochs must be in [0, training.epochs]")
    loss = config.loss
    if loss.angle_class_weighting not in CLASS_WEIGHTING_MODES:
        raise ValueError("unsupported loss.angle_class_weighting")
    if loss.speed_class_weighting not in CLASS_WEIGHTING_MODES:
        raise ValueError("unsupported loss.speed_class_weighting")
    if not 0 <= loss.angle_label_smoothing < 1:
        raise ValueError("loss.angle_label_smoothing must be in [0, 1)")
    if not 0 <= loss.speed_label_smoothing < 1:
        raise ValueError("loss.speed_label_smoothing must be in [0, 1)")
    if loss.min_class_weight <= 0:
        raise ValueError("loss.min_class_weight must be > 0")
    if loss.max_class_weight < loss.min_class_weight:
        raise ValueError("loss.max_class_weight must be >= min_class_weight")
    if loss.speed_loss_weight < 0 or loss.emd_loss_weight < 0:
        raise ValueError("loss weights must be >= 0")


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return dict(payload)


def _expect_keys(payload: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(payload)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _mapping(payload: Mapping[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return dict(value)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _boolean(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _paths_to_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _paths_to_strings(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_paths_to_strings(item) for item in value]
    return value
