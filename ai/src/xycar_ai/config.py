from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import re
from typing import Any

import yaml

from xycar_ai.steering_contract import validate_required_contract_name

CONFIG_SCHEMA_VERSIONS = {1, 2, 3}
CLASS_WEIGHTING_MODES = {"none", "sqrt_inverse_frequency"}
MODEL_ARCHITECTURES = {"task_tokens", "ar_control_tokens"}
HISTORY_UPDATE_MODES = {"predicted_argmax", "externally_executed_commands"}
STATELESS_EMA_MODEL = "vit_small_patch16_224.augreg_in21k_ft_in1k"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    pretrained: bool
    image_size: int
    architecture: str = "task_tokens"
    history_frames: int = 0
    control_token_type_embedding: bool = False
    history_initial_angle: int = 0
    history_initial_speed: int = 25
    history_update: str = "predicted_argmax"


@dataclass(frozen=True)
class DataSourceConfig:
    source_id: str
    root: Path
    control_modes: tuple[str, ...]
    fixed_generation: int | None = None
    require_curriculum_generation: bool = False


@dataclass(frozen=True)
class DataConfig:
    root: Path | None
    split_manifest: Path
    require_all_matching_sessions: bool
    control_mode: str
    max_forward_speed: float | None
    min_forward_speed: float | None
    num_workers: int
    train_angle_mean_window: int = 1
    control_modes: tuple[str, ...] = ()
    current_generation: int = 0
    generation_decay: float = 1.0
    legacy_generation: int = 0
    ema_sampling: bool = False
    sources: tuple[DataSourceConfig, ...] = ()
    source_sampling_masses: dict[str, float] = field(default_factory=dict)
    manual_anchor_split_manifest: Path | None = None
    current_generation_session_counts: dict[str, int] = field(default_factory=dict)
    required_steering_contract: str | None = None
    minimum_total_samples: int = 0
    minimum_total_sessions: int = 0
    minimum_train_sessions: int = 0
    minimum_val_sessions: int = 0
    minimum_test_sessions: int = 0


@dataclass(frozen=True)
class AugmentationConfig:
    brightness: float
    contrast: float
    saturation: float
    hue: float
    horizontal_flip_probability: float


@dataclass(frozen=True)
class PreprocessingConfig:
    road_warp_config: Path | None


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
    early_stopping_patience: int | None
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
    preprocessing: PreprocessingConfig
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
        optional={"preprocessing"},
    )
    schema_version = _integer(payload, "schema_version")
    if schema_version not in CONFIG_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported training config schema_version in {config_path}")

    project_root = config_path.parent.parent
    model_payload = _mapping(payload, "model")
    data_payload = _mapping(payload, "data")
    preprocessing_payload = (
        _mapping(payload, "preprocessing") if "preprocessing" in payload else None
    )
    augmentation_payload = _mapping(payload, "augmentation")
    optimizer_payload = _mapping(payload, "optimizer")
    scheduler_payload = _mapping(payload, "scheduler")
    training_payload = _mapping(payload, "training")
    loss_payload = _mapping(payload, "loss")
    output_payload = _mapping(payload, "output")

    _expect_keys(
        model_payload,
        {"name", "pretrained", "image_size"},
        "model",
        optional={
            "architecture",
            "history_frames",
            "control_token_type_embedding",
            "history_initial_angle",
            "history_initial_speed",
            "history_update",
        },
    )
    _expect_data_keys(data_payload, schema_version=schema_version)
    if preprocessing_payload is not None:
        _expect_keys(
            preprocessing_payload,
            {"road_warp_config"},
            "preprocessing",
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
        optional={"early_stopping_patience"},
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
            architecture=(
                _string(model_payload, "architecture")
                if "architecture" in model_payload
                else "task_tokens"
            ),
            history_frames=(
                _integer(model_payload, "history_frames")
                if "history_frames" in model_payload
                else 0
            ),
            control_token_type_embedding=(
                _boolean(model_payload, "control_token_type_embedding")
                if "control_token_type_embedding" in model_payload
                else False
            ),
            history_initial_angle=(
                _integer(model_payload, "history_initial_angle")
                if "history_initial_angle" in model_payload
                else 0
            ),
            history_initial_speed=(
                _integer(model_payload, "history_initial_speed")
                if "history_initial_speed" in model_payload
                else 25
            ),
            history_update=(
                _string(model_payload, "history_update")
                if "history_update" in model_payload
                else "predicted_argmax"
            ),
        ),
        data=DataConfig(
            root=(
                None
                if schema_version == 3
                else _resolve_project_path(
                    project_root, _string(data_payload, "root")
                )
            ),
            split_manifest=_resolve_project_path(
                project_root, _string(data_payload, "split_manifest")
            ),
            require_all_matching_sessions=_boolean(
                data_payload, "require_all_matching_sessions"
            ),
            control_mode=(
                _string(data_payload, "control_mode")
                if "control_mode" in data_payload
                else ""
            ),
            max_forward_speed=(
                _number(data_payload, "max_forward_speed")
                if "max_forward_speed" in data_payload
                else None
            ),
            min_forward_speed=(
                _number(data_payload, "min_forward_speed")
                if "min_forward_speed" in data_payload
                else None
            ),
            num_workers=_integer(data_payload, "num_workers"),
            train_angle_mean_window=(
                _integer(data_payload, "train_angle_mean_window")
                if "train_angle_mean_window" in data_payload
                else 1
            ),
            control_modes=(
                _string_tuple(data_payload, "control_modes")
                if "control_modes" in data_payload
                else ()
            ),
            current_generation=(
                _integer(data_payload, "current_generation")
                if "current_generation" in data_payload
                else 0
            ),
            generation_decay=(
                _number(data_payload, "generation_decay")
                if "generation_decay" in data_payload
                else 1.0
            ),
            legacy_generation=(
                _integer(data_payload, "legacy_generation")
                if "legacy_generation" in data_payload
                else 0
            ),
            ema_sampling=schema_version in {2, 3},
            sources=(
                _parse_data_sources(project_root, data_payload)
                if schema_version == 3
                else ()
            ),
            source_sampling_masses=(
                _parse_source_sampling_masses(data_payload)
                if "source_sampling_masses" in data_payload
                else {}
            ),
            manual_anchor_split_manifest=(
                _resolve_project_path(
                    project_root,
                    _string(data_payload, "manual_anchor_split_manifest"),
                )
                if "manual_anchor_split_manifest" in data_payload
                else None
            ),
            current_generation_session_counts=(
                _parse_split_session_counts(data_payload)
                if "current_generation_session_counts" in data_payload
                else {}
            ),
            required_steering_contract=(
                _string(data_payload, "required_steering_contract")
                if "required_steering_contract" in data_payload
                else None
            ),
            minimum_total_samples=(
                _integer(data_payload, "minimum_total_samples")
                if "minimum_total_samples" in data_payload
                else 0
            ),
            minimum_total_sessions=(
                _integer(data_payload, "minimum_total_sessions")
                if "minimum_total_sessions" in data_payload
                else 0
            ),
            minimum_train_sessions=(
                _integer(data_payload, "minimum_train_sessions")
                if "minimum_train_sessions" in data_payload
                else 0
            ),
            minimum_val_sessions=(
                _integer(data_payload, "minimum_val_sessions")
                if "minimum_val_sessions" in data_payload
                else 0
            ),
            minimum_test_sessions=(
                _integer(data_payload, "minimum_test_sessions")
                if "minimum_test_sessions" in data_payload
                else 0
            ),
        ),
        preprocessing=PreprocessingConfig(
            road_warp_config=(
                _resolve_project_path(
                    project_root,
                    _string(preprocessing_payload, "road_warp_config"),
                )
                if preprocessing_payload is not None
                else None
            ),
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
            early_stopping_patience=(
                _integer(training_payload, "early_stopping_patience")
                if "early_stopping_patience" in training_payload
                else None
            ),
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
    validate_required_contract_name(config.data.required_steering_contract)
    minimum_fields = (
        config.data.minimum_total_samples,
        config.data.minimum_total_sessions,
        config.data.minimum_train_sessions,
        config.data.minimum_val_sessions,
        config.data.minimum_test_sessions,
    )
    if any(value < 0 for value in minimum_fields):
        raise ValueError("data minimum dataset sizes must be >= 0")
    if config.model.image_size <= 0:
        raise ValueError("model.image_size must be > 0")
    if config.model.pretrained and config.model.image_size != 224:
        raise ValueError("the selected pretrained ViT requires image_size 224")
    if config.model.architecture not in MODEL_ARCHITECTURES:
        raise ValueError("unsupported model.architecture")
    if config.model.history_update not in HISTORY_UPDATE_MODES:
        raise ValueError("unsupported model.history_update")
    if not -100 <= config.model.history_initial_angle <= 100:
        raise ValueError("model.history_initial_angle must be in [-100, 100]")
    if not -100 <= config.model.history_initial_speed <= 100:
        raise ValueError("model.history_initial_speed must be in [-100, 100]")
    if config.model.architecture == "task_tokens":
        if config.model.history_frames != 0:
            raise ValueError("task_tokens model.history_frames must be 0")
        if config.model.control_token_type_embedding:
            raise ValueError(
                "task_tokens model cannot use control_token_type_embedding"
            )
    else:
        if config.model.history_frames != 4:
            raise ValueError("ar_control_tokens model.history_frames must be 4")
        if (
            config.model.history_initial_angle,
            config.model.history_initial_speed,
        ) != (0, 25):
            raise ValueError(
                "ar_control_tokens initial history command must be (0, 25)"
            )
    if config.data.num_workers < 0:
        raise ValueError("data.num_workers must be >= 0")
    if config.data.sources:
        if config.data.root is not None:
            raise ValueError("multi-source data.root must be unset")
        source_ids = [source.source_id for source in config.data.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("data.sources ids must not contain duplicates")
        source_roots = [source.root for source in config.data.sources]
        if len(set(source_roots)) != len(source_roots):
            raise ValueError("data.sources roots must be distinct")
        for source in config.data.sources:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", source.source_id):
                raise ValueError(f"invalid data source id: {source.source_id!r}")
            if len(set(source.control_modes)) != len(source.control_modes):
                raise ValueError(
                    f"data.sources.{source.source_id}.control_modes has duplicates"
                )
            if source.fixed_generation is not None and source.fixed_generation < 0:
                raise ValueError(
                    f"data.sources.{source.source_id}.fixed_generation must be >= 0"
                )
            if (
                source.fixed_generation is not None
                and source.fixed_generation > config.data.current_generation
            ):
                raise ValueError(
                    f"data.sources.{source.source_id}.fixed_generation cannot "
                    "exceed current_generation"
                )
        sources_by_id = {
            source.source_id: source for source in config.data.sources
        }
        if set(sources_by_id) != {"manual", "guided"}:
            raise ValueError(
                "multi-source stateless data requires manual and guided sources"
            )
        manual = sources_by_id["manual"]
        guided = sources_by_id["guided"]
        if manual.control_modes != ("gamepad",) or manual.fixed_generation != 0:
            raise ValueError(
                "manual source must accept only gamepad sessions at fixed_generation 0"
            )
        if (
            guided.control_modes != ("guided_policy",)
            or not guided.require_curriculum_generation
        ):
            raise ValueError(
                "guided source must accept only guided_policy sessions and require generation"
            )
        if config.data.source_sampling_masses:
            masses = config.data.source_sampling_masses
            if set(masses) != set(source_ids):
                raise ValueError(
                    "data.source_sampling_masses keys must exactly match data.sources"
                )
            if any(not math.isfinite(value) or value <= 0 for value in masses.values()):
                raise ValueError(
                    "data.source_sampling_masses values must be finite and positive"
                )
            if not math.isclose(
                sum(masses.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("data.source_sampling_masses must sum to 1")
            if config.data.manual_anchor_split_manifest is None:
                raise ValueError(
                    "data.manual_anchor_split_manifest is required with source masses"
                )
            if not config.data.current_generation_session_counts:
                raise ValueError(
                    "data.current_generation_session_counts is required with "
                    "source masses"
                )
        elif (
            config.data.manual_anchor_split_manifest is not None
            or config.data.current_generation_session_counts
        ):
            raise ValueError(
                "manual anchor fields require data.source_sampling_masses"
            )
        if (
            config.model.name != STATELESS_EMA_MODEL
            or config.model.architecture != "task_tokens"
            or config.model.history_frames != 0
            or not config.model.pretrained
            or config.model.image_size != 224
            or config.data.train_angle_mean_window != 1
            or config.preprocessing.road_warp_config is None
        ):
            raise ValueError(
                "multi-source stateless data requires pretrained ViT-Small, "
                "task_tokens, 224 road warp, and raw instantaneous angle"
            )
        if not config.output.run_name.endswith(
            f"generation{config.data.current_generation}"
        ):
            raise ValueError(
                "multi-source output.run_name must end with current generation"
            )
    else:
        if (
            config.data.source_sampling_masses
            or config.data.manual_anchor_split_manifest is not None
            or config.data.current_generation_session_counts
        ):
            raise ValueError(
                "source anchor fields require multi-source data.sources"
            )
        if config.data.root is None:
            raise ValueError("single-source data.root is required")
        if bool(config.data.control_mode) == bool(config.data.control_modes):
            raise ValueError(
                "exactly one of data.control_mode or data.control_modes is required"
            )
        if config.data.control_modes and len(set(config.data.control_modes)) != len(
            config.data.control_modes
        ):
            raise ValueError("data.control_modes must not contain duplicates")
    if config.data.current_generation < 0 or config.data.legacy_generation < 0:
        raise ValueError("data generation numbers must be >= 0")
    if config.data.legacy_generation > config.data.current_generation:
        raise ValueError("data.legacy_generation cannot exceed current_generation")
    if not 0 < config.data.generation_decay <= 1:
        raise ValueError("data.generation_decay must be in (0, 1]")
    if (
        config.data.train_angle_mean_window <= 0
        or config.data.train_angle_mean_window % 2 == 0
    ):
        raise ValueError("data.train_angle_mean_window must be a positive odd integer")
    if config.data.max_forward_speed is not None and not (
        -100.0 <= config.data.max_forward_speed <= 100.0
    ):
        raise ValueError("data.max_forward_speed must be in [-100, 100]")
    if config.data.min_forward_speed is not None and not (
        -100.0 <= config.data.min_forward_speed <= 100.0
    ):
        raise ValueError("data.min_forward_speed must be in [-100, 100]")
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
    if (
        training.early_stopping_patience is not None
        and training.early_stopping_patience <= 0
    ):
        raise ValueError("training.early_stopping_patience must be > 0")
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


def _expect_keys(
    payload: Mapping[str, object],
    expected: set[str],
    label: str,
    *,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    actual = set(payload)
    missing = expected - actual
    extra = actual - expected - optional
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _expect_data_keys(payload: Mapping[str, object], *, schema_version: int) -> None:
    if schema_version == 3:
        _expect_keys(
            payload,
            {
                "sources",
                "split_manifest",
                "require_all_matching_sessions",
                "num_workers",
                "current_generation",
                "generation_decay",
            },
            "data",
            optional={
                "train_angle_mean_window",
                "required_steering_contract",
                "minimum_total_samples",
                "minimum_total_sessions",
                "minimum_train_sessions",
                "minimum_val_sessions",
                "minimum_test_sessions",
                "source_sampling_masses",
                "manual_anchor_split_manifest",
                "current_generation_session_counts",
            },
        )
        return

    if schema_version == 2:
        required = {
            "root",
            "split_manifest",
            "require_all_matching_sessions",
            "num_workers",
            "current_generation",
            "generation_decay",
            "legacy_generation",
        }
        mode_fields = {"control_mode", "control_modes"}
        filters = {"max_forward_speed", "min_forward_speed"}
        optional = {
            "train_angle_mean_window",
            "required_steering_contract",
            "minimum_total_samples",
            "minimum_total_sessions",
            "minimum_train_sessions",
            "minimum_val_sessions",
            "minimum_test_sessions",
        }
        actual = set(payload)
        missing = required - actual
        extra = actual - required - mode_fields - filters - optional
        if (
            missing
            or extra
            or len(actual & mode_fields) != 1
            or len(actual & filters) > 1
        ):
            raise ValueError(
                "data keys mismatch; "
                f"missing={sorted(missing)}, extra={sorted(extra)}, "
                "exactly one control mode field and at most one speed filter are required"
            )
        return

    required = {
        "root",
        "split_manifest",
        "require_all_matching_sessions",
        "control_mode",
        "num_workers",
    }
    filters = {"max_forward_speed", "min_forward_speed"}
    optional = {
        "train_angle_mean_window",
        "required_steering_contract",
        "minimum_total_samples",
        "minimum_total_sessions",
        "minimum_train_sessions",
        "minimum_val_sessions",
        "minimum_test_sessions",
    }
    actual = set(payload)
    missing = required - actual
    extra = actual - required - filters - optional
    selected_filters = actual & filters
    if missing or extra or len(selected_filters) != 1:
        raise ValueError(
            "data keys mismatch; "
            f"missing={sorted(missing)}, extra={sorted(extra)}, "
            "exactly one of max_forward_speed or min_forward_speed is required"
        )


def _parse_data_sources(
    project_root: Path, data_payload: Mapping[str, object]
) -> tuple[DataSourceConfig, ...]:
    sources_payload = data_payload.get("sources")
    if not isinstance(sources_payload, dict) or not sources_payload:
        raise ValueError("data.sources must be a non-empty mapping")
    sources: list[DataSourceConfig] = []
    for source_id, raw_source in sources_payload.items():
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("data.sources keys must be non-empty strings")
        if not isinstance(raw_source, dict):
            raise TypeError(f"data.sources.{source_id} must be a mapping")
        source_payload = dict(raw_source)
        _expect_keys(
            source_payload,
            {"root", "control_modes"},
            f"data.sources.{source_id}",
            optional={"fixed_generation", "require_curriculum_generation"},
        )
        generation_keys = {
            "fixed_generation",
            "require_curriculum_generation",
        } & set(source_payload)
        if len(generation_keys) != 1:
            raise ValueError(
                f"data.sources.{source_id} must define exactly one generation contract"
            )
        require_generation = (
            _boolean(source_payload, "require_curriculum_generation")
            if "require_curriculum_generation" in source_payload
            else False
        )
        if "require_curriculum_generation" in source_payload and not require_generation:
            raise ValueError(
                f"data.sources.{source_id}.require_curriculum_generation must be true"
            )
        sources.append(
            DataSourceConfig(
                source_id=source_id,
                root=_resolve_project_path(
                    project_root, _string(source_payload, "root")
                ),
                control_modes=_string_tuple(source_payload, "control_modes"),
                fixed_generation=(
                    _integer(source_payload, "fixed_generation")
                    if "fixed_generation" in source_payload
                    else None
                ),
                require_curriculum_generation=require_generation,
            )
        )
    return tuple(sources)


def _parse_source_sampling_masses(
    data_payload: Mapping[str, object],
) -> dict[str, float]:
    raw_masses = data_payload.get("source_sampling_masses")
    if not isinstance(raw_masses, dict) or not raw_masses:
        raise ValueError("data.source_sampling_masses must be a non-empty mapping")
    masses: dict[str, float] = {}
    for source_id, value in raw_masses.items():
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(
                "data.source_sampling_masses keys must be non-empty strings"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"data.source_sampling_masses.{source_id} must be numeric"
            )
        masses[source_id] = float(value)
    return masses


def _parse_split_session_counts(
    data_payload: Mapping[str, object],
) -> dict[str, int]:
    raw_counts = data_payload.get("current_generation_session_counts")
    if not isinstance(raw_counts, dict) or set(raw_counts) != {"train", "val", "test"}:
        raise ValueError(
            "data.current_generation_session_counts must define train, val, and test"
        )
    counts: dict[str, int] = {}
    for split_name, value in raw_counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                "data.current_generation_session_counts values must be positive integers"
            )
        counts[split_name] = value
    return counts


def _string_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{key} must contain only non-empty strings")
    return tuple(value)


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
