from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

from xycar_ai.config import TrainConfig, load_train_config
from xycar_ai.front_cam_policy_data import (
    PolicyDataSplits,
    PolicySession,
    build_policy_data_splits,
    smooth_training_angle_targets,
)
from xycar_ai.front_cam_policy_model import build_policy_model
from xycar_ai.train_front_cam_policy import (
    CHECKPOINT_SCHEMA_VERSION,
    build_label_contract,
    build_preprocessing_contract,
    evaluate_policy,
    load_checkpoint,
    load_configured_road_warp,
    make_loaders,
    resolve_device,
    set_seed,
    sha256_file,
    source_weighted_metric,
    validation_selection_score,
)


@dataclass(frozen=True)
class PromotionThresholds:
    max_manual_angle_mae_regression: float = 2.0
    max_manual_within_10_drop: float = 0.03
    max_source_speed_mae_regression: float = 1.0


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    candidate_path = Path(args.candidate).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else candidate_path.parent / "promotion_gate.json"
    )
    report = compare_policy_checkpoints(
        config_path=Path(args.config),
        parent_path=Path(args.parent),
        candidate_path=candidate_path,
        output_path=output_path,
    )
    print(f"promotion_status={report['status']}")
    print(f"promotion_report={output_path}")
    return 0 if report["status"] == "passed" else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a Guided candidate and its parent on the same anchored "
            "validation split."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default="")
    return parser


def compare_policy_checkpoints(
    *,
    config_path: Path,
    parent_path: Path,
    candidate_path: Path,
    output_path: Path,
) -> dict[str, object]:
    config = load_train_config(config_path)
    _validate_promotion_config(config)
    parent_path = parent_path.expanduser().resolve()
    candidate_path = candidate_path.expanduser().resolve()
    for label, path in (("parent", parent_path), ("candidate", candidate_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint does not exist: {path}")

    device = resolve_device(config.training.device)
    parent_checkpoint = load_checkpoint(parent_path, device)
    candidate_checkpoint = load_checkpoint(candidate_path, device)
    road_warp = load_configured_road_warp(config)
    splits = smooth_training_angle_targets(
        build_policy_data_splits(config.data),
        config.data.train_angle_mean_window,
    )
    expected_split = splits.manifest(include_generation=True)
    label_contract = build_label_contract(config)
    _validate_checkpoint_contract(
        parent_checkpoint,
        label="parent",
        expected_generation=config.data.current_generation - 1,
        config=config,
        expected_split=None,
        expected_label_contract=label_contract,
    )
    _validate_checkpoint_contract(
        candidate_checkpoint,
        label="candidate",
        expected_generation=config.data.current_generation,
        config=config,
        expected_split=expected_split,
        expected_label_contract=label_contract,
    )

    set_seed(config.training.seed, deterministic=config.training.deterministic)
    model = build_policy_model(
        architecture=config.model.architecture,
        model_name=config.model.name,
        pretrained=False,
        image_size=config.model.image_size,
        history_frames=config.model.history_frames,
        control_token_type_embedding=config.model.control_token_type_embedding,
    ).to(device)
    preprocessing = build_preprocessing_contract(
        model.preprocessing_contract(),
        config=config,
        road_warp=road_warp,
    )
    for label, checkpoint in (
        ("parent", parent_checkpoint),
        ("candidate", candidate_checkpoint),
    ):
        if checkpoint.get("preprocessing") != preprocessing:
            raise ValueError(f"{label} checkpoint preprocessing differs from config")

    loaders = make_loaders(
        splits=splits,
        config=config,
        model_data_config=model.model_data_config,
        device=device,
        road_warp=road_warp,
    )
    angle_criterion = nn.CrossEntropyLoss(
        label_smoothing=config.loss.angle_label_smoothing
    )
    speed_criterion = nn.CrossEntropyLoss(
        label_smoothing=config.loss.speed_label_smoothing
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    parent_metrics = _evaluate_checkpoint(
        checkpoint=parent_checkpoint,
        model=model,
        splits=splits,
        config=config,
        loader=loaders["val"],
        angle_criterion=angle_criterion,
        speed_criterion=speed_criterion,
        scaler=scaler,
        device=device,
        road_warp=road_warp,
    )
    candidate_metrics = _evaluate_checkpoint(
        checkpoint=candidate_checkpoint,
        model=model,
        splits=splits,
        config=config,
        loader=loaders["val"],
        angle_criterion=angle_criterion,
        speed_criterion=speed_criterion,
        scaler=scaler,
        device=device,
        road_warp=road_warp,
    )

    parent_sha256 = sha256_file(parent_path)
    candidate_sha256 = sha256_file(candidate_path)
    initialization_sha256 = _candidate_initialization_sha256(candidate_checkpoint)
    report = build_promotion_report(
        generation=config.data.current_generation,
        parent_checkpoint=str(parent_path),
        parent_sha256=parent_sha256,
        candidate_checkpoint=str(candidate_path),
        candidate_sha256=candidate_sha256,
        initialization_sha256=initialization_sha256,
        parent_summary=_metric_summary(
            parent_metrics,
            sessions=splits.val_sessions,
            config=config,
        ),
        candidate_summary=_metric_summary(
            candidate_metrics,
            sessions=splits.val_sessions,
            config=config,
        ),
    )
    _write_json_atomic(output_path, report)
    return report


def build_promotion_report(
    *,
    generation: int,
    parent_checkpoint: str,
    parent_sha256: str,
    candidate_checkpoint: str,
    candidate_sha256: str,
    initialization_sha256: str,
    parent_summary: Mapping[str, float],
    candidate_summary: Mapping[str, float],
    thresholds: PromotionThresholds = PromotionThresholds(),
) -> dict[str, object]:
    checks = {
        "weighted_validation_score_improved": (
            candidate_summary["weighted_validation_score"]
            < parent_summary["weighted_validation_score"]
        ),
        "guided_angle_mae_improved": (
            candidate_summary["guided_angle_mae"]
            < parent_summary["guided_angle_mae"]
        ),
        "manual_angle_mae_regression_within_limit": (
            candidate_summary["manual_angle_mae"]
            <= parent_summary["manual_angle_mae"]
            + thresholds.max_manual_angle_mae_regression
        ),
        "manual_within_10_drop_within_limit": (
            candidate_summary["manual_angle_within_10_acc"]
            >= parent_summary["manual_angle_within_10_acc"]
            - thresholds.max_manual_within_10_drop
        ),
        "manual_speed_mae_regression_within_limit": (
            candidate_summary["manual_speed_mae"]
            <= parent_summary["manual_speed_mae"]
            + thresholds.max_source_speed_mae_regression
        ),
        "guided_speed_mae_regression_within_limit": (
            candidate_summary["guided_speed_mae"]
            <= parent_summary["guided_speed_mae"]
            + thresholds.max_source_speed_mae_regression
        ),
        "candidate_initialized_from_parent": (
            bool(initialization_sha256) and initialization_sha256 == parent_sha256
        ),
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "generation": generation,
        "thresholds": asdict(thresholds),
        "parent": {
            "checkpoint": parent_checkpoint,
            "sha256": parent_sha256,
            "metrics": dict(parent_summary),
        },
        "candidate": {
            "checkpoint": candidate_checkpoint,
            "sha256": candidate_sha256,
            "initialization_sha256": initialization_sha256,
            "metrics": dict(candidate_summary),
        },
        "checks": checks,
    }


def _evaluate_checkpoint(
    *,
    checkpoint: Mapping[str, object],
    model: torch.nn.Module,
    splits: PolicyDataSplits,
    config: TrainConfig,
    loader: torch.utils.data.DataLoader,
    angle_criterion: nn.Module,
    speed_criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    road_warp: object,
) -> dict[str, float]:
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint has no model_state")
    model.load_state_dict(model_state, strict=True)
    return evaluate_policy(
        model=model,
        loader=loader,
        sessions=splits.val_sessions,
        split_name="val",
        config=config,
        model_data_config=model.model_data_config,
        road_warp=road_warp,
        device=device,
        angle_criterion=angle_criterion,
        speed_criterion=speed_criterion,
        scaler=scaler,
        amp_enabled=bool(config.training.amp and device.type == "cuda"),
    )


def _metric_summary(
    metrics: Mapping[str, float],
    *,
    sessions: Sequence[PolicySession],
    config: TrainConfig,
) -> dict[str, float]:
    return {
        "weighted_validation_score": validation_selection_score(
            dict(metrics), sessions=sessions, config=config
        ),
        "manual_angle_mae": source_weighted_metric(
            metrics,
            sessions=sessions,
            config=config,
            source_id="manual",
            metric_name="angle_mae",
        ),
        "manual_angle_within_10_acc": source_weighted_metric(
            metrics,
            sessions=sessions,
            config=config,
            source_id="manual",
            metric_name="angle_within_10_acc",
        ),
        "manual_speed_mae": source_weighted_metric(
            metrics,
            sessions=sessions,
            config=config,
            source_id="manual",
            metric_name="speed_mae",
        ),
        "guided_angle_mae": source_weighted_metric(
            metrics,
            sessions=sessions,
            config=config,
            source_id="guided",
            metric_name="angle_mae",
        ),
        "guided_speed_mae": source_weighted_metric(
            metrics,
            sessions=sessions,
            config=config,
            source_id="guided",
            metric_name="speed_mae",
        ),
    }


def _validate_promotion_config(config: TrainConfig) -> None:
    if config.data.current_generation <= 0:
        raise ValueError("promotion comparison requires current_generation >= 1")
    if config.data.source_sampling_masses != {"manual": 0.5, "guided": 0.5}:
        raise ValueError(
            "promotion comparison requires manual/guided source masses of 0.5/0.5"
        )


def _validate_checkpoint_contract(
    checkpoint: Mapping[str, object],
    *,
    label: str,
    expected_generation: int,
    config: TrainConfig,
    expected_split: Mapping[str, object] | None,
    expected_label_contract: Mapping[str, object],
) -> None:
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"{label} checkpoint schema is incompatible")
    if checkpoint.get("model_name") != config.model.name:
        raise ValueError(f"{label} checkpoint model differs from config")
    checkpoint_config = checkpoint.get("config")
    checkpoint_data = (
        checkpoint_config.get("data")
        if isinstance(checkpoint_config, Mapping)
        else None
    )
    if not isinstance(checkpoint_data, Mapping):
        raise ValueError(f"{label} checkpoint data config is missing")
    if checkpoint_data.get("current_generation") != expected_generation:
        raise ValueError(
            f"{label} checkpoint must be generation {expected_generation}"
        )
    if (
        checkpoint_data.get("required_steering_contract")
        != config.data.required_steering_contract
    ):
        raise ValueError(f"{label} checkpoint steering contract differs from config")
    if checkpoint.get("label_contract") != expected_label_contract:
        raise ValueError(f"{label} checkpoint label contract differs from config")
    if expected_split is not None and checkpoint.get("split_manifest") != expected_split:
        raise ValueError(f"{label} checkpoint validation split differs from config")
    if expected_split is not None:
        expected_data = config.serializable()["data"]
        for field in (
            "generation_decay",
            "source_sampling_masses",
            "manual_anchor_split_manifest",
            "current_generation_session_counts",
        ):
            if checkpoint_data.get(field) != expected_data[field]:
                raise ValueError(
                    f"{label} checkpoint {field} differs from config"
                )


def _candidate_initialization_sha256(
    checkpoint: Mapping[str, object],
) -> str:
    source = checkpoint.get("source")
    initialization = source.get("initialization") if isinstance(source, Mapping) else None
    value = (
        initialization.get("checkpoint_sha256")
        if isinstance(initialization, Mapping)
        else None
    )
    return value if isinstance(value, str) else ""


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
