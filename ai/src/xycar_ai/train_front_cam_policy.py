from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from xycar_ai.compact_control import (
    ANGLE_OUTPUT_CLASSES,
    COMPACT_CONTROL_ENCODING,
    LEGACY_CONTROL_ENCODING,
    SPEED_OUTPUT_CLASSES,
    angle_target_class_id,
    executed_command_to_history_tokens,
    speed_target_class_id,
    unknown_history_pair,
)
from xycar_ai.config import TrainConfig, load_train_config
from xycar_ai.front_cam_policy_data import (
    COMMAND_MAX,
    COMMAND_MIN,
    NUM_COMMAND_CLASSES,
    FrontCamPolicyDataset,
    FrontCamPolicySequenceDataset,
    PolicyDataSplits,
    PolicySample,
    PolicySession,
    attach_constant_control_history,
    attach_executed_command_history,
    attach_unknown_control_history,
    attach_training_teacher_forced_history,
    build_policy_data_splits,
    compute_sqrt_inverse_frequency_weights,
    generation_epoch_sample_count,
    generation_sampling_summary,
    generation_sampling_weights,
    make_policy_transform,
    policy_dataset_stats,
    source_generation_sampling_masses,
    smooth_training_angle_targets,
    validate_session_initial_classes,
)
from xycar_ai.front_cam_policy_metrics import (
    ClassificationMetricAccumulator,
    RegressionMetricAccumulator,
    combine_policy_losses,
    ordinal_emd_loss,
    selection_score,
)
from xycar_ai.front_cam_policy_model import (
    AR_CONTROL_TOKEN_ARCHITECTURE,
    CATEGORICAL_PREDICTION_MODE,
    CONTINUOUS_REGRESSION_PREDICTION_MODE,
    AutoregressiveControlTokenViTPolicy,
    TaskTokenViTPolicy,
    build_policy_model,
)
from xycar_ai.front_cam_policy_warp import (
    ROAD_WARP_GEOMETRY,
    RoadWarpConfig,
    load_road_warp_config,
)

DEFAULT_CONFIG = "config/front_cam_policy_train.yaml"
CHECKPOINT_SCHEMA_VERSION = 1


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_train_config(args.config)
    if args.stop_after_epoch is not None and not (
        1 <= args.stop_after_epoch <= config.training.epochs
    ):
        raise ValueError("--stop-after-epoch must be in [1, training.epochs]")
    road_warp = load_configured_road_warp(config)
    splits = build_policy_data_splits(config.data)
    external_history = (
        config.model.architecture == AR_CONTROL_TOKEN_ARCHITECTURE
        and config.model.history_update == "externally_executed_commands"
    )
    sequence_rollout = config.training.sequence_length > 0
    unknown_frame_history = (
        config.training.history_training_source == "learned_unknown_tokens"
    )
    canonical_frame_history = (
        config.training.history_training_source == "canonical_initial_command"
        and not sequence_rollout
    )
    teacher_forced_frame_history = (
        config.training.history_training_source == "teacher_forced_executed_commands"
    )
    if unknown_frame_history:
        splits = attach_unknown_control_history(
            splits,
            config.model.history_frames,
        )
    elif canonical_frame_history:
        if config.model.control_encoding == COMPACT_CONTROL_ENCODING:
            initial_pair = executed_command_to_history_tokens(
                config.model.history_initial_angle,
                config.model.history_initial_speed,
                speed_max=config.model.speed_output_max,
            )
        else:
            initial_pair = (
                config.model.history_initial_angle + 100,
                config.model.history_initial_speed + 100,
            )
        splits = attach_constant_control_history(
            splits,
            config.model.history_frames,
            initial_pair,
        )
    elif teacher_forced_frame_history or (external_history and not sequence_rollout):
        splits = attach_executed_command_history(
            splits,
            config.model.history_frames,
            control_encoding=config.model.control_encoding,
            initial_command=(
                config.model.history_initial_angle,
                config.model.history_initial_speed,
            ),
            compact_speed_max=config.model.speed_output_max,
        )
    elif (
        config.model.architecture == AR_CONTROL_TOKEN_ARCHITECTURE
        and not external_history
        and not unknown_frame_history
    ):
        validate_session_initial_classes(
            splits,
            angle_class_id=config.model.history_initial_angle + 100,
            speed_class_id=config.model.history_initial_speed + 100,
        )
    splits = smooth_training_angle_targets(
        splits,
        config.data.train_angle_mean_window,
    )
    if (
        config.model.architecture == AR_CONTROL_TOKEN_ARCHITECTURE
        and not external_history
    ):
        splits = attach_training_teacher_forced_history(
            splits,
            config.model.history_frames,
        )
    split_manifest = splits.manifest(include_generation=config.data.ema_sampling)
    dataset_stats = policy_dataset_stats(
        splits,
        include_generation=config.data.ema_sampling,
    )
    if config.data.ema_sampling:
        dataset_stats["training_sampling"] = generation_sampling_summary(
            splits.train_samples,
            current_generation=config.data.current_generation,
            generation_decay=config.data.generation_decay,
            source_sampling_masses=config.data.source_sampling_masses,
        )

    if args.validate_only:
        print(json.dumps(dataset_stats, indent=2, sort_keys=True))
        print(
            "validated "
            f"sessions={dataset_stats['all']['session_count']} "
            f"samples={dataset_stats['all']['sample_count']}"
        )
        return 0

    validate_incremental_initialization(
        config,
        initialize_from=args.initialize_from,
        resume=args.resume,
    )

    set_seed(config.training.seed, deterministic=config.training.deterministic)
    device = resolve_device(config.training.device)
    amp_enabled = bool(config.training.amp and device.type == "cuda")
    model = build_policy_model(
        architecture=config.model.architecture,
        model_name=config.model.name,
        pretrained=config.model.pretrained
        and not bool(args.initialize_from or args.resume),
        image_size=config.model.image_size,
        history_frames=config.model.history_frames,
        control_token_type_embedding=config.model.control_token_type_embedding,
        control_encoding=config.model.control_encoding,
        prediction_mode=config.model.prediction_mode,
        speed_output_max=config.model.speed_output_max,
    ).to(device)
    initialization = initialize_model_weights(
        model=model,
        checkpoint=args.initialize_from,
        config=config,
        device=device,
    )
    preprocessing = build_preprocessing_contract(
        model.preprocessing_contract(),
        config=config,
        road_warp=road_warp,
    )
    label_contract = build_label_contract(config)
    loaders = make_loaders(
        splits=splits,
        config=config,
        model_data_config=model.model_data_config,
        device=device,
        road_warp=road_warp,
    )
    train_samples = splits.train_samples
    if config.model.prediction_mode == CONTINUOUS_REGRESSION_PREDICTION_MODE:
        angle_criterion = nn.SmoothL1Loss(beta=config.loss.angle_regression_beta)
        speed_criterion = nn.SmoothL1Loss(beta=config.loss.speed_regression_beta)
    else:
        angle_weights = class_weights(
            train_samples,
            field="angle_class_id",
            mode=config.loss.angle_class_weighting,
            config=config,
            device=device,
        )
        speed_weights = class_weights(
            train_samples,
            field="speed_class_id",
            mode=config.loss.speed_class_weighting,
            config=config,
            device=device,
        )
        angle_criterion = nn.CrossEntropyLoss(
            weight=angle_weights,
            label_smoothing=config.loss.angle_label_smoothing,
        )
        speed_criterion = nn.CrossEntropyLoss(
            weight=speed_weights,
            label_smoothing=config.loss.speed_label_smoothing,
        )
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=cosine_lr_factor(
            config.training.epochs, config.scheduler.warmup_epochs
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    run_dir, resume_payload = prepare_run_directory(
        config=config,
        resume=args.resume,
        device=device,
        expected_split=split_manifest,
        expected_preprocessing=preprocessing,
        expected_label_contract=label_contract,
    )
    write_yaml(run_dir / "resolved_config.yaml", config.serializable())
    write_json(run_dir / "split.json", split_manifest)
    write_json(run_dir / "dataset_stats.json", dataset_stats)

    start_epoch = 1
    best_score = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"])
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scheduler.load_state_dict(resume_payload["scheduler_state"])
        scaler_state = resume_payload.get("scaler_state")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        start_epoch = int(resume_payload["epoch"]) + 1
        best_score = float(resume_payload.get("best_score", math.inf))
        best_epoch = int(resume_payload.get("best_epoch", 0))
        epochs_without_improvement = int(
            resume_payload.get(
                "epochs_without_improvement",
                max(int(resume_payload["epoch"]) - best_epoch, 0),
            )
        )
    if start_epoch > config.training.epochs:
        raise ValueError(
            f"resume epoch {start_epoch - 1} already reaches configured "
            f"epochs={config.training.epochs}"
        )
    if args.stop_after_epoch is not None and args.stop_after_epoch < start_epoch:
        raise ValueError("--stop-after-epoch is before the resume start epoch")

    metrics_path = run_dir / "metrics.csv"
    metrics_rows = read_metrics_rows(metrics_path) if resume_payload else []
    source_state = collect_source_state(config.project_root)
    if initialization is not None:
        source_state["initialization"] = initialization
    for epoch in range(start_epoch, config.training.epochs + 1):
        if sequence_rollout:
            if not isinstance(model, AutoregressiveControlTokenViTPolicy):
                raise TypeError("sequence rollout requires an AR policy")
            sequence_loader = make_sequence_loader(
                sessions=splits.train_sessions,
                config=config,
                model_data_config=model.model_data_config,
                device=device,
                road_warp=road_warp,
                epoch=epoch,
            )
            train_metrics = run_sequence_rollout_epoch(
                model=model,
                loader=sequence_loader,
                split_name="train",
                config=config,
                device=device,
                angle_criterion=angle_criterion,
                speed_criterion=speed_criterion,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
            )
        else:
            train_metrics = run_epoch(
                model=model,
                loader=loaders["train"],
                split_name="train",
                device=device,
                angle_criterion=angle_criterion,
                speed_criterion=speed_criterion,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip=config.training.grad_clip,
                config=config,
            )
        val_metrics = evaluate_policy(
            model=model,
            loader=loaders["val"],
            sessions=splits.val_sessions,
            split_name="val",
            config=config,
            model_data_config=model.model_data_config,
            road_warp=road_warp,
            device=device,
            angle_criterion=angle_criterion,
            speed_criterion=speed_criterion,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        current_score = validation_selection_score(
            val_metrics,
            sessions=splits.val_sessions,
            config=config,
        )
        is_best = current_score < best_score
        if is_best:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        should_stop = early_stopping_triggered(
            epochs_without_improvement,
            config.training.early_stopping_patience,
        )
        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "lr": current_lr,
            "val_selection_score": current_score,
            "best_score": best_score,
            "epochs_without_improvement": epochs_without_improvement,
            "early_stopping_triggered": int(should_stop),
            **train_metrics,
            **val_metrics,
        }
        metrics_rows.append(row)
        write_metrics_csv(metrics_path, metrics_rows)

        checkpoint = checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            split_manifest=split_manifest,
            dataset_stats=dataset_stats,
            preprocessing=preprocessing,
            label_contract=label_contract,
            source_state=source_state,
            best_score=best_score,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            metrics=row,
        )
        atomic_torch_save(checkpoint, run_dir / "last.pt")
        if is_best:
            atomic_torch_save(checkpoint, run_dir / "best.pt")
        if should_stop:
            print(
                "early_stopping "
                f"epoch={epoch} patience={config.training.early_stopping_patience} "
                f"best_epoch={best_epoch} best_score={best_score:.4f}"
            )
            break
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            print(f"probe_stopped epoch={epoch}")
            break

    completed_epoch = int(metrics_rows[-1]["epoch"])
    if (
        args.stop_after_epoch is not None
        and args.stop_after_epoch < config.training.epochs
        and completed_epoch == args.stop_after_epoch
    ):
        write_json(
            run_dir / "probe_summary.json",
            {
                "schema_version": 1,
                "status": "probe_stopped",
                "completed_epochs": completed_epoch,
                "max_epochs": config.training.epochs,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "last_checkpoint": str(run_dir / "last.pt"),
                "source": source_state,
            },
        )
        print(f"run_dir={run_dir}")
        print(f"best_epoch={best_epoch} best_score={best_score:.4f}")
        return 0

    best_path = run_dir / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(f"best checkpoint is unavailable: {best_path}")
    best_payload = load_checkpoint(best_path, device)
    model.load_state_dict(best_payload["model_state"])
    test_metrics = evaluate_policy(
        model=model,
        loader=loaders["test"],
        sessions=splits.test_sessions,
        split_name="test",
        config=config,
        model_data_config=model.model_data_config,
        road_warp=road_warp,
        device=device,
        angle_criterion=angle_criterion,
        speed_criterion=speed_criterion,
        scaler=scaler,
        amp_enabled=amp_enabled,
    )
    write_json(run_dir / "test_metrics.json", test_metrics)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(run_dir / "last.pt"),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "completed_epochs": int(metrics_rows[-1]["epoch"]),
        "max_epochs": config.training.epochs,
        "early_stopping_patience": config.training.early_stopping_patience,
        "stopped_early": bool(metrics_rows[-1].get("early_stopping_triggered", 0)),
        "device": str(device),
        "amp": amp_enabled,
        "source": source_state,
        "dataset": split_manifest,
        "model_name": config.model.name,
        "model_architecture": config.model.architecture,
        "pretrained": config.model.pretrained,
        "preprocessing": preprocessing,
        "label_contract": label_contract,
        "training_objective": build_training_objective_contract(config),
        "test_metrics": test_metrics,
    }
    write_json(run_dir / "summary.json", summary)
    print(f"run_dir={run_dir}")
    print(f"best_epoch={best_epoch} best_score={best_score:.4f}")
    print(
        f"test_angle_mae={test_metrics['test_angle_mae']:.4f} "
        f"test_speed_mae={test_metrics['test_speed_mae']:.4f}"
    )
    return 0


def build_label_contract(config: TrainConfig) -> dict[str, object]:
    if config.model.prediction_mode == CONTINUOUS_REGRESSION_PREDICTION_MODE:
        return _build_regression_label_contract(config)
    if config.model.control_encoding == COMPACT_CONTROL_ENCODING:
        canonical_initialization = (
            config.training.history_training_source == "canonical_initial_command"
        )
        unknown_initialization = not canonical_initialization and bool(
            config.training.sequence_length
            or config.training.history_training_source == "learned_unknown_tokens"
        )
        compact_initial_pair = (
            unknown_history_pair()
            if unknown_initialization
            else executed_command_to_history_tokens(
                config.model.history_initial_angle,
                config.model.history_initial_speed,
            )
        )
        compact_initialization = (
            "learned_unknown_tokens"
            if unknown_initialization
            else "canonical_initial_command"
        )
        contract: dict[str, object] = {
            "schema_version": 3,
            "control_encoding": COMPACT_CONTROL_ENCODING,
            "output_keys": ["angle_logits", "speed_logits"],
            "output_shapes": {
                "angle_logits": [1, ANGLE_OUTPUT_CLASSES],
                "speed_logits": [1, SPEED_OUTPUT_CLASSES],
            },
            "angle": {
                "num_classes": ANGLE_OUTPUT_CLASSES,
                "driver_range": [-50, 50],
                "class_id_mapping": (
                    "round(clamp(normalized_angle * 0.5, -50, 50)) + 50"
                ),
                "decode_driver_mapping": "class_id - 50",
                "decode_normalized_mapping": "(class_id - 50) / 0.5",
            },
            "speed": {
                "num_classes": SPEED_OUTPUT_CLASSES,
                "command_range": [0, 30],
                "class_id_mapping": "round(clamp(speed, 0, 30))",
                "decode_mapping": "class_id",
            },
            "shared_numeric_vocabulary": {
                "numeric_range": [-50, 50],
                "numeric_token_mapping": "value + 50",
                "numeric_token_ids": [0, 100],
                "unknown_angle_token_id": 101,
                "unknown_speed_token_id": 102,
                "angle_query_token_id": 103,
                "speed_query_token_id": 104,
                "vocabulary_size": 105,
            },
            "train_angle_target": {
                "method": "centered_mean",
                "window_size": config.data.train_angle_mean_window,
                "padding": "repeat_session_edge",
                "applied_splits": ["train"],
                "average_before_quantization": True,
                "speed_target": "unchanged",
            },
            "horizontal_flip_mapping": {
                "angle_raw": "-angle_raw",
                "angle": "-angle",
                "angle_class_id": "100 - angle_class_id",
                "angle_history_token_id": "100 - token_id",
                "unknown_angle_token_id": "unchanged",
                "speed": "unchanged",
                "speed_class_id": "unchanged",
                "speed_history_token_id": "unchanged",
            },
        }
        if config.model.architecture == AR_CONTROL_TOKEN_ARCHITECTURE:
            history_contract: dict[str, object] = {
                "frames": config.model.history_frames,
                "shape": [config.model.history_frames, 2],
                "input_name": "history_token_ids",
                "pair_order": ["angle_token_id", "speed_token_id"],
                "time_order": "oldest_to_newest",
                "initialization": compact_initialization,
                "initial_command": [
                    config.model.history_initial_angle,
                    config.model.history_initial_speed,
                ],
                "initial_token_ids": list(compact_initial_pair),
                "actual_angle_token_range": [0, 100],
                "actual_speed_token_range": [50, 80],
                "update": "externally_executed_commands",
                "known_train_label_leakage": False,
            }
            if config.training.sequence_length:
                fixed_speed = config.training.sequence_rollout_fixed_speed
                prediction_execution: dict[str, object] = {
                    "angle_output_class_clamp": [0, 100],
                    "speed_output_class_clamp": [0, 30],
                    "angle_history_token_mapping": "class_id",
                    "speed_history_token_mapping": "class_id + 50",
                    "gradient": "detached_before_history_update",
                }
                if fixed_speed is not None:
                    prediction_execution.update(
                        {
                            "speed_source": "fixed_command",
                            "fixed_speed": fixed_speed,
                        }
                    )
                history_contract.update(
                    {
                        "train_source": (
                            "self_predicted_argmax_sequence_rollout"
                            if fixed_speed is None
                            else "self_predicted_angle_fixed_speed_sequence_rollout"
                        ),
                        "train_prediction_execution": prediction_execution,
                        "sequence_length": config.training.sequence_length,
                        "sequence_boundaries": (
                            "session_start_reset_non_overlapping_compute_chunks_"
                            "carry_predicted_history"
                        ),
                        "sequence_reverse_probability": 0.0,
                        "evaluation_source": (
                            "predicted_argmax_rollout"
                            if fixed_speed is None
                            else "predicted_argmax_angle_fixed_speed_rollout"
                        ),
                        "session_history_initialization": compact_initialization,
                        "compute_chunk_resets_history": False,
                        "augmentation_scope": "whole_session_per_epoch",
                        "edge_padding": "masked_repeat_last_frame",
                        "teacher_forcing": False,
                        "scheduled_sampling": False,
                    }
                )
            elif unknown_initialization:
                history_contract.update(
                    {
                        "train_source": "learned_unknown_tokens",
                        "evaluation_source": "learned_unknown_tokens",
                        "frame_pretraining": True,
                        "labels_used_as_history": False,
                        "edge_padding": "not_applicable",
                    }
                )
            elif canonical_initialization:
                history_contract.update(
                    {
                        "train_source": "canonical_initial_command",
                        "evaluation_source": "canonical_initial_command",
                        "frame_pretraining": True,
                        "labels_used_as_history": False,
                        "edge_padding": "not_applicable",
                    }
                )
            elif (
                config.training.history_training_source
                == "teacher_forced_executed_commands"
            ):
                history_contract.update(
                    {
                        "train_source": (
                            "ground_truth_teacher_forcing_previous_executed_commands"
                        ),
                        "train_angle_source": "previous_raw_executed_command",
                        "train_speed_source": "previous_raw_executed_command",
                        "evaluation_source": "previous_actual_executed_commands",
                        "edge_padding": "canonical_initial_command",
                        "current_target_used_as_history": False,
                        "future_target_used_as_history": False,
                    }
                )
            else:
                history_contract.update(
                    {
                        "train_source": "previous_actual_executed_commands",
                        "evaluation_source": "previous_actual_executed_commands",
                        "edge_padding": (
                            "session_metadata_or_canonical_initial_command"
                        ),
                    }
                )
            contract["history"] = history_contract
        return contract

    contract: dict[str, object] = {
        "schema_version": 1,
        "output_keys": ["angle_logits", "speed_logits"],
        "num_classes": NUM_COMMAND_CLASSES,
        "command_min": COMMAND_MIN,
        "command_max": COMMAND_MAX,
        "class_id_mapping": "int(round(clamp(value, -100, 100))) + 100",
        "decode_mapping": "class_id - 100",
        "train_angle_target": {
            "method": "centered_mean",
            "window_size": config.data.train_angle_mean_window,
            "padding": "repeat_session_edge",
            "applied_splits": ["train"],
            "average_before_quantization": True,
            "speed_target": "unchanged",
        },
        "horizontal_flip_mapping": {
            "angle_raw": "-angle_raw",
            "angle": "-angle",
            "angle_class_id": "200 - angle_class_id",
            "speed": "unchanged",
            "speed_class_id": "unchanged",
        },
    }
    if config.model.architecture == AR_CONTROL_TOKEN_ARCHITECTURE:
        history_contract: dict[str, object] = {
            "frames": config.model.history_frames,
            "shape": [config.model.history_frames, 2],
            "pair_order": ["angle_class_id", "speed_class_id"],
            "time_order": "oldest_to_newest",
            "initial_command": [
                config.model.history_initial_angle,
                config.model.history_initial_speed,
            ],
            "initial_class_ids": [
                config.model.history_initial_angle + 100,
                config.model.history_initial_speed + 100,
            ],
        }
        if config.training.sequence_length:
            fixed_speed = config.training.sequence_rollout_fixed_speed
            prediction_execution = {
                "angle_class_clamp": [0, 200],
                "speed_class_clamp": [100, 200],
                "gradient": "detached_before_history_update",
            }
            if fixed_speed is not None:
                prediction_execution.update(
                    {
                        "speed_source": "fixed_command",
                        "fixed_speed": fixed_speed,
                    }
                )
            history_contract.update(
                {
                    "update": "externally_executed_commands",
                    "train_source": (
                        "self_predicted_argmax_sequence_rollout"
                        if fixed_speed is None
                        else "self_predicted_angle_fixed_speed_sequence_rollout"
                    ),
                    "train_prediction_execution": prediction_execution,
                    "sequence_length": config.training.sequence_length,
                    "sequence_boundaries": (
                        "session_start_reset_non_overlapping_compute_chunks_"
                        "carry_predicted_history"
                    ),
                    "sequence_reverse_probability": (
                        config.training.sequence_reverse_probability
                    ),
                    "evaluation_source": (
                        "predicted_argmax_rollout"
                        if fixed_speed is None
                        else "predicted_argmax_angle_fixed_speed_rollout"
                    ),
                    "session_history_initialization": ("canonical_initial_command"),
                    "compute_chunk_resets_history": False,
                    "augmentation_scope": "whole_session_per_epoch",
                    "edge_padding": "masked_repeat_last_frame",
                    "known_train_label_leakage": False,
                }
            )
        elif config.training.history_training_source == "canonical_initial_command":
            history_contract.update(
                {
                    "update": "externally_executed_commands",
                    "train_source": "canonical_initial_command",
                    "evaluation_source": "canonical_initial_command",
                    "frame_pretraining": True,
                    "labels_used_as_history": False,
                    "edge_padding": "not_applicable",
                    "known_train_label_leakage": False,
                }
            )
        elif config.model.history_update == "externally_executed_commands":
            history_contract.update(
                {
                    "update": "externally_executed_commands",
                    "train_source": "previous_actual_executed_commands",
                    "train_angle_source": "raw_executed_command",
                    "train_speed_source": "raw_executed_command",
                    "evaluation_source": "previous_actual_executed_commands",
                    "edge_padding": "session_metadata_or_canonical_initial_command",
                    "known_train_label_leakage": False,
                }
            )
        else:
            history_contract.update(
                {
                    "update": "predicted_argmax",
                    "train_source": "ground_truth_teacher_forcing",
                    "train_angle_source": "centered_mean_target",
                    "train_speed_source": "instantaneous_target",
                    "evaluation_source": "predicted_argmax_rollout",
                    "edge_padding": "repeat_session_first_target",
                    "known_train_label_leakage": True,
                }
            )
        contract["history"] = history_contract
    return contract


def _build_regression_label_contract(config: TrainConfig) -> dict[str, object]:
    if config.model.control_encoding != COMPACT_CONTROL_ENCODING:
        raise ValueError("continuous regression requires compact control encoding")
    initial_pair = executed_command_to_history_tokens(
        config.model.history_initial_angle,
        config.model.history_initial_speed,
        speed_max=config.model.speed_output_max,
    )
    sequence_rollout = config.training.sequence_length > 0
    history_contract: dict[str, object] = {
        "frames": config.model.history_frames,
        "shape": [config.model.history_frames, 2],
        "input_name": "history_token_ids",
        "pair_order": ["angle_token_id", "speed_token_id"],
        "time_order": "oldest_to_newest",
        "initialization": "canonical_initial_command",
        "initial_command": [
            config.model.history_initial_angle,
            config.model.history_initial_speed,
        ],
        "initial_token_ids": list(initial_pair),
        "actual_angle_token_range": [0, 100],
        "actual_speed_token_range": [
            50,
            int(config.model.speed_output_max) + 50,
        ],
        "update": "externally_executed_commands",
        "known_train_label_leakage": False,
    }
    if sequence_rollout:
        history_contract.update(
            {
                "train_source": "self_predicted_regression_sequence_rollout",
                "train_prediction_execution": {
                    "angle_driver_clamp": [-50.0, 50.0],
                    "speed_clamp": [0.0, config.model.speed_output_max],
                    "angle_history_token_mapping": "round(angle_driver) + 50",
                    "speed_history_token_mapping": "round(speed) + 50",
                    "gradient": "clamp_round_detach_before_history_update",
                },
                "sequence_length": config.training.sequence_length,
                "sequence_boundaries": (
                    "session_start_reset_non_overlapping_compute_chunks_"
                    "carry_predicted_history"
                ),
                "sequence_reverse_probability": 0.0,
                "evaluation_source": "predicted_regression_full_session_rollout",
                "session_history_initialization": "canonical_initial_command",
                "compute_chunk_resets_history": False,
                "augmentation_scope": "whole_session_per_epoch",
                "edge_padding": "masked_repeat_last_frame",
                "teacher_forcing": False,
                "scheduled_sampling": False,
            }
        )
    else:
        history_contract.update(
            {
                "train_source": "canonical_initial_command",
                "evaluation_source": "canonical_initial_command",
                "frame_pretraining": True,
                "labels_used_as_history": False,
                "edge_padding": "not_applicable",
            }
        )
    return {
        "schema_version": 4,
        "control_encoding": COMPACT_CONTROL_ENCODING,
        "prediction_mode": CONTINUOUS_REGRESSION_PREDICTION_MODE,
        "output_keys": ["angle_driver", "speed"],
        "output_shapes": {"angle_driver": [1, 1], "speed": [1, 1]},
        "angle": {
            "dtype": "float32",
            "unit": "driver_angle",
            "range": [-50.0, 50.0],
            "activation": "tanh_times_50",
            "runtime_normalized_mapping": "angle_driver * 2",
        },
        "speed": {
            "dtype": "float32",
            "unit": "motor_speed",
            "range": [0.0, config.model.speed_output_max],
            "activation": f"sigmoid_times_{config.model.speed_output_max:g}",
        },
        "train_angle_target": {
            "method": "centered_mean",
            "window_size": config.data.train_angle_mean_window,
            "padding": "repeat_session_edge",
            "applied_splits": ["train"],
            "average_before_driver_scale": True,
            "driver_scale": 0.5,
            "validation_test_target": "raw_continuous",
            "speed_target": "raw_continuous_unchanged",
        },
        "history": history_contract,
    }


def build_training_objective_contract(config: TrainConfig) -> dict[str, object]:
    speed_output_trained = config.loss.speed_loss_weight > 0
    angle_only = (
        not speed_output_trained and config.training.validation_speed_mae_weight == 0
    )
    objective: dict[str, object] = {
        "mode": "angle_only" if angle_only else "joint_angle_speed",
        "speed_output_trained": speed_output_trained,
        "speed_loss_weight": config.loss.speed_loss_weight,
        "validation_speed_mae_weight": (config.training.validation_speed_mae_weight),
    }
    if config.model.prediction_mode == CONTINUOUS_REGRESSION_PREDICTION_MODE:
        objective.update(
            {
                "mode": "joint_angle_speed_regression",
                "loss": "smooth_l1_normalized",
                "angle_normalization": 50.0,
                "speed_normalization": config.model.speed_output_max,
                "angle_beta": config.loss.angle_regression_beta,
                "speed_beta": config.loss.speed_regression_beta,
            }
        )
    return objective


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Xycar front-camera angle/speed policy."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", default="")
    checkpoint_group.add_argument("--initialize-from", default="")
    parser.add_argument("--stop-after-epoch", type=int)
    return parser


def validate_incremental_initialization(
    config: TrainConfig, *, initialize_from: str, resume: str
) -> None:
    if not config.data.sources:
        return
    generation = config.data.current_generation
    if generation == 0 and initialize_from:
        raise ValueError(
            "stateless generation 0 must start from pretrained ImageNet weights"
        )
    if generation > 0 and not initialize_from and not resume:
        raise ValueError(
            "stateless generation 1+ requires --initialize-from previous best.pt"
        )


def make_loaders(
    *,
    splits: PolicyDataSplits,
    config: TrainConfig,
    model_data_config: Mapping[str, object],
    device: torch.device,
    road_warp: RoadWarpConfig | None = None,
) -> dict[str, DataLoader]:
    groups = {
        "train": splits.train_samples,
        "val": splits.val_samples,
        "test": splits.test_samples,
    }
    loaders: dict[str, DataLoader] = {}
    for offset, (split_name, samples) in enumerate(groups.items()):
        training = split_name == "train"
        dataset = FrontCamPolicyDataset(
            samples,
            transform=make_policy_transform(
                train=training,
                image_size=config.model.image_size,
                model_data_config=model_data_config,
                augmentation=config.augmentation,
            ),
            horizontal_flip_probability=(
                config.augmentation.horizontal_flip_probability if training else 0.0
            ),
            road_warp=road_warp,
            control_encoding=config.model.control_encoding,
        )
        generator = torch.Generator()
        generator.manual_seed(config.training.seed + offset)
        sampler: WeightedRandomSampler | None = None
        if training and config.data.ema_sampling:
            sampler = WeightedRandomSampler(
                generation_sampling_weights(
                    samples,
                    current_generation=config.data.current_generation,
                    generation_decay=config.data.generation_decay,
                    source_sampling_masses=config.data.source_sampling_masses,
                ),
                num_samples=generation_epoch_sample_count(
                    samples,
                    current_generation=config.data.current_generation,
                    generation_decay=config.data.generation_decay,
                    source_sampling_masses=config.data.source_sampling_masses,
                ),
                replacement=True,
                generator=generator,
            )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=training and sampler is None,
            sampler=sampler,
            num_workers=config.data.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.data.num_workers > 0,
            generator=generator,
            worker_init_fn=seed_worker,
        )
    return loaders


def make_sequence_loader(
    *,
    sessions: Sequence[PolicySession],
    config: TrainConfig,
    model_data_config: Mapping[str, object],
    device: torch.device,
    road_warp: RoadWarpConfig | None = None,
    epoch: int = 1,
) -> DataLoader:
    sequence_length = config.training.sequence_length
    if sequence_length <= 0:
        raise ValueError("sequence_length must be configured")
    dataset = FrontCamPolicySequenceDataset(
        sessions,
        sequence_length=sequence_length,
        transform=make_policy_transform(
            train=False,
            image_size=config.model.image_size,
            model_data_config=model_data_config,
            augmentation=config.augmentation,
        ),
        horizontal_flip_probability=(config.augmentation.horizontal_flip_probability),
        sequence_reverse_probability=(config.training.sequence_reverse_probability),
        road_warp=road_warp,
        control_encoding=config.model.control_encoding,
        augmentation=config.augmentation,
        augmentation_seed=config.training.seed + epoch * 100_003,
    )
    generator = torch.Generator()
    generator.manual_seed(config.training.seed + epoch * 100_019)
    max_batch_size = max(1, config.training.batch_size // sequence_length)
    return DataLoader(
        dataset,
        batch_sampler=dataset.session_ordered_batches(
            max_batch_size=max_batch_size,
            seed=config.training.seed + epoch * 100_043,
        ),
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        generator=generator,
        worker_init_fn=seed_worker,
    )


def evaluate_policy(
    *,
    model: TaskTokenViTPolicy | AutoregressiveControlTokenViTPolicy,
    loader: DataLoader,
    sessions: Sequence[PolicySession],
    split_name: str,
    config: TrainConfig,
    model_data_config: Mapping[str, object],
    road_warp: RoadWarpConfig | None,
    device: torch.device,
    angle_criterion: nn.Module,
    speed_criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> dict[str, float]:
    if isinstance(model, AutoregressiveControlTokenViTPolicy) and (
        config.model.history_update == "predicted_argmax"
        or config.training.sequence_length > 0
    ):
        return run_rollout_evaluation(
            model=model,
            sessions=sessions,
            split_name=split_name,
            config=config,
            model_data_config=model_data_config,
            road_warp=road_warp,
            device=device,
            angle_criterion=angle_criterion,
            speed_criterion=speed_criterion,
            amp_enabled=amp_enabled,
        )
    return run_epoch(
        model=model,
        loader=loader,
        split_name=split_name,
        device=device,
        angle_criterion=angle_criterion,
        speed_criterion=speed_criterion,
        optimizer=None,
        scaler=scaler,
        amp_enabled=amp_enabled,
        grad_clip=config.training.grad_clip,
        config=config,
    )


def run_rollout_evaluation(
    *,
    model: AutoregressiveControlTokenViTPolicy,
    sessions: Sequence[PolicySession],
    split_name: str,
    config: TrainConfig,
    model_data_config: Mapping[str, object],
    road_warp: RoadWarpConfig | None,
    device: torch.device,
    angle_criterion: nn.Module,
    speed_criterion: nn.Module,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    accumulator = _metric_accumulator(split_name, config)
    progress = tqdm(
        total=sum(len(session.samples) for session in sessions),
        desc=split_name,
        leave=False,
    )
    initial_pair = (
        list(
            executed_command_to_history_tokens(
                config.model.history_initial_angle,
                config.model.history_initial_speed,
                speed_max=config.model.speed_output_max,
            )
            if config.training.history_training_source == "canonical_initial_command"
            else unknown_history_pair()
        )
        if config.model.control_encoding == COMPACT_CONTROL_ENCODING
        else [
            config.model.history_initial_angle + 100,
            config.model.history_initial_speed + 100,
        ]
    )
    transform = make_policy_transform(
        train=False,
        image_size=config.model.image_size,
        model_data_config=model_data_config,
        augmentation=config.augmentation,
    )
    with torch.inference_mode():
        for session in sessions:
            history = torch.tensor(
                [[initial_pair] * config.model.history_frames],
                dtype=torch.long,
                device=device,
            )
            dataset = FrontCamPolicyDataset(
                session.samples,
                transform=transform,
                horizontal_flip_probability=0.0,
                road_warp=road_warp,
                control_encoding=config.model.control_encoding,
            )
            for item in dataset:
                images = (
                    item["image_tensor"]
                    .unsqueeze(0)
                    .to(
                        device=device,
                        non_blocking=True,
                    )
                )
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    outputs = model(images, history)
                    total_loss, angle_loss, speed_loss, emd_loss = (
                        compute_policy_losses(
                            outputs=outputs,
                            batch=item,
                            config=config,
                            angle_criterion=angle_criterion,
                            speed_criterion=speed_criterion,
                        )
                    )
                accumulator.update(
                    outputs={key: value.detach() for key, value in outputs.items()},
                    batch=item,
                    total_loss=total_loss.detach(),
                    angle_loss=angle_loss.detach(),
                    speed_loss=speed_loss.detach(),
                    emd_loss=emd_loss.detach(),
                )
                predicted_pair = predicted_executed_class_pair(
                    outputs,
                    control_encoding=config.model.control_encoding,
                    fixed_speed=config.training.sequence_rollout_fixed_speed,
                    speed_max=config.model.speed_output_max,
                )
                history = torch.cat(
                    (history[:, 1:], predicted_pair.unsqueeze(1)),
                    dim=1,
                )
                progress.update(1)
                current = accumulator.compute()
                progress.set_postfix(
                    loss=f"{current[f'{split_name}_loss']:.3f}",
                    angle_mae=f"{current[f'{split_name}_angle_mae']:.2f}",
                )
    progress.close()
    return accumulator.compute()


def predicted_executed_class_pair(
    outputs: Mapping[str, torch.Tensor],
    *,
    control_encoding: str = LEGACY_CONTROL_ENCODING,
    fixed_speed: int | None = None,
    speed_max: float = 30.0,
) -> torch.Tensor:
    """Decode the class pair that the no-reverse runtime can publish."""
    if "angle_driver" in outputs or "speed" in outputs:
        if set(outputs) != {"angle_driver", "speed"}:
            raise ValueError("regression outputs must contain angle_driver and speed")
        if fixed_speed is not None:
            raise ValueError("regression rollout cannot use fixed speed")
        angle_driver = outputs["angle_driver"].reshape(-1)
        speed = outputs["speed"].reshape(-1)
        if angle_driver.shape != speed.shape:
            raise ValueError("regression output batch sizes must match")
        return torch.stack(
            (
                angle_driver.clamp(-50.0, 50.0).round().long() + 50,
                speed.clamp(0.0, speed_max).round().long() + 50,
            ),
            dim=1,
        ).detach()
    if control_encoding == COMPACT_CONTROL_ENCODING:
        angle_class = outputs["angle_logits"].argmax(dim=1).clamp(0, 100)
        speed_class = (
            outputs["speed_logits"].argmax(dim=1).clamp(0, 30)
            if fixed_speed is None
            else torch.full_like(angle_class, fixed_speed)
        )
        return torch.stack(
            (angle_class, speed_class + 50),
            dim=1,
        ).detach()
    angle_class = outputs["angle_logits"].argmax(dim=1).clamp(0, 200)
    speed_class = (
        outputs["speed_logits"].argmax(dim=1).clamp(100, 200)
        if fixed_speed is None
        else torch.full_like(angle_class, fixed_speed + 100)
    )
    return torch.stack((angle_class, speed_class), dim=1).detach()


def rollout_predicted_histories(
    *,
    model: AutoregressiveControlTokenViTPolicy,
    images: torch.Tensor,
    valid_mask: torch.Tensor,
    initial_history: torch.Tensor,
    amp_enabled: bool,
    fixed_speed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect detached self-predicted prompts for one session-local clip."""
    if images.ndim != 5:
        raise ValueError("sequence images must have shape [B,T,C,H,W]")
    if valid_mask.shape != images.shape[:2]:
        raise ValueError("valid_mask must have shape [B,T]")
    expected_history_shape = (images.shape[0], model.history_frames, 2)
    if initial_history.shape != expected_history_shape:
        raise ValueError(f"initial_history must have shape {expected_history_shape}")
    history = initial_history.clone()
    prompts: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for frame_index in range(images.shape[1]):
                prompts.append(history.clone())
                with torch.autocast(
                    device_type=images.device.type,
                    enabled=amp_enabled,
                ):
                    outputs = model(images[:, frame_index], history)
                predicted_pair = predicted_executed_class_pair(
                    outputs,
                    control_encoding=getattr(
                        model,
                        "control_encoding",
                        LEGACY_CONTROL_ENCODING,
                    ),
                    fixed_speed=fixed_speed,
                    speed_max=getattr(model, "speed_output_max", 30.0),
                )
                updated = torch.cat(
                    (history[:, 1:], predicted_pair.unsqueeze(1)), dim=1
                )
                active = valid_mask[:, frame_index].reshape(-1, 1, 1)
                history = torch.where(active, updated, history)
    finally:
        model.train(was_training)
    return torch.stack(prompts, dim=1), history


def run_sequence_rollout_epoch(
    *,
    model: AutoregressiveControlTokenViTPolicy,
    loader: DataLoader,
    split_name: str,
    config: TrainConfig,
    device: torch.device,
    angle_criterion: nn.Module,
    speed_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> dict[str, float]:
    """Train on detached predictions carried across ordered session chunks."""
    model.train()
    accumulator = _metric_accumulator(split_name, config)
    reversed_clips = 0
    clip_count = 0
    history_reset_count = 0
    carried_chunk_count = 0
    seen_sessions: set[str] = set()
    active_histories: dict[str, torch.Tensor] = {}
    progress = tqdm(loader, desc=split_name, leave=False)
    initial_pair = torch.tensor(
        (
            (
                executed_command_to_history_tokens(
                    config.model.history_initial_angle,
                    config.model.history_initial_speed,
                    speed_max=config.model.speed_output_max,
                )
                if config.training.history_training_source
                == "canonical_initial_command"
                else unknown_history_pair()
            )
            if config.model.control_encoding == COMPACT_CONTROL_ENCODING
            else (
                config.model.history_initial_angle + 100,
                config.model.history_initial_speed + 100,
            )
        ),
        dtype=torch.long,
        device=device,
    )
    for batch in progress:
        images = batch["image_tensor"].to(device=device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device=device, non_blocking=True)
        session_ids = [str(value) for value in batch["session_id"]]
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("one sequence batch contains duplicate session IDs")
        starts_session = batch["starts_session"].tolist()
        ends_session = batch["ends_session"].tolist()
        chunk_indices = batch["chunk_index"].tolist()
        histories: list[torch.Tensor] = []
        for session_id, starts, chunk_index in zip(
            session_ids,
            starts_session,
            chunk_indices,
            strict=True,
        ):
            if starts:
                if chunk_index != 0 or session_id in seen_sessions:
                    raise ValueError(
                        f"invalid repeated sequence start for {session_id}"
                    )
                history = initial_pair.reshape(1, 2).expand(
                    config.model.history_frames, 2
                )
                seen_sessions.add(session_id)
                history_reset_count += 1
            else:
                history = active_histories.get(session_id)
                if history is None:
                    raise ValueError(
                        f"missing carried sequence history for {session_id} "
                        f"chunk {chunk_index}"
                    )
                carried_chunk_count += 1
            histories.append(history)
        initial_history = torch.stack(histories)
        history_prompts, final_histories = rollout_predicted_histories(
            model=model,
            images=images,
            valid_mask=valid_mask,
            initial_history=initial_history,
            amp_enabled=amp_enabled,
            fixed_speed=config.training.sequence_rollout_fixed_speed,
        )
        for session_id, ends, final_history in zip(
            session_ids,
            ends_session,
            final_histories,
            strict=True,
        ):
            if ends:
                active_histories.pop(session_id, None)
            else:
                active_histories[session_id] = final_history.detach().clone()
        flat_images = images[valid_mask]
        flat_history = history_prompts[valid_mask]
        metric_batch = {
            key: value.to(device=device, non_blocking=True)[valid_mask]
            for key, value in batch.items()
            if key
            in {
                "angle",
                "speed",
                "angle_class_id",
                "speed_class_id",
                "angle_raw",
                "speed_raw",
                "horizontal_flipped",
                "generation",
            }
        }

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(flat_images, flat_history)
            total_loss, angle_loss, speed_loss, emd_loss = compute_policy_losses(
                outputs=outputs,
                batch=metric_batch,
                config=config,
                angle_criterion=angle_criterion,
                speed_criterion=speed_criterion,
            )
        scaler.scale(total_loss).backward()
        if config.training.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.grad_clip
            )
        scaler.step(optimizer)
        scaler.update()

        accumulator.update(
            outputs={key: value.detach() for key, value in outputs.items()},
            batch=metric_batch,
            total_loss=total_loss.detach(),
            angle_loss=angle_loss.detach(),
            speed_loss=speed_loss.detach(),
            emd_loss=emd_loss.detach(),
        )
        reversed_batch = batch["sequence_reversed"]
        reversed_clips += int(reversed_batch.sum())
        clip_count += int(reversed_batch.numel())
        current = accumulator.compute()
        progress.set_postfix(
            loss=f"{current[f'{split_name}_loss']:.3f}",
            angle_mae=f"{current[f'{split_name}_angle_mae']:.2f}",
        )
    metrics = accumulator.compute()
    if active_histories:
        raise ValueError(
            "sequence epoch ended with unfinished sessions: "
            + ", ".join(sorted(active_histories))
        )
    metrics[f"{split_name}_sequence_clip_count"] = float(clip_count)
    metrics[f"{split_name}_sequence_session_count"] = float(len(seen_sessions))
    metrics[f"{split_name}_sequence_history_reset_count"] = float(history_reset_count)
    metrics[f"{split_name}_sequence_carried_chunk_count"] = float(carried_chunk_count)
    metrics[f"{split_name}_sequence_reverse_rate"] = reversed_clips / max(clip_count, 1)
    return metrics


def _metric_accumulator(
    split_name: str,
    config: TrainConfig,
) -> ClassificationMetricAccumulator | RegressionMetricAccumulator:
    if config.model.prediction_mode == CONTINUOUS_REGRESSION_PREDICTION_MODE:
        return RegressionMetricAccumulator(split_name)
    return ClassificationMetricAccumulator(
        split_name,
        control_encoding=config.model.control_encoding,
    )


def compute_policy_losses(
    *,
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, object],
    config: TrainConfig,
    angle_criterion: nn.Module,
    speed_criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if config.model.prediction_mode == CONTINUOUS_REGRESSION_PREDICTION_MODE:
        angle_prediction = outputs["angle_driver"].reshape(-1)
        speed_prediction = outputs["speed"].reshape(-1)
        device = angle_prediction.device
        angle_target = _batch_tensor(batch["angle_raw"], device).float().reshape(-1)
        speed_target = _batch_tensor(batch["speed_raw"], device).float().reshape(-1)
        if angle_prediction.shape != angle_target.shape:
            raise ValueError("regression angle output shape must match the target")
        if speed_prediction.shape != speed_target.shape:
            raise ValueError("regression speed output shape must match the target")
        angle_loss = angle_criterion(
            angle_prediction / 50.0,
            angle_target / 100.0,
        )
        speed_loss = speed_criterion(
            speed_prediction / config.model.speed_output_max,
            speed_target / config.model.speed_output_max,
        )
        emd_loss = angle_loss.new_zeros(())
        total_loss = angle_loss + config.loss.speed_loss_weight * speed_loss
        return total_loss, angle_loss, speed_loss, emd_loss

    angle_logits = outputs["angle_logits"]
    speed_logits = outputs["speed_logits"]
    angle_class = _batch_tensor(batch["angle_class_id"], angle_logits.device).long()
    speed_class = _batch_tensor(batch["speed_class_id"], speed_logits.device).long()
    angle_class = angle_class.reshape(-1)
    speed_class = speed_class.reshape(-1)
    angle_loss = angle_criterion(angle_logits, angle_class)
    speed_loss = speed_criterion(speed_logits, speed_class)
    angle_emd = ordinal_emd_loss(angle_logits, angle_class)
    speed_emd = ordinal_emd_loss(speed_logits, speed_class)
    total_loss, emd_loss = combine_policy_losses(
        angle_loss=angle_loss,
        speed_loss=speed_loss,
        angle_emd_loss=angle_emd,
        speed_emd_loss=speed_emd,
        speed_loss_weight=config.loss.speed_loss_weight,
        emd_loss_weight=config.loss.emd_loss_weight,
    )
    return total_loss, angle_loss, speed_loss, emd_loss


def _batch_tensor(value: object, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    return torch.as_tensor(value, device=device)


def run_epoch(
    *,
    model: TaskTokenViTPolicy | AutoregressiveControlTokenViTPolicy,
    loader: DataLoader,
    split_name: str,
    device: torch.device,
    angle_criterion: nn.Module,
    speed_criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    grad_clip: float,
    config: TrainConfig,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    control_encoding = getattr(model, "control_encoding", LEGACY_CONTROL_ENCODING)
    accumulator = _metric_accumulator(split_name, config)
    progress = tqdm(loader, desc=split_name, leave=False)
    for batch in progress:
        images = batch["image_tensor"].to(device=device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                if isinstance(model, AutoregressiveControlTokenViTPolicy):
                    history_key = (
                        "history_token_ids"
                        if control_encoding == COMPACT_CONTROL_ENCODING
                        else "history_class_ids"
                    )
                    history_class_ids = batch[history_key].to(
                        device=device,
                        non_blocking=True,
                    )
                    outputs = model(images, history_class_ids)
                else:
                    outputs = model(images)
                total_loss, angle_loss, speed_loss, emd_loss = (
                    compute_policy_losses(
                        outputs=outputs,
                        batch=batch,
                        config=config,
                        angle_criterion=angle_criterion,
                        speed_criterion=speed_criterion,
                    )
                )
            if training:
                scaler.scale(total_loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        accumulator.update(
            outputs={key: value.detach() for key, value in outputs.items()},
            batch=batch,
            total_loss=total_loss.detach(),
            angle_loss=angle_loss.detach(),
            speed_loss=speed_loss.detach(),
            emd_loss=emd_loss.detach(),
        )
        current = accumulator.compute()
        progress.set_postfix(
            loss=f"{current[f'{split_name}_loss']:.3f}",
            angle_mae=f"{current[f'{split_name}_angle_mae']:.2f}",
        )
    return accumulator.compute()


def class_weights(
    samples: Sequence[PolicySample],
    *,
    field: str,
    mode: str,
    config: TrainConfig,
    device: torch.device,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode != "sqrt_inverse_frequency":
        raise ValueError(f"unsupported class weighting mode: {mode}")
    sample_weights = (
        generation_sampling_weights(
            samples,
            current_generation=config.data.current_generation,
            generation_decay=config.data.generation_decay,
            source_sampling_masses=config.data.source_sampling_masses,
        )
        if config.data.ema_sampling
        else None
    )
    compact = config.model.control_encoding == COMPACT_CONTROL_ENCODING
    if compact:
        class_ids = [
            (
                angle_target_class_id(sample.angle)
                if field == "angle_class_id"
                else speed_target_class_id(sample.speed)
            )
            for sample in samples
        ]
        num_classes = (
            ANGLE_OUTPUT_CLASSES if field == "angle_class_id" else SPEED_OUTPUT_CLASSES
        )
    else:
        class_ids = None
        num_classes = NUM_COMMAND_CLASSES
    return compute_sqrt_inverse_frequency_weights(
        samples,
        field=field,
        num_classes=num_classes,
        class_ids=class_ids,
        min_weight=config.loss.min_class_weight,
        max_weight=config.loss.max_class_weight,
        mirror_probability=(
            config.augmentation.horizontal_flip_probability
            if field == "angle_class_id"
            else 0.0
        ),
        sample_weights=sample_weights,
    ).to(device)


def validation_selection_score(
    metrics: dict[str, float],
    *,
    sessions: Sequence[PolicySession],
    config: TrainConfig,
) -> float:
    if not config.data.ema_sampling:
        return selection_score(
            metrics,
            speed_mae_weight=config.training.validation_speed_mae_weight,
        )
    if config.data.source_sampling_masses:
        samples = tuple(sample for session in sessions for sample in session.samples)
        pair_masses = source_generation_sampling_masses(
            samples,
            current_generation=config.data.current_generation,
            generation_decay=config.data.generation_decay,
            source_sampling_masses=config.data.source_sampling_masses,
        )
        return sum(
            mass
            * (
                metrics[f"val_source_{source_id}_generation_{generation}_angle_mae"]
                + config.training.validation_speed_mae_weight
                * metrics[f"val_source_{source_id}_generation_{generation}_speed_mae"]
            )
            for (source_id, generation), mass in pair_masses.items()
        )
    generations = sorted({session.generation for session in sessions})
    if not generations:
        raise ValueError("validation split has no generations")
    future = [
        generation
        for generation in generations
        if generation > config.data.current_generation
    ]
    if future:
        raise ValueError(f"validation contains future generation(s): {future}")
    weighted_score = 0.0
    total_mass = 0.0
    for generation in generations:
        mass = config.data.generation_decay ** (
            config.data.current_generation - generation
        )
        prefix = f"val_generation_{generation}"
        weighted_score += mass * (
            metrics[f"{prefix}_angle_mae"]
            + config.training.validation_speed_mae_weight
            * metrics[f"{prefix}_speed_mae"]
        )
        total_mass += mass
    return weighted_score / total_mass


def source_weighted_metric(
    metrics: Mapping[str, float],
    *,
    sessions: Sequence[PolicySession],
    config: TrainConfig,
    source_id: str,
    metric_name: str,
    split_name: str = "val",
) -> float:
    """Combine one source's per-generation metrics using configured decay."""
    if source_id not in config.data.source_sampling_masses:
        raise ValueError(f"source is not configured for anchored sampling: {source_id}")
    samples = tuple(sample for session in sessions for sample in session.samples)
    pair_masses = source_generation_sampling_masses(
        samples,
        current_generation=config.data.current_generation,
        generation_decay=config.data.generation_decay,
        source_sampling_masses=config.data.source_sampling_masses,
    )
    source_mass = config.data.source_sampling_masses[source_id]
    return sum(
        pair_mass
        / source_mass
        * metrics[
            f"{split_name}_source_{observed_source}_generation_"
            f"{generation}_{metric_name}"
        ]
        for (observed_source, generation), pair_mass in pair_masses.items()
        if observed_source == source_id
    )


def checkpoint_payload(
    *,
    epoch: int,
    model: TaskTokenViTPolicy | AutoregressiveControlTokenViTPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    split_manifest: dict[str, object],
    dataset_stats: dict[str, object],
    preprocessing: dict[str, object],
    label_contract: dict[str, object],
    source_state: dict[str, object],
    best_score: float,
    best_epoch: int,
    epochs_without_improvement: int,
    metrics: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": config.serializable(),
        "model_name": config.model.name,
        "pretrained": config.model.pretrained,
        "preprocessing": preprocessing,
        "label_contract": label_contract,
        "split_manifest": split_manifest,
        "dataset_stats": dataset_stats,
        "source": source_state,
        "training_objective": build_training_objective_contract(config),
        "best_score": best_score,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "metrics": metrics,
    }


def initialize_model_weights(
    *,
    model: TaskTokenViTPolicy | AutoregressiveControlTokenViTPolicy,
    checkpoint: str,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, object] | None:
    """Load model parameters only; optimizer, scheduler, and run state stay fresh."""
    if not checkpoint:
        return None
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"initialization checkpoint does not exist: {checkpoint_path}"
        )
    payload = load_checkpoint(checkpoint_path, device)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("initialization checkpoint schema is incompatible")
    if payload.get("model_name") != config.model.name:
        raise ValueError("initialization checkpoint model_name differs from config")
    checkpoint_config = payload.get("config")
    checkpoint_model = (
        checkpoint_config.get("model") if isinstance(checkpoint_config, dict) else None
    )
    expected = {
        "architecture": config.model.architecture,
        "history_frames": config.model.history_frames,
        "control_token_type_embedding": config.model.control_token_type_embedding,
        "control_encoding": config.model.control_encoding,
        "prediction_mode": config.model.prediction_mode,
    }
    actual = {
        "architecture": (
            checkpoint_model.get("architecture", "task_tokens")
            if isinstance(checkpoint_model, dict)
            else None
        ),
        "history_frames": (
            checkpoint_model.get("history_frames", 0)
            if isinstance(checkpoint_model, dict)
            else None
        ),
        "control_token_type_embedding": (
            checkpoint_model.get("control_token_type_embedding", False)
            if isinstance(checkpoint_model, dict)
            else None
        ),
        "control_encoding": (
            checkpoint_model.get("control_encoding", LEGACY_CONTROL_ENCODING)
            if isinstance(checkpoint_model, dict)
            else None
        ),
        "prediction_mode": (
            checkpoint_model.get("prediction_mode", CATEGORICAL_PREDICTION_MODE)
            if isinstance(checkpoint_model, dict)
            else None
        ),
    }
    source_speed_output_max = (
        float(checkpoint_model.get("speed_output_max", 30.0))
        if isinstance(checkpoint_model, dict)
        else None
    )
    shared_categorical_to_regression = (
        actual["prediction_mode"] == CATEGORICAL_PREDICTION_MODE
        and expected["prediction_mode"] == CONTINUOUS_REGRESSION_PREDICTION_MODE
        and all(
            actual[key] == expected[key]
            for key in (
                "architecture",
                "history_frames",
                "control_token_type_embedding",
                "control_encoding",
            )
        )
    )
    if actual != expected and not shared_categorical_to_regression:
        raise ValueError(
            "initialization checkpoint model architecture differs from config"
        )
    if config.data.sources:
        checkpoint_data = (
            checkpoint_config.get("data")
            if isinstance(checkpoint_config, dict)
            else None
        )
        source_generation = (
            checkpoint_data.get("current_generation")
            if isinstance(checkpoint_data, dict)
            else None
        )
        source_steering_contract = (
            checkpoint_data.get("required_steering_contract")
            if isinstance(checkpoint_data, dict)
            else None
        )
        if source_steering_contract != config.data.required_steering_contract:
            raise ValueError(
                "initialization checkpoint steering contract differs from config"
            )
        expected_generation = config.data.current_generation - 1
        if source_generation != expected_generation:
            raise ValueError(
                "initialization checkpoint must be from the immediately "
                f"previous stateless generation {expected_generation}"
            )
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("initialization checkpoint has no model_state")
    if shared_categorical_to_regression:
        incompatible = model.load_state_dict(model_state, strict=False)
        expected_missing = {
            key
            for key in model.state_dict()
            if key.startswith("angle_regression_head.")
            or key.startswith("speed_regression_head.")
        }
        expected_unexpected = {"angle_output_bias", "speed_output_bias"}
        if (
            set(incompatible.missing_keys) != expected_missing
            or set(incompatible.unexpected_keys) != expected_unexpected
        ):
            raise ValueError(
                "categorical checkpoint differs outside the regression output heads"
            )
        initialization_mode = "shared_model_weights_only"
    else:
        model.load_state_dict(model_state, strict=True)
        initialization_mode = "model_weights_only"
    return {
        "mode": initialization_mode,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_epoch": int(payload.get("epoch", 0)),
        "source_prediction_mode": actual["prediction_mode"],
        "target_prediction_mode": expected["prediction_mode"],
        "source_speed_output_max": source_speed_output_max,
        "target_speed_output_max": config.model.speed_output_max,
    }


def prepare_run_directory(
    *,
    config: TrainConfig,
    resume: str,
    device: torch.device,
    expected_split: dict[str, object],
    expected_preprocessing: dict[str, object],
    expected_label_contract: dict[str, object],
) -> tuple[Path, dict[str, object] | None]:
    if resume:
        checkpoint_path = Path(resume).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"resume checkpoint does not exist: {checkpoint_path}"
            )
        payload = load_checkpoint(checkpoint_path, device)
        validate_resume_payload(
            payload,
            config=config,
            expected_split=expected_split,
            expected_preprocessing=expected_preprocessing,
            expected_label_contract=expected_label_contract,
        )
        return checkpoint_path.parent, payload

    run_name = config.output.run_name or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = unique_directory(config.output.root / run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, None


def validate_resume_payload(
    payload: Mapping[str, object],
    *,
    config: TrainConfig,
    expected_split: dict[str, object],
    expected_preprocessing: dict[str, object] | None = None,
    expected_label_contract: dict[str, object] | None = None,
) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("resume checkpoint schema is incompatible")
    if payload.get("model_name") != config.model.name:
        raise ValueError("resume checkpoint model_name differs from the config")
    label_contract = payload.get("label_contract")
    if not isinstance(label_contract, dict):
        raise ValueError("resume checkpoint label contract is incompatible")
    if config.model.control_encoding == COMPACT_CONTROL_ENCODING:
        expected_shapes = (
            {"angle_driver": [1, 1], "speed": [1, 1]}
            if config.model.prediction_mode
            == CONTINUOUS_REGRESSION_PREDICTION_MODE
            else {
                "angle_logits": [1, ANGLE_OUTPUT_CLASSES],
                "speed_logits": [1, SPEED_OUTPUT_CLASSES],
            }
        )
        if label_contract.get("output_shapes") != expected_shapes:
            raise ValueError("resume checkpoint compact output shapes are incompatible")
    elif label_contract.get("num_classes") != NUM_COMMAND_CLASSES:
        raise ValueError("resume checkpoint label contract is incompatible")
    if expected_label_contract is not None and label_contract.get(
        "train_angle_target"
    ) != expected_label_contract.get("train_angle_target"):
        raise ValueError("resume checkpoint train angle target differs from the config")
    if expected_label_contract is not None and label_contract.get(
        "history"
    ) != expected_label_contract.get("history"):
        raise ValueError("resume checkpoint history contract differs from the config")
    if payload.get("split_manifest") != expected_split:
        raise ValueError("resume checkpoint dataset split differs from the config")
    checkpoint_preprocessing = payload.get("preprocessing")
    if expected_preprocessing is not None and (
        not isinstance(checkpoint_preprocessing, Mapping)
        or _resume_geometry_contract(checkpoint_preprocessing)
        != _resume_geometry_contract(expected_preprocessing)
    ):
        raise ValueError(
            "resume checkpoint preprocessing differs from the current warp config"
        )
    checkpoint_config = payload.get("config")
    checkpoint_model = (
        checkpoint_config.get("model") if isinstance(checkpoint_config, dict) else None
    )
    expected_model_contract = {
        "architecture": config.model.architecture,
        "history_frames": config.model.history_frames,
        "control_token_type_embedding": (config.model.control_token_type_embedding),
        "control_encoding": config.model.control_encoding,
        "history_initial_angle": config.model.history_initial_angle,
        "history_initial_speed": config.model.history_initial_speed,
        "prediction_mode": config.model.prediction_mode,
        "speed_output_max": config.model.speed_output_max,
    }
    checkpoint_model_contract = {
        "architecture": (
            checkpoint_model.get("architecture", "task_tokens")
            if isinstance(checkpoint_model, dict)
            else None
        ),
        "history_frames": (
            checkpoint_model.get("history_frames", 0)
            if isinstance(checkpoint_model, dict)
            else None
        ),
        "control_token_type_embedding": (
            checkpoint_model.get("control_token_type_embedding", False)
            if isinstance(checkpoint_model, dict)
            else None
        ),
        "control_encoding": (
            checkpoint_model.get("control_encoding", LEGACY_CONTROL_ENCODING)
            if isinstance(checkpoint_model, dict)
            else None
        ),
        "history_initial_angle": (
            checkpoint_model.get("history_initial_angle", 0)
            if isinstance(checkpoint_model, dict)
            else None
        ),
        "history_initial_speed": (
            checkpoint_model.get("history_initial_speed", 25)
            if isinstance(checkpoint_model, dict)
            else None
        ),
        "prediction_mode": (
            checkpoint_model.get("prediction_mode", CATEGORICAL_PREDICTION_MODE)
            if isinstance(checkpoint_model, dict)
            else None
        ),
        "speed_output_max": (
            checkpoint_model.get("speed_output_max", 30.0)
            if isinstance(checkpoint_model, dict)
            else None
        ),
    }
    if checkpoint_model_contract != expected_model_contract:
        raise ValueError("resume checkpoint model history settings differ from config")
    checkpoint_augmentation = (
        checkpoint_config.get("augmentation")
        if isinstance(checkpoint_config, dict)
        else None
    )
    if (
        not isinstance(checkpoint_augmentation, dict)
        or checkpoint_augmentation.get("horizontal_flip_probability")
        != config.augmentation.horizontal_flip_probability
    ):
        raise ValueError(
            "resume checkpoint horizontal flip probability differs from the config"
        )
    checkpoint_training = (
        checkpoint_config.get("training")
        if isinstance(checkpoint_config, dict)
        else None
    )
    checkpoint_patience = (
        checkpoint_training.get("early_stopping_patience")
        if isinstance(checkpoint_training, dict)
        else None
    )
    if checkpoint_patience != config.training.early_stopping_patience:
        raise ValueError(
            "resume checkpoint early stopping patience differs from the config"
        )
    required = {
        "epoch",
        "model_state",
        "optimizer_state",
        "scheduler_state",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"resume checkpoint fields are missing: {sorted(missing)}")


def load_configured_road_warp(config: TrainConfig) -> RoadWarpConfig | None:
    path = config.preprocessing.road_warp_config
    return load_road_warp_config(path) if path is not None else None


def _resume_geometry_contract(
    preprocessing: Mapping[str, object],
) -> dict[str, object]:
    return {
        "geometry": preprocessing.get("geometry"),
        "road_warp": preprocessing.get("road_warp"),
    }


def build_preprocessing_contract(
    model_contract: Mapping[str, object],
    *,
    config: TrainConfig,
    road_warp: RoadWarpConfig | None,
) -> dict[str, object]:
    contract = dict(model_contract)
    training_augmentation: dict[str, object] = {
        "horizontal_flip_probability": (
            config.augmentation.horizontal_flip_probability
        ),
        "horizontal_flip_before_resize": True,
    }
    if road_warp is not None:
        contract["geometry"] = ROAD_WARP_GEOMETRY
        contract["road_warp"] = road_warp.contract(
            source_path=config.preprocessing.road_warp_config
        )
        training_augmentation["horizontal_flip_after_road_warp"] = True
    contract["training_augmentation"] = training_augmentation
    return contract


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint must contain a mapping: {path}")
    return dict(payload)


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def collect_source_state(project_root: Path) -> dict[str, object]:
    source_root = next(
        (
            candidate
            for candidate in (project_root, *project_root.parents)
            if (candidate / "uv.lock").is_file()
            and (candidate / "src" / "xycar_ai").is_dir()
        ),
        project_root,
    )
    uv_lock = source_root / "uv.lock"
    state: dict[str, object] = {
        "uv_lock_sha256": sha256_file(uv_lock) if uv_lock.is_file() else "unknown",
    }
    git_root = source_root.parent
    source_pathspec = source_root.name
    try:
        commit = subprocess.run(
            ["git", "-C", str(git_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(git_root),
                    "status",
                    "--porcelain",
                    "--",
                    source_pathspec,
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        state.update({"mgw_commit": commit, "dirty": dirty})
    except (OSError, subprocess.CalledProcessError):
        state.update({"mgw_commit": "unknown", "dirty": True})
    return state


def set_seed(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_optimizer(
    model: TaskTokenViTPolicy | AutoregressiveControlTokenViTPolicy,
    config: TrainConfig,
) -> torch.optim.AdamW:
    """Create fresh AdamW state, optionally using a lower backbone LR."""
    base_lr = config.optimizer.learning_rate
    multiplier = config.optimizer.backbone_learning_rate_multiplier
    if multiplier == 1.0:
        parameter_groups: object = model.parameters()
    else:
        backbone_parameters = list(model.backbone.parameters())
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        policy_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in backbone_ids
        ]
        parameter_groups = [
            {
                "params": backbone_parameters,
                "lr": base_lr * multiplier,
                "name": "backbone",
            },
            {
                "params": policy_parameters,
                "lr": base_lr,
                "name": "policy",
            },
        ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=base_lr,
        weight_decay=config.optimizer.weight_decay,
    )


def early_stopping_triggered(
    epochs_without_improvement: int,
    patience: int | None,
) -> bool:
    return patience is not None and epochs_without_improvement >= patience


def cosine_lr_factor(epochs: int, warmup_epochs: int):
    def factor(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / float(warmup_epochs)
        denominator = max(epochs - warmup_epochs, 1)
        progress = min(max(epoch_index - warmup_epochs, 0) / denominator, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def unique_directory(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.name}_{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate output directory from {path}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def write_metrics_csv(
    path: Path, rows: Sequence[Mapping[str, float | int | str]]
) -> None:
    if not rows:
        return
    fieldnames = sorted({field for row in rows for field in row})
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_metrics_rows(path: Path) -> list[dict[str, float | int | str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return [dict(row) for row in csv.DictReader(csv_file)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
