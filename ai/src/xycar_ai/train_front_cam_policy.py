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

from xycar_ai.config import TrainConfig, load_train_config
from xycar_ai.front_cam_policy_data import (
    COMMAND_MAX,
    COMMAND_MIN,
    NUM_COMMAND_CLASSES,
    FrontCamPolicyDataset,
    PolicyDataSplits,
    PolicySample,
    PolicySession,
    attach_executed_command_history,
    attach_training_teacher_forced_history,
    build_policy_data_splits,
    compute_sqrt_inverse_frequency_weights,
    generation_epoch_sample_count,
    generation_sampling_summary,
    generation_sampling_weights,
    make_policy_transform,
    policy_dataset_stats,
    smooth_training_angle_targets,
    validate_session_initial_classes,
)
from xycar_ai.front_cam_policy_metrics import (
    ClassificationMetricAccumulator,
    combine_policy_losses,
    ordinal_emd_loss,
    selection_score,
)
from xycar_ai.front_cam_policy_model import (
    AR_CONTROL_TOKEN_ARCHITECTURE,
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
    if external_history:
        splits = attach_executed_command_history(
            splits,
            config.model.history_frames,
            require_recorded_history=(
                config.model.history_sample_clock == "camera_frame"
            ),
        )
    elif config.model.architecture == AR_CONTROL_TOKEN_ARCHITECTURE:
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
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
            speed_loss_weight=config.loss.speed_loss_weight,
            emd_loss_weight=config.loss.emd_loss_weight,
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
        if config.model.history_update == "externally_executed_commands":
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
            if config.model.history_sample_clock == "camera_frame":
                history_contract.update(
                    {
                        "sample_clock": "camera_frame",
                        "train_source": "samples_csv_recorded_executed_history",
                        "evaluation_source": [
                            "samples_csv_recorded_executed_history",
                            "predicted_argmax_rollout",
                        ],
                        "edge_padding": "recorded_per_sample",
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
    if config.model.history_sample_clock == "camera_frame":
        if config.data.stage == "base":
            if generation != 0:
                raise ValueError("history base must use generation 0")
            if initialize_from:
                raise ValueError(
                    "history base must start from pretrained ImageNet weights"
                )
            return
        if not initialize_from and not resume:
            raise ValueError(
                "history guided training requires --initialize-from the "
                "validated base/previous guided best.pt"
            )
        return
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
                ),
                num_samples=generation_epoch_sample_count(
                    samples,
                    current_generation=config.data.current_generation,
                    generation_decay=config.data.generation_decay,
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
    if isinstance(model, AutoregressiveControlTokenViTPolicy):
        if config.model.history_update == "predicted_argmax":
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
        if config.model.history_sample_clock == "camera_frame":
            execution_metrics = run_epoch(
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
                speed_loss_weight=config.loss.speed_loss_weight,
                emd_loss_weight=config.loss.emd_loss_weight,
            )
            rollout_metrics = run_rollout_evaluation(
                model=model,
                sessions=sessions,
                split_name=f"{split_name}_rollout",
                config=config,
                model_data_config=model_data_config,
                road_warp=road_warp,
                device=device,
                angle_criterion=angle_criterion,
                speed_criterion=speed_criterion,
                amp_enabled=amp_enabled,
            )
            return {**execution_metrics, **rollout_metrics}
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
        speed_loss_weight=config.loss.speed_loss_weight,
        emd_loss_weight=config.loss.emd_loss_weight,
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
    accumulator = ClassificationMetricAccumulator(split_name)
    progress = tqdm(
        total=sum(len(session.samples) for session in sessions),
        desc=split_name,
        leave=False,
    )
    initial_pair = [
        config.model.history_initial_angle + 100,
        config.model.history_initial_speed + 100,
    ]
    transform = make_policy_transform(
        train=False,
        image_size=config.model.image_size,
        model_data_config=model_data_config,
        augmentation=config.augmentation,
    )
    with torch.inference_mode():
        for session in sessions:
            first_history = session.samples[0].history_class_ids
            if (
                config.model.history_sample_clock == "camera_frame"
                and first_history is not None
            ):
                initial_history = list(first_history)
            else:
                initial_history = [initial_pair] * config.model.history_frames
            history = torch.tensor(
                [initial_history],
                dtype=torch.long,
                device=device,
            )
            dataset = FrontCamPolicyDataset(
                session.samples,
                transform=transform,
                horizontal_flip_probability=0.0,
                road_warp=road_warp,
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
                angle_class = torch.tensor(
                    [item["angle_class_id"]], dtype=torch.long, device=device
                )
                speed_class = torch.tensor(
                    [item["speed_class_id"]], dtype=torch.long, device=device
                )
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    outputs = model(images, history)
                    angle_loss = angle_criterion(outputs["angle_logits"], angle_class)
                    speed_loss = speed_criterion(outputs["speed_logits"], speed_class)
                    angle_emd = ordinal_emd_loss(outputs["angle_logits"], angle_class)
                    speed_emd = ordinal_emd_loss(outputs["speed_logits"], speed_class)
                    total_loss, emd_loss = combine_policy_losses(
                        angle_loss=angle_loss,
                        speed_loss=speed_loss,
                        angle_emd_loss=angle_emd,
                        speed_emd_loss=speed_emd,
                        speed_loss_weight=config.loss.speed_loss_weight,
                        emd_loss_weight=config.loss.emd_loss_weight,
                    )
                accumulator.update(
                    outputs={key: value.detach() for key, value in outputs.items()},
                    batch=item,
                    total_loss=total_loss.detach(),
                    angle_loss=angle_loss.detach(),
                    speed_loss=speed_loss.detach(),
                    emd_loss=emd_loss.detach(),
                )
                predicted_pair = torch.stack(
                    (
                        outputs["angle_logits"].argmax(dim=1),
                        outputs["speed_logits"].argmax(dim=1),
                    ),
                    dim=1,
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
    speed_loss_weight: float,
    emd_loss_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    accumulator = ClassificationMetricAccumulator(split_name)
    progress = tqdm(loader, desc=split_name, leave=False)
    for batch in progress:
        images = batch["image_tensor"].to(device=device, non_blocking=True)
        angle_class = batch["angle_class_id"].to(device=device, non_blocking=True)
        speed_class = batch["speed_class_id"].to(device=device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                if isinstance(model, AutoregressiveControlTokenViTPolicy):
                    history_class_ids = batch["history_class_ids"].to(
                        device=device,
                        non_blocking=True,
                    )
                    outputs = model(images, history_class_ids)
                else:
                    outputs = model(images)
                angle_loss = angle_criterion(outputs["angle_logits"], angle_class)
                speed_loss = speed_criterion(outputs["speed_logits"], speed_class)
                angle_emd = ordinal_emd_loss(outputs["angle_logits"], angle_class)
                speed_emd = ordinal_emd_loss(outputs["speed_logits"], speed_class)
                total_loss, emd_loss = combine_policy_losses(
                    angle_loss=angle_loss,
                    speed_loss=speed_loss,
                    angle_emd_loss=angle_emd,
                    speed_emd_loss=speed_emd,
                    speed_loss_weight=speed_loss_weight,
                    emd_loss_weight=emd_loss_weight,
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
        )
        if config.data.ema_sampling
        else None
    )
    return compute_sqrt_inverse_frequency_weights(
        samples,
        field=field,
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
    execution_score = _selection_score_for_prefix(
        metrics,
        sessions=sessions,
        config=config,
        split_name="val",
    )
    if config.model.history_sample_clock != "camera_frame":
        return execution_score
    rollout_score = _selection_score_for_prefix(
        metrics,
        sessions=sessions,
        config=config,
        split_name="val_rollout",
    )
    return (execution_score + rollout_score) / 2.0


def _selection_score_for_prefix(
    metrics: dict[str, float],
    *,
    sessions: Sequence[PolicySession],
    config: TrainConfig,
    split_name: str,
) -> float:
    if not config.data.ema_sampling:
        return selection_score(metrics, split_name=split_name)
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
        prefix = f"{split_name}_generation_{generation}"
        weighted_score += mass * (
            metrics[f"{prefix}_angle_mae"] + 0.25 * metrics[f"{prefix}_speed_mae"]
        )
        total_mass += mass
    return weighted_score / total_mass


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
        "history_initial_angle": config.model.history_initial_angle,
        "history_initial_speed": config.model.history_initial_speed,
        "history_update": config.model.history_update,
        "history_sample_clock": config.model.history_sample_clock,
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
        "history_update": (
            checkpoint_model.get("history_update", "predicted_argmax")
            if isinstance(checkpoint_model, dict)
            else None
        ),
        "history_sample_clock": (
            checkpoint_model.get("history_sample_clock")
            if isinstance(checkpoint_model, dict)
            else None
        ),
    }
    if actual != expected:
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
        if config.model.history_sample_clock == "camera_frame":
            source_stage = (
                checkpoint_data.get("stage")
                if isinstance(checkpoint_data, dict)
                else None
            )
            expected_generation = max(config.data.current_generation - 1, 0)
            expected_stage = (
                "base" if config.data.current_generation == 0 else "guided"
            )
            if (
                source_generation != expected_generation
                or source_stage != expected_stage
            ):
                raise ValueError(
                    "history guided initialization checkpoint must be from "
                    f"{expected_stage} generation {expected_generation}"
                )
        else:
            expected_generation = config.data.current_generation - 1
            if source_generation != expected_generation:
                raise ValueError(
                    "initialization checkpoint must be from the immediately "
                    f"previous stateless generation {expected_generation}"
                )
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("initialization checkpoint has no model_state")
    model.load_state_dict(model_state, strict=True)
    return {
        "mode": "model_weights_only",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_epoch": int(payload.get("epoch", 0)),
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
    if not isinstance(label_contract, dict) or label_contract.get("num_classes") != 201:
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
        "history_initial_angle": config.model.history_initial_angle,
        "history_initial_speed": config.model.history_initial_speed,
        "history_update": config.model.history_update,
        "history_sample_clock": config.model.history_sample_clock,
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
        "history_update": (
            checkpoint_model.get("history_update", "predicted_argmax")
            if isinstance(checkpoint_model, dict)
            else None
        ),
        "history_sample_clock": (
            checkpoint_model.get("history_sample_clock")
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
    uv_lock = project_root / "uv.lock"
    state: dict[str, object] = {
        "uv_lock_sha256": sha256_file(uv_lock) if uv_lock.is_file() else "unknown",
    }
    git_root = project_root.parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(git_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(git_root), "status", "--porcelain", "--", "ai/"],
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
