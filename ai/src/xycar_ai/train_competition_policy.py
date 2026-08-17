"""Train temporal traffic-signal and shortcut policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

from xycar_ai.steering_contract import (
    STEERING_CONTRACT_NAME,
    is_exact_steering_contract,
    steering_contract_mapping,
)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from xycar_ai.competition_data import (
    LAMP_NAMES,
    MissionSession,
    materialize_signal_labels,
)
from xycar_ai.competition_datasets import (
    ShortcutSequenceDataset,
    SignalSequenceDataset,
    select_split_sessions,
)
from xycar_ai.competition_models import (
    ShortcutModelConfig,
    ShortcutTemporalPolicy,
    SignalModelConfig,
    SignalTemporalPolicy,
)


class CompetitionTrainingError(ValueError):
    """Raised when a competition training run is unsafe or inconsistent."""


def signal_main(argv: Iterable[str] | None = None) -> None:
    _main("signal", argv)


def shortcut_main(argv: Iterable[str] | None = None) -> None:
    _main("shortcut", argv)


def _main(kind: str, argv: Iterable[str] | None) -> None:
    parser = argparse.ArgumentParser(description=f"Train {kind} temporal policy")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)
    config_path = Path(arguments.config).expanduser().resolve()
    config = _load_config(config_path, expected_kind=kind)
    dataset_root = _resolve_project_path(config_path, config["data"]["root"])
    split_manifest = _resolve_project_path(
        config_path,
        config["data"]["split_manifest"],
    )
    sessions = {
        split: select_split_sessions(
            dataset_root,
            split_manifest,
            split,
            capture_kind="shortcut" if kind == "shortcut" else None,
        )
        for split in ("train", "validation", "test")
    }
    if kind == "shortcut":
        for split, selected in sessions.items():
            legacy = [
                session.session_id
                for session in selected
                if not is_exact_steering_contract(
                    session.steering_contract,
                    include_topics=True,
                )
            ]
            if legacy:
                raise CompetitionTrainingError(
                    f"{split} shortcut sessions lack normalized steering: "
                    f"{legacy}"
                )
    _validate_coverage(kind, sessions, config["data"])
    datasets = {
        split: _build_dataset(kind, selected, config, augment=split == "train")
        for split, selected in sessions.items()
    }
    summary = {
        "kind": kind,
        "sessions": {split: len(value) for split, value in sessions.items()},
        "windows": {split: len(value) for split, value in datasets.items()},
    }
    if kind == "signal":
        summary["train_sampling_categories"] = datasets[
            "train"
        ].sampling_category_counts()
    print(json.dumps(summary, indent=2, sort_keys=True))
    if arguments.validate_only:
        return
    _train(kind, config_path, config, sessions, datasets)


def _train(
    kind: str,
    config_path: Path,
    config: Mapping[str, Any],
    sessions: Mapping[str, tuple[MissionSession, ...]],
    datasets: Mapping[str, Any],
) -> None:
    training = config["training"]
    seed = int(training["seed"])
    _seed_everything(seed, deterministic=bool(training["deterministic"]))
    requested_device = str(training["device"])
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise CompetitionTrainingError("CUDA training was requested but unavailable")
    device = torch.device(requested_device)
    model = _build_model(kind, config["model"]).to(device)
    loaders: dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        sampler = None
        shuffle = split == "train"
        if kind == "signal" and split == "train":
            assert isinstance(dataset, SignalSequenceDataset)
            sampler = WeightedRandomSampler(
                dataset.sampling_weights(),
                num_samples=len(dataset),
                replacement=True,
                generator=torch.Generator().manual_seed(seed),
            )
            shuffle = False
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
    )
    amp_enabled = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_root = _resolve_project_path(config_path, config["output"]["root"])
    run_dir = output_root / str(config["output"]["run_name"])
    if run_dir.exists():
        raise CompetitionTrainingError(f"refusing to overwrite run: {run_dir}")
    run_dir.mkdir(parents=True)
    provenance = {
        split: [
            {
                "session_id": session.session_id,
                "annotation_sha256": session.annotation_sha256,
            }
            for session in selected
        ]
        for split, selected in sessions.items()
    }
    if kind == "shortcut":
        provenance["steering_contract"] = steering_contract_mapping()
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "data_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    patience = int(training["early_stopping_patience"])
    best_score = math.inf
    stale_epochs = 0
    metrics_rows: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            kind,
            model,
            loaders["train"],
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip=float(training["grad_clip"]),
        )
        validation_metrics = _run_epoch(
            kind,
            model,
            loaders["validation"],
            device,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
            grad_clip=0.0,
        )
        scheduler.step()
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update(
            {f"val_{key}": value for key, value in validation_metrics.items()}
        )
        validation_score = _selection_score(kind, validation_metrics)
        row["val_selection_score"] = validation_score
        metrics_rows.append(row)
        _write_metrics(run_dir / "metrics.csv", metrics_rows)
        checkpoint = _checkpoint_payload(
            kind,
            config,
            model,
            optimizer,
            scheduler,
            epoch,
            validation_metrics,
            provenance,
        )
        torch.save(checkpoint, run_dir / "last.pt")
        if validation_score < best_score:
            best_score = validation_score
            stale_epochs = 0
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            stale_epochs += 1
        print(json.dumps(row, sort_keys=True))
        if stale_epochs >= patience:
            break
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    test_metrics = _run_epoch(
        kind,
        model,
        loaders["test"],
        device,
        optimizer=None,
        scaler=None,
        amp_enabled=amp_enabled,
        grad_clip=0.0,
    )
    (run_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"run_dir={run_dir}")


def _selection_score(kind: str, metrics: Mapping[str, float]) -> float:
    if kind == "signal":
        return (
            float(metrics["loss"])
            + 25.0 * float(metrics["stop_false_negative_rate"])
            + 10.0 * float(metrics["false_left_rate"])
        )
    return (
        float(metrics["loss"])
        + 25.0 * float(metrics["early_handoff_rate"])
        + 0.05 * float(metrics["first_angle_mae"])
    )


def _run_epoch(
    kind: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in raw_batch.items()
            }
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                if kind == "signal":
                    output = model(batch["images"])
                    metrics = _signal_losses(output, batch)
                else:
                    output = model(batch["images"], batch["previous_commands"])
                    metrics = _shortcut_losses(output, batch)
                loss = metrics["loss"]
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item())
            batches += 1
    if batches == 0:
        raise CompetitionTrainingError("data loader produced no batches")
    return {key: value / batches for key, value in totals.items()}


def _signal_losses(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    status_loss = F.binary_cross_entropy_with_logits(
        output["status_logits"],
        batch["status"],
    )
    bbox_mask = batch["bbox_valid"].unsqueeze(-1)
    bbox_denominator = bbox_mask.sum().clamp_min(1.0)
    bbox_loss = (
        F.smooth_l1_loss(output["bbox"], batch["bbox"], reduction="none")
        * bbox_mask
    ).sum() / (bbox_denominator * 4.0)
    approach_mask = batch["status"][..., 0]
    progress_loss = (
        F.smooth_l1_loss(
            output["progress"],
            batch["progress"],
            reduction="none",
        )
        * approach_mask
    ).sum() / approach_mask.sum().clamp_min(1.0)
    total = status_loss + bbox_loss * 2.0 + progress_loss * 0.5
    status_prediction = output["status_logits"].sigmoid() >= 0.5
    status_accuracy = (
        status_prediction == (batch["status"] >= 0.5)
    ).float().mean()
    actual_stop = (batch["status"][..., 3] >= 0.5) | (
        batch["status"][..., 4] >= 0.5
    )
    predicted_stop = status_prediction[..., 3] | status_prediction[..., 4]
    stop_false_negative = (
        actual_stop & ~predicted_stop
    ).float().sum() / actual_stop.float().sum().clamp_min(1.0)
    actual_left = batch["status"][..., 5] >= 0.5
    false_left = (
        ~actual_left & status_prediction[..., 5]
    ).float().sum() / (~actual_left).float().sum().clamp_min(1.0)
    return {
        "loss": total,
        "status_loss": status_loss,
        "bbox_loss": bbox_loss,
        "progress_loss": progress_loss,
        "status_accuracy": status_accuracy,
        "stop_false_negative_rate": stop_false_negative,
        "false_left_rate": false_left,
    }


def _shortcut_losses(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    horizon = output["angle_logits"].shape[2]
    weights = torch.linspace(
        1.0,
        0.25,
        horizon,
        device=output["angle_logits"].device,
    )
    angle_per_step = F.cross_entropy(
        output["angle_logits"].flatten(0, 1).transpose(1, 2),
        batch["angle_targets"].flatten(0, 1),
        reduction="none",
    )
    speed_per_step = F.cross_entropy(
        output["speed_logits"].flatten(0, 1).transpose(1, 2),
        batch["speed_targets"].flatten(0, 1),
        reduction="none",
    )
    angle_loss = (angle_per_step * weights).mean()
    speed_loss = (speed_per_step * weights).mean()
    phase_loss = F.cross_entropy(
        output["phase_logits"].flatten(0, 1),
        batch["phase_targets"].flatten(),
    )
    handoff_loss = F.binary_cross_entropy_with_logits(
        output["handoff_logits"],
        batch["handoff_targets"],
    )
    total = angle_loss + speed_loss * 0.5 + phase_loss * 0.5 + handoff_loss
    first_angle = output["angle_logits"][:, :, 0].argmax(dim=-1)
    angle_mae = (
        first_angle.float() - batch["angle_targets"][:, :, 0].float()
    ).abs().mean()
    early_handoff = (
        (output["handoff_logits"].sigmoid() >= 0.5)
        & (batch["handoff_targets"] < 0.5)
    ).float().mean()
    return {
        "loss": total,
        "angle_loss": angle_loss,
        "speed_loss": speed_loss,
        "phase_loss": phase_loss,
        "handoff_loss": handoff_loss,
        "first_angle_mae": angle_mae,
        "early_handoff_rate": early_handoff,
    }


def _build_dataset(
    kind: str,
    sessions: tuple[MissionSession, ...],
    config: Mapping[str, Any],
    *,
    augment: bool,
) -> SignalSequenceDataset | ShortcutSequenceDataset:
    data = config["data"]
    model = config["model"]
    if kind == "signal":
        return SignalSequenceDataset(
            sessions,
            sequence_length=int(data["sequence_length"]),
            frame_stride=int(data["frame_stride"]),
            window_step=int(data["window_step"]),
            input_height=int(model["input_height"]),
            input_width=int(model["input_width"]),
            augment=augment,
        )
    return ShortcutSequenceDataset(
        sessions,
        sequence_length=int(data["sequence_length"]),
        horizon_steps=int(model["horizon_steps"]),
        sample_rate_hz=float(data["sample_rate_hz"]),
        window_step=int(data["window_step"]),
        image_size=int(model["image_size"]),
        augment=augment,
    )


def _build_model(
    kind: str,
    model_config: Mapping[str, Any],
) -> SignalTemporalPolicy | ShortcutTemporalPolicy:
    if kind == "signal":
        return SignalTemporalPolicy(
            SignalModelConfig(
                backbone=str(model_config["backbone"]),
                pretrained=bool(model_config["pretrained"]),
                hidden_size=int(model_config["hidden_size"]),
                input_height=int(model_config["input_height"]),
                input_width=int(model_config["input_width"]),
            )
        )
    return ShortcutTemporalPolicy(
        ShortcutModelConfig(
            backbone=str(model_config["backbone"]),
            pretrained=bool(model_config["pretrained"]),
            hidden_size=int(model_config["hidden_size"]),
            image_size=int(model_config["image_size"]),
            horizon_steps=int(model_config["horizon_steps"]),
        )
    )


def _validate_coverage(
    kind: str,
    sessions: Mapping[str, tuple[MissionSession, ...]],
    data_config: Mapping[str, Any],
) -> None:
    all_sessions = tuple(
        session for selected in sessions.values() for session in selected
    )
    if kind == "shortcut":
        minimum = int(data_config["minimum_shortcut_sessions"])
        if len(all_sessions) < minimum:
            raise CompetitionTrainingError(
                f"shortcut requires at least {minimum} approved sessions; "
                f"found {len(all_sessions)}"
            )
        return
    minimum = int(data_config["minimum_signal_sessions_per_state"])
    minimum_negative = int(data_config["minimum_negative_sessions"])
    counts, negative = _signal_session_counts(all_sessions)
    missing = {name: count for name, count in counts.items() if count < minimum}
    if missing:
        raise CompetitionTrainingError(
            f"signal session coverage below {minimum}: {missing}"
        )
    if negative < minimum_negative:
        raise CompetitionTrainingError(
            f"signal requires {minimum_negative} negative sessions; found {negative}"
        )
    for split, selected in sessions.items():
        split_counts, split_negative = _signal_session_counts(selected)
        absent = [name for name, count in split_counts.items() if count == 0]
        if absent or split_negative == 0:
            raise CompetitionTrainingError(
                f"signal {split} split lacks states={absent}, "
                f"negative_sessions={split_negative}; regenerate a "
                "session-disjoint split with another seed"
            )


def _signal_session_counts(
    sessions: tuple[MissionSession, ...],
) -> tuple[dict[str, int], int]:
    counts = {name: 0 for name in LAMP_NAMES}
    negative = 0
    for session in sessions:
        labels = materialize_signal_labels(session)
        if not any(label["approach"] > 0.5 for label in labels):
            negative += 1
        for name in LAMP_NAMES:
            if any(label[name] > 0.5 for label in labels):
                counts[name] += 1
    return counts, negative


def _checkpoint_payload(
    kind: str,
    config: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Mapping[str, float],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    model_contract = asdict(model.config)
    return {
        "schema_version": 1,
        "policy_kind": kind,
        "epoch": epoch,
        "model_config": model_contract,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "validation_metrics": dict(metrics),
        "training_config": dict(config),
        "data_provenance": dict(provenance),
    }


def _load_config(path: Path, *, expected_kind: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise CompetitionTrainingError(f"config does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CompetitionTrainingError(f"invalid config YAML: {path}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise CompetitionTrainingError("training config schema_version must be 1")
    if payload.get("policy_kind") != expected_kind:
        raise CompetitionTrainingError(
            f"expected policy_kind {expected_kind}, got {payload.get('policy_kind')}"
        )
    for section in ("model", "data", "training", "output"):
        if not isinstance(payload.get(section), Mapping):
            raise CompetitionTrainingError(f"config.{section} must be a mapping")
    if expected_kind == "shortcut" and payload["data"].get(
        "required_steering_contract"
    ) != STEERING_CONTRACT_NAME:
        raise CompetitionTrainingError(
            "shortcut data must require normalized_percent_v1 steering"
        )
    return payload


def _resolve_project_path(config_path: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.parent.parent / path
    return path.resolve()


def _seed_everything(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _write_metrics(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
