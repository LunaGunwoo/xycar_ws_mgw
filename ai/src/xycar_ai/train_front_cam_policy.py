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
from torch.utils.data import DataLoader
from tqdm import tqdm

from xycar_ai.config import TrainConfig, load_train_config
from xycar_ai.front_cam_policy_data import (
    COMMAND_MAX,
    COMMAND_MIN,
    NUM_COMMAND_CLASSES,
    FrontCamPolicyDataset,
    PolicyDataSplits,
    PolicySample,
    build_policy_data_splits,
    compute_sqrt_inverse_frequency_weights,
    make_policy_transform,
    policy_dataset_stats,
)
from xycar_ai.front_cam_policy_metrics import (
    ClassificationMetricAccumulator,
    combine_policy_losses,
    ordinal_emd_loss,
    selection_score,
)
from xycar_ai.front_cam_policy_model import TaskTokenViTPolicy

DEFAULT_CONFIG = "config/front_cam_policy_train.yaml"
CHECKPOINT_SCHEMA_VERSION = 1


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_train_config(args.config)
    splits = build_policy_data_splits(config.data)
    split_manifest = splits.manifest()
    dataset_stats = policy_dataset_stats(splits)

    if args.validate_only:
        print(json.dumps(dataset_stats, indent=2, sort_keys=True))
        print(
            "validated "
            f"sessions={dataset_stats['all']['session_count']} "
            f"samples={dataset_stats['all']['sample_count']}"
        )
        return 0

    set_seed(config.training.seed, deterministic=config.training.deterministic)
    device = resolve_device(config.training.device)
    amp_enabled = bool(config.training.amp and device.type == "cuda")
    model = TaskTokenViTPolicy(
        model_name=config.model.name,
        pretrained=config.model.pretrained,
        image_size=config.model.image_size,
    ).to(device)
    preprocessing = {
        **model.preprocessing_contract(),
        "training_augmentation": {
            "horizontal_flip_probability": (
                config.augmentation.horizontal_flip_probability
            ),
            "horizontal_flip_before_resize": True,
        },
    }
    loaders = make_loaders(
        splits=splits,
        config=config,
        model_data_config=model.model_data_config,
        device=device,
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
    )
    write_yaml(run_dir / "resolved_config.yaml", config.serializable())
    write_json(run_dir / "split.json", split_manifest)
    write_json(run_dir / "dataset_stats.json", dataset_stats)

    start_epoch = 1
    best_score = math.inf
    best_epoch = 0
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
    if start_epoch > config.training.epochs:
        raise ValueError(
            f"resume epoch {start_epoch - 1} already reaches configured "
            f"epochs={config.training.epochs}"
        )

    metrics_path = run_dir / "metrics.csv"
    metrics_rows = read_metrics_rows(metrics_path) if resume_payload else []
    source_state = collect_source_state(config.project_root)
    label_contract = {
        "schema_version": 1,
        "output_keys": ["angle_logits", "speed_logits"],
        "num_classes": NUM_COMMAND_CLASSES,
        "command_min": COMMAND_MIN,
        "command_max": COMMAND_MAX,
        "class_id_mapping": "int(round(clamp(value, -100, 100))) + 100",
        "decode_mapping": "class_id - 100",
        "horizontal_flip_mapping": {
            "angle_raw": "-angle_raw",
            "angle": "-angle",
            "angle_class_id": "200 - angle_class_id",
            "speed": "unchanged",
            "speed_class_id": "unchanged",
        },
    }

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
        val_metrics = run_epoch(
            model=model,
            loader=loaders["val"],
            split_name="val",
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
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "lr": current_lr,
            **train_metrics,
            **val_metrics,
        }
        metrics_rows.append(row)
        write_metrics_csv(metrics_path, metrics_rows)

        current_score = selection_score(val_metrics)
        is_best = current_score < best_score
        if is_best:
            best_score = current_score
            best_epoch = epoch
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
            metrics=row,
        )
        atomic_torch_save(checkpoint, run_dir / "last.pt")
        if is_best:
            atomic_torch_save(checkpoint, run_dir / "best.pt")

    best_path = run_dir / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(f"best checkpoint is unavailable: {best_path}")
    best_payload = load_checkpoint(best_path, device)
    model.load_state_dict(best_payload["model_state"])
    test_metrics = run_epoch(
        model=model,
        loader=loaders["test"],
        split_name="test",
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
    write_json(run_dir / "test_metrics.json", test_metrics)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(run_dir / "last.pt"),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "device": str(device),
        "amp": amp_enabled,
        "source": source_state,
        "dataset": split_manifest,
        "model_name": config.model.name,
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Xycar front-camera angle/speed policy."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", default="")
    return parser


def make_loaders(
    *,
    splits: PolicyDataSplits,
    config: TrainConfig,
    model_data_config: Mapping[str, object],
    device: torch.device,
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
        )
        generator = torch.Generator()
        generator.manual_seed(config.training.seed + offset)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=training,
            num_workers=config.data.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.data.num_workers > 0,
            generator=generator,
            worker_init_fn=seed_worker,
        )
    return loaders


def run_epoch(
    *,
    model: TaskTokenViTPolicy,
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
    ).to(device)


def checkpoint_payload(
    *,
    epoch: int,
    model: TaskTokenViTPolicy,
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
        "metrics": metrics,
    }


def prepare_run_directory(
    *,
    config: TrainConfig,
    resume: str,
    device: torch.device,
    expected_split: dict[str, object],
) -> tuple[Path, dict[str, object] | None]:
    if resume:
        checkpoint_path = Path(resume).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"resume checkpoint does not exist: {checkpoint_path}"
            )
        payload = load_checkpoint(checkpoint_path, device)
        validate_resume_payload(payload, config=config, expected_split=expected_split)
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
) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("resume checkpoint schema is incompatible")
    if payload.get("model_name") != config.model.name:
        raise ValueError("resume checkpoint model_name differs from the config")
    label_contract = payload.get("label_contract")
    if not isinstance(label_contract, dict) or label_contract.get("num_classes") != 201:
        raise ValueError("resume checkpoint label contract is incompatible")
    if payload.get("split_manifest") != expected_split:
        raise ValueError("resume checkpoint dataset split differs from the config")
    checkpoint_config = payload.get("config")
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
    required = {
        "epoch",
        "model_state",
        "optimizer_state",
        "scheduler_state",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"resume checkpoint fields are missing: {sorted(missing)}")


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
