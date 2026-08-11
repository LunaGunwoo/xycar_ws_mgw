from __future__ import annotations

import csv
import copy
from pathlib import Path

import pytest
import torch
import yaml
from conftest import write_session, write_split_manifest

from xycar_ai.config import load_train_config
from xycar_ai.train_front_cam_policy import (
    build_preprocessing_contract,
    early_stopping_triggered,
    load_configured_road_warp,
    main,
    validate_resume_payload,
)


def test_ab_configs_only_change_flip_and_run_name():
    project_root = Path(__file__).parents[1]
    candidate = yaml.safe_load(
        (project_root / "config" / "front_cam_policy_train.yaml").read_text()
    )
    baseline = yaml.safe_load(
        (project_root / "config" / "front_cam_policy_train_no_flip.yaml").read_text()
    )

    assert candidate["augmentation"]["horizontal_flip_probability"] == 0.5
    assert baseline["augmentation"]["horizontal_flip_probability"] == 0.0
    assert candidate["output"]["run_name"] == "hflip_p05_seed20260810"
    assert baseline["output"]["run_name"] == "baseline_seed20260810"
    assert candidate["training"]["early_stopping_patience"] == 5
    assert baseline["training"]["early_stopping_patience"] == 5
    candidate["augmentation"]["horizontal_flip_probability"] = 0.0
    candidate["output"]["run_name"] = "baseline_seed20260810"
    assert candidate == baseline


def test_early_stopping_patience_counts_consecutive_non_improvements():
    assert not early_stopping_triggered(100, None)
    assert not early_stopping_triggered(4, 5)
    assert early_stopping_triggered(5, 5)


def test_small_config_uses_minimum_speed_and_flip():
    project_root = Path(__file__).parents[1]
    config = load_train_config(
        project_root / "config" / "front_cam_policy_train_small.yaml"
    )

    assert config.model.name == "vit_small_patch16_224.augreg_in21k_ft_in1k"
    assert config.data.max_forward_speed is None
    assert config.data.min_forward_speed == 20.0
    assert config.augmentation.horizontal_flip_probability == 0.5
    assert config.training.batch_size == 128
    assert config.output.run_name == "vit_small_hflip_p05_seed20260810"
    assert config.preprocessing.road_warp_config is None


def test_small_warp_config_embeds_preprocessing_contract():
    project_root = Path(__file__).parents[1]
    config = load_train_config(
        project_root / "config" / "front_cam_policy_train_small_warp.yaml"
    )
    road_warp = load_configured_road_warp(config)

    assert road_warp is not None
    contract = build_preprocessing_contract(
        {
            "geometry": "full_frame_bicubic_resize",
            "image_size": 224,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        config=config,
        road_warp=road_warp,
    )
    assert contract["geometry"] == (
        "perspective_road_warp_then_bicubic_resize"
    )
    assert contract["road_warp"]["parameters"]["bev_width"] == 224
    assert len(contract["road_warp"]["sha256"]) == 64
    assert (
        contract["training_augmentation"]["horizontal_flip_after_road_warp"]
        is True
    )

    split = {"schema_version": 1, "dataset_snapshot": "synthetic", "splits": {}}
    checkpoint = {
        "schema_version": 1,
        "model_name": config.model.name,
        "label_contract": {"num_classes": 201},
        "split_manifest": split,
        "preprocessing": contract,
        "config": config.serializable(),
        "epoch": 1,
        "model_state": {},
        "optimizer_state": {},
        "scheduler_state": {},
    }
    validate_resume_payload(
        checkpoint,
        config=config,
        expected_split=split,
        expected_preprocessing=contract,
    )
    changed_contract = copy.deepcopy(contract)
    changed_contract["road_warp"]["parameters"]["top_y"] = 0.7
    with pytest.raises(ValueError, match="preprocessing differs"):
        validate_resume_payload(
            checkpoint,
            config=config,
            expected_split=split,
            expected_preprocessing=changed_contract,
        )


def test_config_requires_exactly_one_forward_speed_filter(tmp_path: Path):
    split_path = tmp_path / "config" / "split.yaml"
    config_path = _write_config(tmp_path, split_path, epochs=1)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["data"]["min_forward_speed"] = 20.0
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        load_train_config(config_path)


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_config_rejects_horizontal_flip_probability_out_of_range(
    tmp_path: Path, probability: float
):
    config_path = _write_config(
        tmp_path,
        tmp_path / "config" / "split.yaml",
        epochs=1,
        flip_probability=probability,
    )

    with pytest.raises(ValueError, match="horizontal_flip_probability"):
        load_train_config(config_path)


def test_one_epoch_training_and_resume(tmp_path: Path):
    data_root = tmp_path / "datasets" / "teleop"
    names = [
        "20260803_150534_000_session",
        "20260810_130735_027_session",
        "20260810_130818_255_session",
        "20260803_150428_657_session",
        "20260805_144222_487_session",
    ]
    labels = [(-20.2, 25.0), (-5.4, 24.4), (0.0, 25.0), (15.6, 25.0), (40.1, 23.4)]
    for name, label in zip(names, labels):
        write_session(data_root, name, labels=[label])
    split_path = write_split_manifest(
        tmp_path / "config" / "split.yaml",
        train=names[:2],
        val=[names[2]],
        test=names[3:],
    )
    config_path = _write_config(tmp_path, split_path, epochs=1)

    assert main(["--config", str(config_path), "--validate-only"]) == 0
    assert main(["--config", str(config_path)]) == 0

    run_dir = tmp_path / "artifacts" / "runs" / "smoke"
    expected_outputs = {
        "resolved_config.yaml",
        "dataset_stats.json",
        "split.json",
        "metrics.csv",
        "best.pt",
        "last.pt",
        "test_metrics.json",
        "summary.json",
    }
    assert expected_outputs <= {path.name for path in run_dir.iterdir()}
    checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["label_contract"]["num_classes"] == 201
    assert checkpoint["label_contract"]["horizontal_flip_mapping"] == {
        "angle_raw": "-angle_raw",
        "angle": "-angle",
        "angle_class_id": "200 - angle_class_id",
        "speed": "unchanged",
        "speed_class_id": "unchanged",
    }
    assert checkpoint["preprocessing"]["geometry"] == "full_frame_bicubic_resize"
    assert (
        checkpoint["preprocessing"]["training_augmentation"][
            "horizontal_flip_probability"
        ]
        == 1.0
    )
    assert checkpoint["split_manifest"]["dataset_snapshot"] == "synthetic"
    assert checkpoint["source"]["uv_lock_sha256"] == "unknown"
    assert "optimizer_state" in checkpoint
    assert "scheduler_state" in checkpoint
    assert "scaler_state" in checkpoint
    with (run_dir / "metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as metrics_file:
        first_row = next(iter(csv.DictReader(metrics_file)))
    assert first_row["train_horizontal_flip_rate"] == "1.0"
    assert first_row["val_horizontal_flip_rate"] == "0.0"
    assert "val_angle_bucket_near_zero_mae" in first_row
    test_metrics = yaml.safe_load((run_dir / "test_metrics.json").read_text())
    assert test_metrics["test_horizontal_flip_rate"] == 0.0
    assert "test_angle_bucket_left_within_10_acc" in test_metrics

    _write_config(tmp_path, split_path, epochs=2, flip_probability=0.0)
    with pytest.raises(ValueError, match="flip probability differs"):
        main(
            [
                "--config",
                str(config_path),
                "--resume",
                str(run_dir / "last.pt"),
            ]
        )

    _write_config(tmp_path, split_path, epochs=2)
    assert (
        main(
            [
                "--config",
                str(config_path),
                "--resume",
                str(run_dir / "last.pt"),
            ]
        )
        == 0
    )
    with (run_dir / "metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert rows[1]["train_horizontal_flip_rate"] == "1.0"
    assert rows[1]["val_horizontal_flip_rate"] == "0.0"
    assert "val_angle_bucket_hard_right_within_10_acc" in rows[1]


def _write_config(
    root: Path,
    split_path: Path,
    *,
    epochs: int,
    flip_probability: float = 1.0,
) -> Path:
    config_path = root / "config" / "train.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model": {
            "name": "vit_small_patch16_224.augreg_in21k_ft_in1k",
            "pretrained": False,
            "image_size": 32,
        },
        "data": {
            "root": "datasets/teleop",
            "split_manifest": str(split_path.relative_to(root)),
            "require_all_matching_sessions": True,
            "control_mode": "gamepad",
            "max_forward_speed": 25.0,
            "num_workers": 0,
        },
        "augmentation": {
            "brightness": 0.0,
            "contrast": 0.0,
            "saturation": 0.0,
            "hue": 0.0,
            "horizontal_flip_probability": flip_probability,
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 0.0003,
            "weight_decay": 0.05,
        },
        "scheduler": {"name": "cosine", "warmup_epochs": 0},
        "training": {
            "epochs": epochs,
            "batch_size": 2,
            "grad_clip": 1.0,
            "seed": 7,
            "device": "cpu",
            "amp": False,
            "deterministic": True,
        },
        "loss": {
            "angle_label_smoothing": 0.02,
            "speed_label_smoothing": 0.02,
            "angle_class_weighting": "sqrt_inverse_frequency",
            "speed_class_weighting": "sqrt_inverse_frequency",
            "min_class_weight": 0.5,
            "max_class_weight": 3.0,
            "speed_loss_weight": 0.5,
            "emd_loss_weight": 0.2,
        },
        "output": {"root": "artifacts/runs", "run_name": "smoke"},
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path
