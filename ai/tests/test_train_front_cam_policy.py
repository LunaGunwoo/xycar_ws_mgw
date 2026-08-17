from __future__ import annotations

import csv
import copy
from pathlib import Path

import pytest
import torch
import yaml
from conftest import write_session, write_split_manifest

from xycar_ai.compare_front_cam_policy import build_promotion_report
from xycar_ai.config import load_train_config
from xycar_ai.front_cam_policy_data import PolicySample, PolicySession
from xycar_ai.train_front_cam_policy import (
    build_label_contract,
    build_preprocessing_contract,
    class_weights,
    early_stopping_triggered,
    initialize_model_weights,
    load_configured_road_warp,
    main,
    source_weighted_metric,
    validation_selection_score,
    validate_incremental_initialization,
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


def test_initialize_from_loads_model_weights_only(tmp_path: Path):
    split_path = tmp_path / "config" / "split.yaml"
    config = load_train_config(
        _write_config(tmp_path, split_path, epochs=1, flip_probability=0.0)
    )
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.fill_(2.0)
        source.bias.fill_(3.0)
        target.weight.zero_()
        target.bias.zero_()
    checkpoint_path = tmp_path / "source.pt"
    torch.save(
        {
            "schema_version": 1,
            "epoch": 7,
            "model_name": config.model.name,
            "config": config.serializable(),
            "model_state": source.state_dict(),
            "optimizer_state": {"must_not": "load"},
        },
        checkpoint_path,
    )

    metadata = initialize_model_weights(
        model=target,
        checkpoint=str(checkpoint_path),
        config=config,
        device=torch.device("cpu"),
    )

    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)
    assert metadata is not None
    assert metadata["mode"] == "model_weights_only"
    assert metadata["source_epoch"] == 7
    assert len(metadata["checkpoint_sha256"]) == 64


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
    assert config.data.train_angle_mean_window == 1


def test_guided_ema_config_uses_external_history_and_generation_decay():
    project_root = Path(__file__).parents[1]
    config = load_train_config(
        project_root / "config" / "front_cam_policy_train_guided_ema.yaml"
    )

    assert config.model.history_update == "externally_executed_commands"
    assert config.data.ema_sampling
    assert config.data.control_modes == ("gamepad", "guided_policy")
    assert config.data.current_generation == 0
    assert config.data.generation_decay == 0.5


def test_stateless_ema_config_uses_two_qualified_sources_and_raw_angle(
    tmp_path: Path,
):
    project_root = Path(__file__).parents[1]
    config = load_train_config(
        project_root / "config" / "front_cam_policy_train_stateless_ema.yaml"
    )

    assert config.model.name == "vit_small_patch16_224.augreg_in21k_ft_in1k"
    assert config.model.architecture == "task_tokens"
    assert config.model.history_frames == 0
    assert config.model.image_size == 224
    assert config.preprocessing.road_warp_config == (
        project_root / "config" / "front_cam_policy_preprocess.yaml"
    )
    assert config.data.train_angle_mean_window == 1
    assert config.data.ema_sampling
    assert config.data.current_generation == 0
    assert config.data.generation_decay == 0.5
    assert config.training.early_stopping_patience == 5
    assert [(source.source_id, source.fixed_generation) for source in config.data.sources] == [
        ("manual", 0),
        ("guided", None),
    ]
    assert config.data.sources[1].require_curriculum_generation
    assert config.output.run_name == "vit_small_stateless_manual_20260817_generation0"
    validate_incremental_initialization(
        config, initialize_from="", resume=""
    )
    with pytest.raises(ValueError, match="pretrained ImageNet"):
        validate_incremental_initialization(
            config, initialize_from="previous.pt", resume=""
        )

    generation_one = copy.deepcopy(config)
    object.__setattr__(generation_one.data, "current_generation", 1)
    with pytest.raises(ValueError, match="requires --initialize-from"):
        validate_incremental_initialization(
            generation_one, initialize_from="", resume=""
        )
    validate_incremental_initialization(
        generation_one, initialize_from="previous.pt", resume=""
    )

    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    checkpoint_path = tmp_path / "generation0-best.pt"
    torch.save(
        {
            "schema_version": 1,
            "epoch": 3,
            "model_name": config.model.name,
            "config": config.serializable(),
            "model_state": source.state_dict(),
        },
        checkpoint_path,
    )
    metadata = initialize_model_weights(
        model=target,
        checkpoint=str(checkpoint_path),
        config=generation_one,
        device=torch.device("cpu"),
    )
    assert metadata is not None
    assert metadata["mode"] == "model_weights_only"
    assert torch.equal(target.weight, source.weight)


def test_normalized_stateless_config_requires_raw_normalized_sessions(
    tmp_path: Path,
):
    project_root = Path(__file__).parents[1]
    config = load_train_config(
        project_root
        / "config"
        / "front_cam_policy_train_stateless_normalized_v1.yaml"
    )

    assert (
        config.data.required_steering_contract
        == "normalized_percent_v1"
    )
    assert config.data.train_angle_mean_window == 1
    assert config.data.current_generation == 0
    assert config.data.root == project_root / "datasets/stateless_manual"
    assert not config.data.sources
    assert config.data.minimum_total_samples == 10000
    assert config.data.minimum_total_sessions == 11
    assert config.data.minimum_train_sessions == 7
    assert config.data.minimum_val_sessions == 2
    assert config.data.minimum_test_sessions == 2
    assert config.output.run_name == (
        "vit_small_stateless_normalized_steering_v1_generation0"
    )

    guided_config = load_train_config(
        project_root
        / "config"
        / "front_cam_policy_train_stateless_normalized_v1_g1.yaml"
    )
    assert guided_config.data.required_steering_contract == (
        "normalized_percent_v1"
    )
    assert guided_config.data.current_generation == 1
    assert guided_config.data.generation_decay == 0.8
    assert guided_config.data.source_sampling_masses == {
        "manual": 0.5,
        "guided": 0.5,
    }
    assert guided_config.data.manual_anchor_split_manifest == (
        project_root
        / "config"
        / "front_cam_policy_split_stateless_normalized_v1.yaml"
    )
    assert guided_config.data.current_generation_session_counts == {
        "train": 3,
        "val": 1,
        "test": 1,
    }
    assert guided_config.data.train_angle_mean_window == 1
    assert {source.source_id for source in guided_config.data.sources} == {
        "manual",
        "guided",
    }
    assert guided_config.optimizer.learning_rate == 0.0001
    assert guided_config.training.epochs == 10
    assert guided_config.training.early_stopping_patience == 3
    assert guided_config.loss.angle_label_smoothing == 0.02
    assert guided_config.loss.speed_label_smoothing == 0.02
    assert guided_config.output.run_name.endswith("generation1")

    expanded_payload = yaml.safe_load(
        (
            project_root
            / "config"
            / "front_cam_policy_train_stateless_normalized_v1_g1.yaml"
        ).read_text(encoding="utf-8")
    )
    expanded_payload["data"]["current_generation_session_counts"] = {
        "train": 12,
        "val": 2,
        "test": 2,
    }
    expanded_config_path = tmp_path / "expanded-guided.yaml"
    expanded_config_path.write_text(
        yaml.safe_dump(expanded_payload, sort_keys=False), encoding="utf-8"
    )
    expanded_config = load_train_config(expanded_config_path)
    assert expanded_config.data.current_generation_session_counts == {
        "train": 12,
        "val": 2,
        "test": 2,
    }


@pytest.mark.parametrize(
    ("masses", "message"),
    [
        ({"manual": 0.5}, "exactly match"),
        ({"manual": 0.6, "guided": 0.5}, "sum to 1"),
        ({"manual": 1.0, "guided": 0.0}, "finite and positive"),
    ],
)
def test_source_sampling_masses_fail_closed(
    tmp_path: Path, masses: dict[str, float], message: str
):
    project_root = Path(__file__).parents[1]
    payload = yaml.safe_load(
        (
            project_root
            / "config"
            / "front_cam_policy_train_stateless_normalized_v1_g1.yaml"
        ).read_text(encoding="utf-8")
    )
    payload["data"]["source_sampling_masses"] = masses
    payload["data"]["split_manifest"] = str(tmp_path / "split.yaml")
    payload["preprocessing"]["road_warp_config"] = str(
        project_root / "config" / "front_cam_policy_preprocess.yaml"
    )
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        load_train_config(config_path)


def test_promotion_report_requires_all_regression_and_lineage_gates():
    parent = {
        "weighted_validation_score": 10.0,
        "guided_angle_mae": 12.0,
        "manual_angle_mae": 8.0,
        "manual_angle_within_10_acc": 0.8,
        "manual_speed_mae": 2.0,
        "guided_speed_mae": 3.0,
    }
    candidate = {
        "weighted_validation_score": 9.0,
        "guided_angle_mae": 10.0,
        "manual_angle_mae": 10.0,
        "manual_angle_within_10_acc": 0.77,
        "manual_speed_mae": 3.0,
        "guided_speed_mae": 4.0,
    }
    report = build_promotion_report(
        generation=1,
        parent_checkpoint="parent.pt",
        parent_sha256="parent-hash",
        candidate_checkpoint="candidate.pt",
        candidate_sha256="candidate-hash",
        initialization_sha256="parent-hash",
        parent_summary=parent,
        candidate_summary=candidate,
    )

    assert report["status"] == "passed"
    assert all(report["checks"].values())

    failed = build_promotion_report(
        generation=1,
        parent_checkpoint="parent.pt",
        parent_sha256="parent-hash",
        candidate_checkpoint="candidate.pt",
        candidate_sha256="candidate-hash",
        initialization_sha256="different-parent",
        parent_summary=parent,
        candidate_summary={**candidate, "manual_angle_mae": 10.01},
    )
    assert failed["status"] == "failed"
    assert not failed["checks"]["candidate_initialized_from_parent"]
    assert not failed["checks"]["manual_angle_mae_regression_within_limit"]


def test_source_anchored_validation_score_uses_same_half_and_half_mass():
    project_root = Path(__file__).parents[1]
    config = load_train_config(
        project_root
        / "config"
        / "front_cam_policy_train_stateless_normalized_v1_g1.yaml"
    )
    sessions = (
        _metric_session(source_id="manual", generation=0),
        _metric_session(source_id="guided", generation=1),
    )
    metrics = {
        "val_source_manual_generation_0_angle_mae": 8.0,
        "val_source_manual_generation_0_angle_within_10_acc": 0.8,
        "val_source_manual_generation_0_speed_mae": 2.0,
        "val_source_guided_generation_1_angle_mae": 12.0,
        "val_source_guided_generation_1_angle_within_10_acc": 0.7,
        "val_source_guided_generation_1_speed_mae": 4.0,
    }

    assert validation_selection_score(
        metrics, sessions=sessions, config=config
    ) == pytest.approx(10.75)
    assert source_weighted_metric(
        metrics,
        sessions=sessions,
        config=config,
        source_id="manual",
        metric_name="angle_mae",
    ) == 8.0
    assert source_weighted_metric(
        metrics,
        sessions=sessions,
        config=config,
        source_id="guided",
        metric_name="angle_mae",
    ) == 12.0


def test_class_weights_use_source_anchored_sample_mass():
    project_root = Path(__file__).parents[1]
    config = load_train_config(
        project_root
        / "config"
        / "front_cam_policy_train_stateless_normalized_v1_g1.yaml"
    )
    samples = (
        _metric_sample(source_id="manual", generation=0, speed_class_id=110),
        _metric_sample(source_id="manual", generation=0, speed_class_id=110),
        _metric_sample(source_id="guided", generation=1, speed_class_id=120),
    )

    weights = class_weights(
        samples,
        field="speed_class_id",
        mode="sqrt_inverse_frequency",
        config=config,
        device=torch.device("cpu"),
    )

    assert weights is not None
    assert weights[110] == pytest.approx(weights[120])


def test_stateless_two_root_config_validate_only_with_synthetic_sessions(
    tmp_path: Path,
):
    project_root = Path(__file__).parents[1]
    manual_root = tmp_path / "datasets" / "stateless_manual"
    guided_root = tmp_path / "datasets" / "stateless_guided"
    guided_root.mkdir(parents=True)
    names = [
        "20260813_010101_001_session",
        "20260813_010102_001_session",
        "20260813_010103_001_session",
    ]
    for name in names:
        write_session(manual_root, name, labels=[(0.0, 7.0)])
    split_path = write_split_manifest(
        tmp_path / "config" / "split.yaml",
        train=[f"manual/{names[0]}"],
        val=[f"manual/{names[1]}"],
        test=[f"manual/{names[2]}"],
        schema_version=2,
    )
    payload = yaml.safe_load(
        (
            project_root
            / "config"
            / "front_cam_policy_train_stateless_ema.yaml"
        ).read_text(encoding="utf-8")
    )
    payload["data"]["sources"]["manual"]["root"] = str(manual_root)
    payload["data"]["sources"]["guided"]["root"] = str(guided_root)
    payload["data"]["split_manifest"] = str(split_path)
    payload["data"]["current_generation"] = 0
    payload["output"]["run_name"] = "vit_small_stateless_ema_generation0"
    payload["preprocessing"]["road_warp_config"] = str(
        project_root / "config" / "front_cam_policy_preprocess.yaml"
    )
    payload["output"]["root"] = str(tmp_path / "artifacts")
    config_path = tmp_path / "config" / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )

    assert main(["--config", str(config_path), "--validate-only"]) == 0


def test_small_warp_config_embeds_preprocessing_contract():
    project_root = Path(__file__).parents[1]
    config = load_train_config(
        project_root / "config" / "front_cam_policy_train_small_warp.yaml"
    )
    road_warp = load_configured_road_warp(config)

    assert road_warp is not None
    assert config.data.train_angle_mean_window == 5
    assert config.output.run_name == (
        "vit_small_warp_angle_mean5_hflip_p05_seed20260811"
    )
    label_contract = build_label_contract(config)
    assert label_contract["train_angle_target"] == {
        "method": "centered_mean",
        "window_size": 5,
        "padding": "repeat_session_edge",
        "applied_splits": ["train"],
        "average_before_quantization": True,
        "speed_target": "unchanged",
    }
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
    assert contract["geometry"] == ("perspective_road_warp_then_bicubic_resize")
    assert contract["road_warp"]["parameters"]["bev_width"] == 224
    assert len(contract["road_warp"]["sha256"]) == 64
    assert contract["training_augmentation"]["horizontal_flip_after_road_warp"] is True

    split = {"schema_version": 1, "dataset_snapshot": "synthetic", "splits": {}}
    checkpoint = {
        "schema_version": 1,
        "model_name": config.model.name,
        "label_contract": label_contract,
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
        expected_label_contract=label_contract,
    )
    changed_contract = copy.deepcopy(contract)
    changed_contract["road_warp"]["parameters"]["top_y"] = 0.7
    with pytest.raises(ValueError, match="preprocessing differs"):
        validate_resume_payload(
            checkpoint,
            config=config,
            expected_split=split,
            expected_preprocessing=changed_contract,
            expected_label_contract=label_contract,
        )

    changed_label_contract = copy.deepcopy(label_contract)
    changed_label_contract["train_angle_target"]["window_size"] = 3
    with pytest.raises(ValueError, match="train angle target differs"):
        validate_resume_payload(
            checkpoint,
            config=config,
            expected_split=split,
            expected_preprocessing=contract,
            expected_label_contract=changed_label_contract,
        )


def test_ar_probe_configs_only_change_type_embedding_and_run_name():
    project_root = Path(__file__).parents[1]
    shared = yaml.safe_load(
        (
            project_root / "config" / "front_cam_policy_train_small_warp_ar_shared.yaml"
        ).read_text()
    )
    shared_type = yaml.safe_load(
        (
            project_root
            / "config"
            / "front_cam_policy_train_small_warp_ar_shared_type.yaml"
        ).read_text()
    )
    assert shared["model"]["architecture"] == "ar_control_tokens"
    assert shared["model"]["history_frames"] == 4
    assert shared["model"]["history_initial_angle"] == 0
    assert shared["model"]["history_initial_speed"] == 25
    assert shared["model"]["control_token_type_embedding"] is False
    assert shared_type["model"]["control_token_type_embedding"] is True
    assert shared["training"]["epochs"] == shared_type["training"]["epochs"] == 20
    assert shared["data"]["train_angle_mean_window"] == 5
    shared["model"]["control_token_type_embedding"] = True
    shared["output"]["run_name"] = shared_type["output"]["run_name"]
    assert shared == shared_type


def test_speed_balanced_configs_share_warp_and_session_disjoint_split():
    project_root = Path(__file__).parents[1]
    split = yaml.safe_load(
        (
            project_root
            / "config"
            / "front_cam_policy_split_min_speed20_speed_balanced.yaml"
        ).read_text()
    )
    groups = split["splits"]
    assert [len(groups[name]) for name in ("train", "val", "test")] == [7, 2, 2]
    assert not (set(groups["train"]) & set(groups["val"]))
    assert not (set(groups["train"]) & set(groups["test"]))
    assert not (set(groups["val"]) & set(groups["test"]))
    deceleration_sessions = {
        "20260810_130735_027_session",
        "20260810_131011_524_session",
        "20260810_131116_609_session",
    }
    assert [
        len(set(groups[name]) & deceleration_sessions)
        for name in ("train", "val", "test")
    ] == [1, 1, 1]

    for config_name in (
        "front_cam_policy_train_small_warp_speed_balanced.yaml",
        "front_cam_policy_train_small_warp_ar_shared_speed_balanced.yaml",
    ):
        config = load_train_config(project_root / "config" / config_name)
        assert config.preprocessing.road_warp_config == (
            project_root / "config" / "front_cam_policy_preprocess.yaml"
        )
        assert config.data.split_manifest.name == (
            "front_cam_policy_split_min_speed20_speed_balanced.yaml"
        )
        assert config.data.train_angle_mean_window == 5


def test_config_requires_exactly_one_forward_speed_filter(tmp_path: Path):
    split_path = tmp_path / "config" / "split.yaml"
    config_path = _write_config(tmp_path, split_path, epochs=1)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["data"]["min_forward_speed"] = 20.0
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        load_train_config(config_path)


@pytest.mark.parametrize("window_size", [0, -1, 2, 4])
def test_config_rejects_invalid_train_angle_mean_window(
    tmp_path: Path, window_size: int
):
    config_path = _write_config(tmp_path, tmp_path / "config" / "split.yaml", epochs=1)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["data"]["train_angle_mean_window"] = window_size
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="positive odd integer"):
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


def test_config_rejects_noncanonical_ar_initial_history(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        tmp_path / "config" / "split.yaml",
        epochs=1,
        autoregressive=True,
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["model"]["history_initial_speed"] = 24
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"\(0, 25\)"):
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


def test_ar_probe_resume_preserves_scheduler_and_uses_rollout_evaluation(
    tmp_path: Path,
):
    data_root = tmp_path / "datasets" / "teleop"
    names = [
        "20260803_150534_000_session",
        "20260810_130735_027_session",
        "20260810_130818_255_session",
        "20260803_150428_657_session",
    ]
    for name in names:
        write_session(data_root, name, labels=[(0.0, 25.0)])
    split_path = write_split_manifest(
        tmp_path / "config" / "split.yaml",
        train=names[:2],
        val=[names[2]],
        test=[names[3]],
    )
    config_path = _write_config(
        tmp_path,
        split_path,
        epochs=2,
        flip_probability=0.0,
        autoregressive=True,
    )

    assert (
        main(
            [
                "--config",
                str(config_path),
                "--stop-after-epoch",
                "1",
            ]
        )
        == 0
    )
    run_dir = tmp_path / "artifacts" / "runs" / "smoke"
    probe = yaml.safe_load((run_dir / "probe_summary.json").read_text())
    first_checkpoint = torch.load(
        run_dir / "last.pt", map_location="cpu", weights_only=True
    )
    assert probe["completed_epochs"] == 1
    assert not (run_dir / "test_metrics.json").exists()
    assert first_checkpoint["label_contract"]["history"]["evaluation_source"] == (
        "predicted_argmax_rollout"
    )
    assert first_checkpoint["scheduler_state"]["last_epoch"] == 1

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
    final_checkpoint = torch.load(
        run_dir / "last.pt", map_location="cpu", weights_only=True
    )
    assert final_checkpoint["epoch"] == 2
    assert final_checkpoint["scheduler_state"]["last_epoch"] == 2
    assert (run_dir / "test_metrics.json").is_file()


def _write_config(
    root: Path,
    split_path: Path,
    *,
    epochs: int,
    flip_probability: float = 1.0,
    autoregressive: bool = False,
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
    if autoregressive:
        payload["model"].update(
            {
                "architecture": "ar_control_tokens",
                "history_frames": 4,
                "control_token_type_embedding": False,
                "history_initial_angle": 0,
                "history_initial_speed": 25,
            }
        )
        payload["data"]["train_angle_mean_window"] = 5
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _metric_session(*, source_id: str, generation: int) -> PolicySession:
    sample = _metric_sample(source_id=source_id, generation=generation)
    return PolicySession(
        session_id=sample.session_id,
        path=Path(source_id),
        metadata={},
        samples=(sample,),
        generation=generation,
        source_id=source_id,
    )


def _metric_sample(
    *, source_id: str, generation: int, speed_class_id: int = 100
) -> PolicySample:
    return PolicySample(
        session_id=f"{source_id}/session",
        image_path=Path("unused.png"),
        relative_image=f"{source_id}/unused.png",
        angle_raw=0.0,
        speed_raw=0.0,
        angle=0,
        speed=0,
        angle_class_id=100,
        speed_class_id=speed_class_id,
        generation=generation,
        source_id=source_id,
    )
