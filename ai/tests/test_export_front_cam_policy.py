from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

import xycar_ai.export_front_cam_policy as export_module
from xycar_ai.export_front_cam_policy import (
    PolicyExportError,
    export_checkpoint,
    verify_artifact,
)


class _TinyPolicy(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        pretrained: bool,
        image_size: int,
    ) -> None:
        super().__init__()
        del model_name, pretrained, image_size
        self.angle_bias = nn.Parameter(torch.zeros(201))
        self.speed_bias = nn.Parameter(torch.ones(201))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = images.shape[0]
        return {
            "angle_logits": self.angle_bias.expand(batch_size, -1),
            "speed_logits": self.speed_bias.expand(batch_size, -1),
        }


class _TinyARPolicy(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        pretrained: bool,
        image_size: int,
        history_frames: int,
        use_control_type_embedding: bool,
        control_encoding: str = "legacy_command_201",
        prediction_mode: str = "categorical",
        speed_output_max: float = 30.0,
    ) -> None:
        super().__init__()
        del (
            model_name,
            pretrained,
            image_size,
            use_control_type_embedding,
            speed_output_max,
        )
        self.history_frames = history_frames
        compact = control_encoding == "driver_compact_v2"
        self.prediction_mode = prediction_mode
        if prediction_mode == "continuous_regression":
            self.angle_bias = nn.Parameter(torch.zeros(1))
            self.speed_bias = nn.Parameter(torch.ones(1))
        else:
            self.angle_bias = nn.Parameter(torch.zeros(101 if compact else 201))
            self.speed_bias = nn.Parameter(torch.ones(31 if compact else 201))

    def forward(
        self,
        images: torch.Tensor,
        history_class_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = images.shape[0]
        history_signal = history_class_ids[:, -1, :].to(images.dtype)
        if self.prediction_mode == "continuous_regression":
            return {
                "angle_driver": self.angle_bias.expand(batch_size, -1),
                "speed": self.speed_bias.expand(batch_size, -1) * 15.0,
            }
        return {
            "angle_logits": self.angle_bias.expand(batch_size, -1)
            + history_signal[:, :1],
            "speed_logits": self.speed_bias.expand(batch_size, -1)
            + history_signal[:, 1:],
        }


def test_export_checkpoint_writes_verified_artifact(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(export_module, "TaskTokenViTPolicy", _TinyPolicy)
    checkpoint_path = tmp_path / "best.pt"
    model = _TinyPolicy(model_name="tiny", pretrained=False, image_size=16)
    torch.save(
        {
            "epoch": 6,
            "best_epoch": 6,
            "best_score": 12.5,
            "config": {
                "model": {"name": "tiny", "image_size": 16},
                "data": {"required_steering_contract": "normalized_percent_v2"},
            },
            "model_state": model.state_dict(),
            "preprocessing": {
                "geometry": "full_frame_bicubic_resize",
                "image_size": 16,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "label_contract": {
                "num_classes": 201,
                "decode_mapping": "class_id - 100",
            },
            "dataset_stats": {"dataset_snapshot": "fixture"},
            "source": {"mgw_commit": "abc", "dirty": True},
        },
        checkpoint_path,
    )

    artifact = export_checkpoint(
        checkpoint_path=checkpoint_path,
        artifact_id="fixture-policy",
        output_root=tmp_path / "models",
        require_schema_version=1,
    )

    assert {path.name for path in artifact.iterdir()} == {
        "model.ts",
        "manifest.yaml",
        "SHA256SUMS",
    }
    verify_artifact(artifact)
    manifest = yaml.safe_load((artifact / "manifest.yaml").read_text())
    assert manifest["schema_version"] == 1
    assert "history" not in manifest
    assert manifest["artifact_id"] == "fixture-policy"
    assert manifest["source"]["best_epoch"] == 6
    assert manifest["model"]["input"]["shape"] == [1, 3, 16, 16]
    assert manifest["steering_contract"] == {
        "schema_version": 1,
        "name": "normalized_percent_v2",
        "command_min": -100.0,
        "command_max": 100.0,
        "driver_min": -50.0,
        "driver_max": 50.0,
        "mapping": "linear_scale_0.5",
    }
    model_ts = torch.jit.load(str(artifact / "model.ts"), map_location="cpu")
    angle, speed = model_ts(torch.zeros(1, 3, 16, 16))
    assert tuple(angle.shape) == (1, 201)
    assert tuple(speed.shape) == (1, 201)

    with pytest.raises(FileExistsError):
        export_checkpoint(
            checkpoint_path=checkpoint_path,
            artifact_id="fixture-policy",
            output_root=tmp_path / "models",
        )

    alias = tmp_path / "artifact-alias"
    alias.symlink_to(artifact, target_is_directory=True)
    with pytest.raises(PolicyExportError, match="must not be a symlink"):
        verify_artifact(alias)

    legacy_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    del legacy_checkpoint["config"]["data"]
    legacy_path = tmp_path / "legacy-best.pt"
    torch.save(legacy_checkpoint, legacy_path)
    with pytest.raises(PolicyExportError, match="checkpoint.config.data"):
        export_checkpoint(
            checkpoint_path=legacy_path,
            artifact_id="legacy-policy",
            output_root=tmp_path / "models",
        )


def test_export_validation_rejects_unsafe_id_and_tampering(tmp_path: Path):
    with pytest.raises(PolicyExportError, match="invalid artifact id"):
        export_checkpoint(
            checkpoint_path=tmp_path / "missing.pt",
            artifact_id="../unsafe",
            output_root=tmp_path,
        )

    artifact = tmp_path / "tampered"
    artifact.mkdir()
    (artifact / "manifest.yaml").write_text("changed\n", encoding="utf-8")
    (artifact / "model.ts").write_bytes(b"model")
    (artifact / "SHA256SUMS").write_text(
        f"{'0' * 64}  manifest.yaml\n{'0' * 64}  model.ts\n",
        encoding="utf-8",
    )
    with pytest.raises(PolicyExportError, match="checksum mismatch"):
        verify_artifact(artifact)


def test_guided_export_records_advisory_promotion_status(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(export_module, "TaskTokenViTPolicy", _TinyPolicy)
    checkpoint_path = tmp_path / "guided-best.pt"
    model = _TinyPolicy(model_name="tiny", pretrained=False, image_size=16)
    torch.save(
        {
            "epoch": 3,
            "config": {
                "model": {"name": "tiny", "image_size": 16},
                "data": {
                    "required_steering_contract": "normalized_percent_v2",
                    "current_generation": 1,
                },
            },
            "model_state": model.state_dict(),
            "preprocessing": {
                "geometry": "full_frame_bicubic_resize",
                "image_size": 16,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "label_contract": {
                "num_classes": 201,
                "decode_mapping": "class_id - 100",
            },
        },
        checkpoint_path,
    )

    artifact_without_report = export_checkpoint(
        checkpoint_path=checkpoint_path,
        artifact_id="guided-without-gate",
        output_root=tmp_path / "models",
    )
    manifest_without_report = yaml.safe_load(
        (artifact_without_report / "manifest.yaml").read_text()
    )
    assert manifest_without_report["promotion"] == {
        "offline_gate": "not_evaluated",
        "generation": 1,
        "candidate_checkpoint_sha256": export_module.sha256_file(checkpoint_path),
    }

    report_path = tmp_path / "promotion_gate.json"
    report_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "status": "failed",
                "generation": 1,
                "parent": {"sha256": "parent-hash"},
                "candidate": {"sha256": export_module.sha256_file(checkpoint_path)},
                "checks": {
                    "guided_angle_mae_improved": False,
                    "candidate_initialized_from_parent": True,
                },
            }
        ),
        encoding="utf-8",
    )
    failed_artifact = export_checkpoint(
        checkpoint_path=checkpoint_path,
        artifact_id="guided-with-failed-gate",
        output_root=tmp_path / "models",
        promotion_report_path=report_path,
    )
    failed_manifest = yaml.safe_load((failed_artifact / "manifest.yaml").read_text())
    assert failed_manifest["promotion"]["offline_gate"] == "failed"
    assert failed_manifest["promotion"]["generation"] == 1
    assert failed_manifest["promotion"]["parent_checkpoint_sha256"] == ("parent-hash")
    assert failed_manifest["promotion"]["failed_checks"] == [
        "guided_angle_mae_improved"
    ]

    report_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "status": "passed",
                "generation": 1,
                "parent": {"sha256": "parent-hash"},
                "candidate": {"sha256": export_module.sha256_file(checkpoint_path)},
                "checks": {"all_required_checks": True},
            }
        ),
        encoding="utf-8",
    )
    passed_artifact = export_checkpoint(
        checkpoint_path=checkpoint_path,
        artifact_id="guided-with-passed-gate",
        output_root=tmp_path / "models",
        promotion_report_path=report_path,
    )
    passed_manifest = yaml.safe_load((passed_artifact / "manifest.yaml").read_text())
    assert passed_manifest["promotion"]["offline_gate"] == "passed"
    assert passed_manifest["promotion"]["failed_checks"] == []

    mismatched_report = yaml.safe_load(report_path.read_text())
    mismatched_report["candidate"]["sha256"] = "wrong-hash"
    report_path.write_text(yaml.safe_dump(mismatched_report), encoding="utf-8")
    with pytest.raises(PolicyExportError, match="candidate hash differs"):
        export_checkpoint(
            checkpoint_path=checkpoint_path,
            artifact_id="guided-with-mismatched-gate",
            output_root=tmp_path / "models",
            promotion_report_path=report_path,
        )


def test_export_ar_checkpoint_writes_v3_external_history_contract(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        export_module,
        "AutoregressiveControlTokenViTPolicy",
        _TinyARPolicy,
    )
    checkpoint_path = tmp_path / "best.pt"
    model = _TinyARPolicy(
        model_name="tiny",
        pretrained=False,
        image_size=16,
        history_frames=4,
        use_control_type_embedding=True,
    )
    torch.save(
        {
            "epoch": 5,
            "best_epoch": 4,
            "best_score": 10.0,
            "config": {
                "model": {
                    "name": "tiny",
                    "image_size": 16,
                    "architecture": "ar_control_tokens",
                    "history_frames": 4,
                    "control_token_type_embedding": True,
                    "history_initial_angle": 0,
                    "history_initial_speed": 25,
                },
                "data": {"required_steering_contract": "normalized_percent_v2"},
            },
            "model_state": model.state_dict(),
            "preprocessing": {
                "geometry": "full_frame_bicubic_resize",
                "image_size": 16,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "label_contract": {
                "num_classes": 201,
                "decode_mapping": "class_id - 100",
            },
        },
        checkpoint_path,
    )

    with pytest.raises(PolicyExportError, match="required 1"):
        export_checkpoint(
            checkpoint_path=checkpoint_path,
            artifact_id="fixture-ar-rejected",
            output_root=tmp_path / "models",
            require_schema_version=1,
        )

    artifact = export_checkpoint(
        checkpoint_path=checkpoint_path,
        artifact_id="fixture-ar-policy",
        output_root=tmp_path / "models",
    )
    manifest = yaml.safe_load((artifact / "manifest.yaml").read_text())
    assert manifest["schema_version"] == 3
    assert manifest["model"]["input"] == {
        "kind": "tuple",
        "order": ["images", "history_class_ids"],
        "images": {
            "color_space": "RGB",
            "dtype": "float32",
            "shape": [1, 3, 16, 16],
        },
        "history_class_ids": {"dtype": "int64", "shape": [1, 4, 2]},
    }
    assert manifest["history"]["initial_class_ids"] == [100, 125]
    assert manifest["history"]["update"] == "externally_executed_commands"
    assert manifest["steering_contract"]["name"] == "normalized_percent_v2"
    model_ts = torch.jit.load(str(artifact / "model.ts"), map_location="cpu")
    angle, speed = model_ts(
        torch.zeros(1, 3, 16, 16),
        torch.tensor([[[100, 125]] * 4], dtype=torch.long),
    )
    assert tuple(angle.shape) == (1, 201)
    assert tuple(speed.shape) == (1, 201)


@pytest.mark.parametrize(
    ("initialization", "initial_ids", "initial_command", "angle_only"),
    [
        ("learned_unknown_tokens", [101, 102], None, False),
        ("canonical_initial_command", [50, 65], [0, 15], False),
        ("canonical_initial_command", [50, 65], [0, 15], True),
    ],
)
def test_export_compact_ar_checkpoint_writes_schema_v5(
    monkeypatch,
    tmp_path: Path,
    initialization: str,
    initial_ids: list[int],
    initial_command: list[int] | None,
    angle_only: bool,
):
    monkeypatch.setattr(
        export_module,
        "AutoregressiveControlTokenViTPolicy",
        _TinyARPolicy,
    )
    checkpoint_path = tmp_path / "compact.pt"
    model = _TinyARPolicy(
        model_name="tiny",
        pretrained=False,
        image_size=16,
        history_frames=4,
        use_control_type_embedding=False,
        control_encoding="driver_compact_v2",
    )
    label_contract = {
        "schema_version": 3,
        "control_encoding": "driver_compact_v2",
        "output_shapes": {
            "angle_logits": [1, 101],
            "speed_logits": [1, 31],
        },
        "angle": {"num_classes": 101, "driver_range": [-50, 50]},
        "speed": {"num_classes": 31, "command_range": [0, 30]},
        "shared_numeric_vocabulary": {
            "numeric_range": [-50, 50],
            "unknown_angle_token_id": 101,
            "unknown_speed_token_id": 102,
            "angle_query_token_id": 103,
            "speed_query_token_id": 104,
            "vocabulary_size": 105,
        },
        "history": {
            "initialization": initialization,
            "initial_token_ids": initial_ids,
        },
    }
    if initial_command is not None:
        label_contract["history"]["initial_command"] = initial_command
    checkpoint = {
        "epoch": 1,
        "config": {
            "model": {
                "name": "tiny",
                "image_size": 16,
                "architecture": "ar_control_tokens",
                "control_encoding": "driver_compact_v2",
                "history_frames": 4,
                "control_token_type_embedding": False,
            },
            "data": {"required_steering_contract": "normalized_percent_v2"},
        },
        "model_state": model.state_dict(),
        "preprocessing": {
            "geometry": "full_frame_bicubic_resize",
            "image_size": 16,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "label_contract": label_contract,
    }
    if angle_only:
        checkpoint["training_objective"] = {
            "mode": "angle_only",
            "speed_output_trained": False,
            "speed_loss_weight": 0.0,
            "validation_speed_mae_weight": 0.0,
        }
        checkpoint["dataset_stats"] = {
            "all": {"sample_count": 24_675, "speed_range": [15.0, 15.0]}
        }
    torch.save(checkpoint, checkpoint_path)

    artifact = export_checkpoint(
        checkpoint_path=checkpoint_path,
        artifact_id="compact-ar-policy",
        output_root=tmp_path / "models",
        require_schema_version=5,
    )
    manifest = yaml.safe_load((artifact / "manifest.yaml").read_text())
    assert manifest["schema_version"] == 5
    assert manifest["model"]["input"]["order"] == [
        "images",
        "history_token_ids",
    ]
    assert manifest["model"]["output"]["shapes"] == [[1, 101], [1, 31]]
    assert manifest["history"]["initialization"] == initialization
    assert manifest["history"]["initial_token_ids"] == initial_ids
    assert manifest["history"]["update"] == "externally_executed_commands"
    if angle_only:
        assert manifest["training_objective"] == {
            "mode": "angle_only",
            "speed_output_trained": False,
            "speed_loss_weight": 0.0,
            "validation_speed_mae_weight": 0.0,
        }
        assert manifest["speed_output"] == {
            "mode": "fixed_class",
            "command": 15,
            "class_id": 15,
            "checkpoint_head_trained": False,
        }
    else:
        assert "training_objective" not in manifest
        assert "speed_output" not in manifest
    model_ts = torch.jit.load(str(artifact / "model.ts"), map_location="cpu")
    angle, speed = model_ts(
        torch.zeros(1, 3, 16, 16),
        torch.tensor([[initial_ids] * 4], dtype=torch.long),
    )
    assert tuple(angle.shape) == (1, 101)
    assert tuple(speed.shape) == (1, 31)
    assert int(speed.argmax(dim=1).item()) == (15 if angle_only else 0)


def test_export_regression_checkpoint_writes_schema_v6(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        export_module,
        "AutoregressiveControlTokenViTPolicy",
        _TinyARPolicy,
    )
    checkpoint_path = tmp_path / "regression.pt"
    model = _TinyARPolicy(
        model_name="tiny",
        pretrained=False,
        image_size=16,
        history_frames=4,
        use_control_type_embedding=False,
        control_encoding="driver_compact_v2",
        prediction_mode="continuous_regression",
    )
    label_contract = {
        "schema_version": 4,
        "control_encoding": "driver_compact_v2",
        "prediction_mode": "continuous_regression",
        "output_shapes": {"angle_driver": [1, 1], "speed": [1, 1]},
        "angle": {
            "unit": "driver_angle",
            "range": [-50.0, 50.0],
            "runtime_normalized_mapping": "angle_driver * 2",
        },
        "speed": {"unit": "motor_speed", "range": [0.0, 30.0]},
        "history": {
            "initialization": "canonical_initial_command",
            "initial_command": [0, 25],
            "initial_token_ids": [50, 75],
            "actual_speed_token_range": [50, 80],
        },
    }
    torch.save(
        {
            "epoch": 2,
            "config": {
                "model": {
                    "name": "tiny",
                    "image_size": 16,
                    "architecture": "ar_control_tokens",
                    "control_encoding": "driver_compact_v2",
                    "prediction_mode": "continuous_regression",
                    "history_frames": 4,
                    "control_token_type_embedding": False,
                    "history_initial_angle": 0,
                    "history_initial_speed": 25,
                },
                "data": {"required_steering_contract": "normalized_percent_v2"},
            },
            "model_state": model.state_dict(),
            "preprocessing": {
                "geometry": "full_frame_bicubic_resize",
                "image_size": 16,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "label_contract": label_contract,
            "training_objective": {
                "mode": "joint_angle_speed_regression",
                "speed_output_trained": True,
                "speed_loss_weight": 0.5,
                "validation_speed_mae_weight": 0.25,
                "loss": "smooth_l1_normalized",
                "angle_normalization": 50.0,
                "speed_normalization": 30.0,
                "angle_beta": 0.1,
                "speed_beta": 1.0 / 30.0,
            },
        },
        checkpoint_path,
    )

    artifact = export_checkpoint(
        checkpoint_path=checkpoint_path,
        artifact_id="regression-policy",
        output_root=tmp_path / "models",
        require_schema_version=6,
    )
    manifest = yaml.safe_load((artifact / "manifest.yaml").read_text())
    assert manifest["schema_version"] == 6
    assert manifest["model"]["prediction_mode"] == "continuous_regression"
    assert manifest["model"]["output"]["order"] == ["angle_driver", "speed"]
    assert manifest["model"]["output"]["shapes"] == [[1, 1], [1, 1]]
    assert manifest["history"]["initial_token_ids"] == [50, 75]
    model_ts = torch.jit.load(str(artifact / "model.ts"), map_location="cpu")
    angle, speed = model_ts(
        torch.zeros(1, 3, 16, 16),
        torch.tensor([[[50, 75]] * 4], dtype=torch.long),
    )
    assert angle.item() == pytest.approx(0.0)
    assert speed.item() == pytest.approx(15.0)

    speed_35_checkpoint = tmp_path / "regression-speed-35.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["config"]["model"]["speed_output_max"] = 35.0
    payload["config"]["model"]["history_initial_speed"] = 35
    payload["label_contract"]["speed"]["range"] = [0.0, 35.0]
    payload["label_contract"]["history"]["initial_command"] = [0, 35]
    payload["label_contract"]["history"]["initial_token_ids"] = [50, 85]
    payload["label_contract"]["history"]["actual_speed_token_range"] = [50, 85]
    payload["training_objective"]["speed_normalization"] = 35.0
    payload["training_objective"]["speed_beta"] = 1.0 / 35.0
    torch.save(payload, speed_35_checkpoint)
    speed_35_artifact = export_checkpoint(
        checkpoint_path=speed_35_checkpoint,
        artifact_id="regression-speed-35-policy",
        output_root=tmp_path / "models",
        require_schema_version=6,
    )
    speed_35_manifest = yaml.safe_load(
        (speed_35_artifact / "manifest.yaml").read_text()
    )
    assert speed_35_manifest["model"]["output"]["values"][1]["range"] == [
        0.0,
        35.0,
    ]
    assert speed_35_manifest["history"]["initial_token_ids"] == [50, 85]
    assert speed_35_manifest["history"]["actual_speed_token_range"] == [50, 85]
    speed_35_model = torch.jit.load(
        str(speed_35_artifact / "model.ts"),
        map_location="cpu",
    )
    _, speed_35 = speed_35_model(
        torch.zeros(1, 3, 16, 16),
        torch.tensor([[[50, 85]] * 4], dtype=torch.long),
    )
    assert 0.0 <= speed_35.item() <= 35.0
