from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import pytest
import yaml
from torch import nn

from xycar_ai.competition_models import (
    ShortcutModelConfig,
    ShortcutTemporalPolicy,
    SignalModelConfig,
    SignalTemporalPolicy,
)
from xycar_ai.export_competition_policy import (
    build_bundle,
    export_temporal_policy,
    verify_bundle,
)
from xycar_ai.export_traffic_shortcut_bundle import (
    ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    EXPANDED_SHORTCUT_ID,
    HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    HUMAN_BBOX_CLASSIFIER_SHA256,
    HUMAN_BBOX_TRAFFIC_SHA256,
    STABILIZED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    STOP10_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    SPEED35_BASE_ID,
    SPEED35_FIX_BASE_ID,
    SPEED35_FIX_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    SPEED35_FIX_INITIAL_WAIT_GO1_SESSION_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    SPEED35_STOP10_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    SPEED35_STOP15_GO15_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    SPEED35_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    SPEED35_INITIAL_STOP_ONCE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    SPEED35_INITIAL_WAIT_FRESH3_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    SPEED35_INITIAL_WAIT_FRESH5_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    SPEED35_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    _bundle_manifest,
)
from xycar_ai.steering_contract import steering_contract_mapping


class _BaseTuplePolicy(nn.Module):
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = image.shape[0]
        return torch.zeros(batch, 201), torch.zeros(batch, 201)


def _write_base_artifact(root: Path) -> Path:
    artifact = root / "base-fixture"
    artifact.mkdir(parents=True)
    model = torch.jit.trace(_BaseTuplePolicy(), torch.zeros(1, 3, 16, 16))
    model.save(str(artifact / "model.ts"))
    manifest = {
        "schema_version": 1,
        "artifact_id": artifact.name,
        "model": {
            "format": "torchscript",
            "file": "model.ts",
            "input": {"shape": [1, 3, 16, 16]},
        },
        "preprocessing": {
            "geometry": "full_frame_bicubic_resize",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "steering_contract": steering_contract_mapping(),
    }
    (artifact / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    lines = []
    for name in ("manifest.yaml", "model.ts"):
        digest = hashlib.sha256((artifact / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}\n")
    (artifact / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    return artifact


def _write_checkpoint(
    directory: Path,
    *,
    kind: str,
) -> Path:
    directory.mkdir()
    if kind == "signal":
        model = SignalTemporalPolicy(
            SignalModelConfig(
                backbone="mobilenetv3_small_050",
                pretrained=False,
                hidden_size=16,
                input_height=64,
                input_width=96,
            )
        )
        metrics = {"stop_false_negative_rate": 0.0, "false_left_rate": 0.0}
    else:
        model = ShortcutTemporalPolicy(
            ShortcutModelConfig(
                backbone="vit_tiny_patch16_224",
                pretrained=False,
                hidden_size=16,
                image_size=224,
                horizon_steps=3,
            )
        )
        metrics = {"early_handoff_rate": 0.0, "first_angle_mae": 5.0}
    checkpoint = directory / "best.pt"
    torch.save(
        {
            "schema_version": 1,
            "policy_kind": kind,
            "epoch": 1,
            "model_config": vars(model.config),
            "model_state": model.state_dict(),
            "data_provenance": {
                "fixture": True,
                **(
                    {"steering_contract": steering_contract_mapping()}
                    if kind == "shortcut"
                    else {}
                ),
            },
        },
        checkpoint,
    )
    (directory / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return checkpoint


def test_temporal_exports_build_one_verified_bundle(tmp_path: Path):
    model_root = tmp_path / "models"
    base = _write_base_artifact(model_root)
    signal = export_temporal_policy(
        kind="signal",
        checkpoint_path=_write_checkpoint(tmp_path / "signal-run", kind="signal"),
        artifact_id="signal-fixture",
        output_root=model_root,
    )
    shortcut = export_temporal_policy(
        kind="shortcut",
        checkpoint_path=_write_checkpoint(tmp_path / "shortcut-run", kind="shortcut"),
        artifact_id="shortcut-fixture",
        output_root=model_root,
    )

    bundle = build_bundle(
        base_artifact=base,
        signal_artifact=signal,
        shortcut_artifact=shortcut,
        artifact_id="competition-fixture",
        output_root=model_root,
    )

    manifest = verify_bundle(bundle)
    assert manifest["runtime"]["all_models_preloaded"] is True
    assert manifest["mission"]["action_priority"] == [
        "STOP",
        "LEFT",
        "STRAIGHT",
    ]
    assert manifest["steering_contract"] == steering_contract_mapping()

    base_manifest_path = base / "manifest.yaml"
    base_manifest = yaml.safe_load(base_manifest_path.read_text())
    del base_manifest["steering_contract"]
    base_manifest_path.write_text(
        yaml.safe_dump(base_manifest, sort_keys=False), encoding="utf-8"
    )
    lines = []
    for name in ("manifest.yaml", "model.ts"):
        digest = hashlib.sha256((base / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}\n")
    (base / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match="base artifact lacks normalized steering"):
        build_bundle(
            base_artifact=base,
            signal_artifact=signal,
            shortcut_artifact=shortcut,
            artifact_id="legacy-base-rejected",
            output_root=model_root,
        )


def test_human_bbox_traffic_bundle_manifests_preserve_models_and_version_votes(
    tmp_path: Path,
):
    base = tmp_path / "base"
    shortcut = tmp_path / "shortcut"
    base.mkdir()
    shortcut.mkdir()
    (base / "SHA256SUMS").write_text("base\n", encoding="utf-8")
    (shortcut / "SHA256SUMS").write_text("shortcut\n", encoding="utf-8")

    manifest = _bundle_manifest(
        artifact_id=HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
        schema_version=6,
        consecutive_signal_reads_by_action=(2, 2, 2),
        base_artifact=base,
        shortcut_artifact=shortcut,
        shortcut_artifact_id=EXPANDED_SHORTCUT_ID,
        traffic_classifier=tmp_path / "classifier.onnx",
    )

    assert manifest["components"]["traffic_light"]["sha256"] == (
        HUMAN_BBOX_TRAFFIC_SHA256
    )
    classifier = manifest["components"]["traffic_classifier"]
    assert classifier["sha256"] == HUMAN_BBOX_CLASSIFIER_SHA256
    assert classifier["input"]["shape"] == [1, 3, 128, 416]
    assert classifier["output"]["shape"] == [1, 3]
    assert classifier["classes"] == ["STOP", "STRAIGHT", "LEFT"]
    detector = manifest["detector"]
    assert detector["bbox_width_px"] == [40, 225]
    assert detector["max_detections"] == 1
    assert detector["classifier"]["minimum_probability"] == 0.5
    assert manifest["mission"]["red_stop_yolo_missing_release_frames"] == 30
    assert manifest["signal_vote"]["consecutive_reads"] == 2

    stabilized = _bundle_manifest(
        artifact_id=STABILIZED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
        schema_version=7,
        consecutive_signal_reads_by_action=(3, 15, 15),
        base_artifact=base,
        shortcut_artifact=shortcut,
        shortcut_artifact_id=EXPANDED_SHORTCUT_ID,
        traffic_classifier=tmp_path / "classifier.onnx",
    )

    assert stabilized["components"] == manifest["components"]
    assert stabilized["detector"] == manifest["detector"]
    assert stabilized["mission"] == manifest["mission"]
    assert stabilized["signal_vote"]["consecutive_reads_by_raw_class"] == {
        "STOP": 3,
        "STRAIGHT": 15,
        "LEFT": 15,
    }

    adaptive = _bundle_manifest(
        artifact_id=ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
        schema_version=8,
        consecutive_signal_reads_by_action=(3, 15, 15),
        base_artifact=base,
        shortcut_artifact=shortcut,
        shortcut_artifact_id=EXPANDED_SHORTCUT_ID,
        traffic_classifier=tmp_path / "classifier.onnx",
    )

    adaptive_detector = adaptive["detector"]
    assert adaptive_detector["classification_every_n_frames_after_detection"] == 1
    assert adaptive_detector["reuse_detected_bbox_between_yolo_frames"] is True
    assert adaptive["signal_vote"] == stabilized["signal_vote"]

    stop10_adaptive = _bundle_manifest(
        artifact_id=STOP10_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
        schema_version=9,
        consecutive_signal_reads_by_action=(10, 15, 15),
        base_artifact=base,
        shortcut_artifact=shortcut,
        shortcut_artifact_id=EXPANDED_SHORTCUT_ID,
        traffic_classifier=tmp_path / "classifier.onnx",
    )

    assert stop10_adaptive["detector"] == adaptive_detector
    assert stop10_adaptive["signal_vote"]["consecutive_reads_by_raw_class"] == {
        "STOP": 10,
        "STRAIGHT": 15,
        "LEFT": 15,
    }

    stop30_go30_adaptive = _bundle_manifest(
        artifact_id=STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
        schema_version=10,
        consecutive_signal_reads_by_action=(30, 30, 30),
        base_artifact=base,
        shortcut_artifact=shortcut,
        shortcut_artifact_id=EXPANDED_SHORTCUT_ID,
        traffic_classifier=tmp_path / "classifier.onnx",
    )

    assert stop30_go30_adaptive["detector"] == adaptive_detector
    assert stop30_go30_adaptive["signal_vote"]["consecutive_reads_by_raw_class"] == {
        "STOP": 30,
        "STRAIGHT": 30,
        "LEFT": 30,
    }

    speed35_stop30_go30_adaptive = _bundle_manifest(
        artifact_id=(SPEED35_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID),
        schema_version=11,
        consecutive_signal_reads_by_action=(30, 30, 30),
        base_artifact=base,
        shortcut_artifact=shortcut,
        shortcut_artifact_id=EXPANDED_SHORTCUT_ID,
        traffic_classifier=tmp_path / "classifier.onnx",
    )

    assert speed35_stop30_go30_adaptive["detector"] == adaptive_detector
    assert (
        speed35_stop30_go30_adaptive["signal_vote"]
        == (stop30_go30_adaptive["signal_vote"])
    )
    assert speed35_stop30_go30_adaptive["mission"]["base_speed_cap"] == 35.0
    assert (
        speed35_stop30_go30_adaptive["components"]["base"]["artifact_id"]
        == SPEED35_BASE_ID
    )

    speed35_stop10_go30_adaptive = _bundle_manifest(
        artifact_id=(SPEED35_STOP10_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID),
        schema_version=12,
        consecutive_signal_reads_by_action=(10, 30, 30),
        base_artifact=base,
        shortcut_artifact=shortcut,
        shortcut_artifact_id=EXPANDED_SHORTCUT_ID,
        traffic_classifier=tmp_path / "classifier.onnx",
    )

    assert speed35_stop10_go30_adaptive["detector"] == adaptive_detector
    assert speed35_stop10_go30_adaptive["signal_vote"][
        "consecutive_reads_by_raw_class"
    ] == {"STOP": 10, "STRAIGHT": 30, "LEFT": 30}
    assert speed35_stop10_go30_adaptive["mission"]["base_speed_cap"] == 35.0
    assert (
        speed35_stop10_go30_adaptive["components"]["base"]["artifact_id"]
        == SPEED35_BASE_ID
    )

    speed35_stop15_go15_adaptive = _bundle_manifest(
        artifact_id=(SPEED35_STOP15_GO15_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID),
        schema_version=13,
        consecutive_signal_reads_by_action=(15, 15, 15),
        base_artifact=base,
        shortcut_artifact=shortcut,
        shortcut_artifact_id=EXPANDED_SHORTCUT_ID,
        traffic_classifier=tmp_path / "classifier.onnx",
    )

    assert speed35_stop15_go15_adaptive["detector"] == adaptive_detector
    assert speed35_stop15_go15_adaptive["signal_vote"][
        "consecutive_reads_by_raw_class"
    ] == {"STOP": 15, "STRAIGHT": 15, "LEFT": 15}
    assert speed35_stop15_go15_adaptive["mission"]["base_speed_cap"] == 35.0
    assert (
        speed35_stop15_go15_adaptive["components"]["base"]["artifact_id"]
        == SPEED35_BASE_ID
    )

    initial_stop_once = _bundle_manifest(
        artifact_id=(SPEED35_INITIAL_STOP_ONCE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID),
        schema_version=14,
        consecutive_signal_reads_by_action=(15, 15, 15),
        base_artifact=base,
        shortcut_artifact=shortcut,
        shortcut_artifact_id=EXPANDED_SHORTCUT_ID,
        traffic_classifier=tmp_path / "classifier.onnx",
    )

    assert initial_stop_once["detector"] == adaptive_detector
    assert initial_stop_once["signal_vote"] == {
        "raw_classes": ["STOP", "STRAIGHT", "LEFT"],
        "consecutive_reads_by_raw_class": {
            "STOP": 15,
            "STRAIGHT": 15,
            "LEFT": 15,
        },
        "unknown_behavior": "reset_candidate",
        "different_raw_class_behavior": "restart_candidate_at_one",
        "stop_classes": ["STOP"],
        "stop_vote_behavior": "only_while_initial_stop_armed",
        "post_initial_stop_behavior": "ignore_stop",
        "navigation_actions": ["LEFT", "STRAIGHT"],
    }
    assert initial_stop_once["mission"]["initial_stop"]["clear_consecutive_reads"] == 3
    assert initial_stop_once["mission"]["red_cancels_shortcut"] is False
    assert "red_stop_yolo_missing_release_frames" not in (initial_stop_once["mission"])

    initial_wait_fresh5 = _bundle_manifest(
        artifact_id=(SPEED35_INITIAL_WAIT_FRESH5_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID),
        schema_version=15,
        consecutive_signal_reads_by_action=(5, 5, 5),
        base_artifact=base,
        shortcut_artifact=shortcut,
        shortcut_artifact_id=EXPANDED_SHORTCUT_ID,
        traffic_classifier=tmp_path / "classifier.onnx",
    )

    assert initial_wait_fresh5["detector"] == adaptive_detector
    assert initial_wait_fresh5["signal_vote"] == {
        "raw_classes": ["STOP", "STRAIGHT", "LEFT"],
        "consecutive_reads_by_raw_class": {
            "STOP": 5,
            "STRAIGHT": 5,
            "LEFT": 5,
        },
        "unknown_behavior": "reset_candidate",
        "different_raw_class_behavior": "restart_candidate_at_one",
        "stop_classes": ["STOP"],
        "stop_vote_behavior": "only_while_initial_stop_armed",
        "post_initial_stop_behavior": "ignore_stop",
        "navigation_actions": ["LEFT", "STRAIGHT"],
        "control_vote_source": "fresh_yolo_classifier_only",
        "cached_classifier_behavior": "diagnostics_only",
    }
    assert initial_wait_fresh5["mission"]["initial_stop"] == {
        "gamepad_activation": "lb_held_on_a_enable_wait_for_signal",
        "headless_activation": "wait_for_first_valid_signal",
        "stop_consecutive_reads": 5,
        "clear_classes": ["STRAIGHT", "LEFT"],
        "clear_consecutive_reads": 5,
        "clear_different_class_behavior": "restart_candidate_at_one",
        "unknown_or_missing_behavior": "reset_candidate_retain_stop",
        "post_clear_action_by_class": {
            "STRAIGHT": "BASE",
            "LEFT": "SHORTCUT",
        },
        "post_clear_stop_behavior": "ignore",
        "ready_behavior": "log_once_on_first_valid_fresh_class",
    }
    assert initial_wait_fresh5["mission"]["red_cancels_shortcut"] is False
    assert (
        "red_stop_yolo_missing_release_frames" not in (initial_wait_fresh5["mission"])
    )

    initial_wait_fresh3 = _bundle_manifest(
        artifact_id=(SPEED35_INITIAL_WAIT_FRESH3_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID),
        schema_version=16,
        consecutive_signal_reads_by_action=(5, 3, 3),
        base_artifact=base,
        shortcut_artifact=shortcut,
        shortcut_artifact_id=EXPANDED_SHORTCUT_ID,
        traffic_classifier=tmp_path / "classifier.onnx",
    )
    assert initial_wait_fresh3["detector"] == {
        **adaptive_detector,
        "classification_every_n_frames_after_detection": 3,
        "reuse_detected_bbox_between_yolo_frames": False,
    }
    assert initial_wait_fresh3["signal_vote"] == {
        **initial_wait_fresh5["signal_vote"],
        "consecutive_reads_by_raw_class": {
            "STOP": 5,
            "STRAIGHT": 3,
            "LEFT": 3,
        },
        "cached_classifier_behavior": "disabled",
    }
    assert initial_wait_fresh3["mission"]["initial_stop"] == {
        **initial_wait_fresh5["mission"]["initial_stop"],
        "clear_consecutive_reads": 3,
    }
    assert initial_wait_fresh3["mission"]["base_speed_cap"] == 35.0
    assert (
        "red_stop_yolo_missing_release_frames" not in (initial_wait_fresh3["mission"])
    )

    initial_wait_go1 = _bundle_manifest(
        artifact_id=(SPEED35_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID),
        schema_version=17,
        consecutive_signal_reads_by_action=(5, 1, 1),
        base_artifact=base,
        shortcut_artifact=shortcut,
        shortcut_artifact_id=EXPANDED_SHORTCUT_ID,
        traffic_classifier=tmp_path / "classifier.onnx",
    )
    assert initial_wait_go1["detector"] == initial_wait_fresh3["detector"]
    assert initial_wait_go1["signal_vote"] == {
        **initial_wait_fresh3["signal_vote"],
        "consecutive_reads_by_raw_class": {
            "STOP": 5,
            "STRAIGHT": 1,
            "LEFT": 1,
        },
    }
    assert initial_wait_go1["mission"]["initial_stop"] == {
        **initial_wait_fresh3["mission"]["initial_stop"],
        "clear_consecutive_reads": 1,
    }
    assert initial_wait_go1["mission"]["base_speed_cap"] == 35.0
    assert "red_stop_yolo_missing_release_frames" not in (initial_wait_go1["mission"])

    fix_initial_wait_go1 = _bundle_manifest(
        artifact_id=(SPEED35_FIX_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID),
        schema_version=18,
        consecutive_signal_reads_by_action=(5, 1, 1),
        base_artifact=base,
        shortcut_artifact=shortcut,
        shortcut_artifact_id=EXPANDED_SHORTCUT_ID,
        traffic_classifier=tmp_path / "classifier.onnx",
    )
    assert fix_initial_wait_go1["detector"] == initial_wait_go1["detector"]
    assert fix_initial_wait_go1["signal_vote"] == initial_wait_go1["signal_vote"]
    assert fix_initial_wait_go1["mission"] == initial_wait_go1["mission"]
    assert (
        fix_initial_wait_go1["components"]["base"]["artifact_id"] == SPEED35_FIX_BASE_ID
    )

    session_initial_wait_go1 = _bundle_manifest(
        artifact_id=(
            SPEED35_FIX_INITIAL_WAIT_GO1_SESSION_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
        ),
        schema_version=19,
        consecutive_signal_reads_by_action=(5, 1, 1),
        base_artifact=base,
        shortcut_artifact=shortcut,
        shortcut_artifact_id=EXPANDED_SHORTCUT_ID,
        traffic_classifier=tmp_path / "classifier.onnx",
    )
    assert session_initial_wait_go1["detector"] == (fix_initial_wait_go1["detector"])
    assert (
        session_initial_wait_go1["signal_vote"] == (fix_initial_wait_go1["signal_vote"])
    )
    assert session_initial_wait_go1["mission"] == {
        **fix_initial_wait_go1["mission"],
        "successful_shortcut_once_scope": "drive_gate_activation",
        "base_shadow": {
            **fix_initial_wait_go1["mission"]["base_shadow"],
            "stale_timeout_sec": 0.50,
        },
    }
    assert (
        session_initial_wait_go1["components"]["base"]["artifact_id"]
        == SPEED35_FIX_BASE_ID
    )
