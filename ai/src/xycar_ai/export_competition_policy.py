"""Export temporal mission policies and assemble one competition bundle."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import yaml

from xycar_ai.competition_data import SHORTCUT_PHASES
from xycar_ai.competition_models import (
    SIGNAL_STATUS_NAMES,
    ShortcutModelConfig,
    ShortcutStepWrapper,
    ShortcutTemporalPolicy,
    SignalModelConfig,
    SignalStepWrapper,
    SignalTemporalPolicy,
)
from xycar_ai.export_front_cam_policy import sha256_file, verify_artifact


ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CHECKSUM_FILENAME = "SHA256SUMS"
MANIFEST_FILENAME = "manifest.yaml"
MODEL_FILENAME = "model.ts"


class CompetitionExportError(ValueError):
    """Raised when a mission artifact is not safe to deploy."""


def signal_main(argv: Iterable[str] | None = None) -> None:
    _individual_main("signal", argv)


def shortcut_main(argv: Iterable[str] | None = None) -> None:
    _individual_main("shortcut", argv)


def bundle_main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a competition bundle")
    parser.add_argument("--base-artifact", required=True)
    parser.add_argument("--signal-artifact", required=True)
    parser.add_argument("--shortcut-artifact", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--output-root", default="artifacts/models")
    arguments = parser.parse_args(argv)
    result = build_bundle(
        base_artifact=Path(arguments.base_artifact),
        signal_artifact=Path(arguments.signal_artifact),
        shortcut_artifact=Path(arguments.shortcut_artifact),
        artifact_id=arguments.artifact_id,
        output_root=Path(arguments.output_root),
    )
    print(result)


def _individual_main(kind: str, argv: Iterable[str] | None) -> None:
    parser = argparse.ArgumentParser(description=f"Export {kind} policy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--output-root", default="artifacts/models")
    parser.add_argument(
        "--development-artifact",
        action="store_true",
        help="Allow export before race qualification; bundles reject it.",
    )
    arguments = parser.parse_args(argv)
    result = export_temporal_policy(
        kind=kind,
        checkpoint_path=Path(arguments.checkpoint),
        artifact_id=arguments.artifact_id,
        output_root=Path(arguments.output_root),
        development_artifact=arguments.development_artifact,
    )
    print(result)


def export_temporal_policy(
    *,
    kind: str,
    checkpoint_path: Path,
    artifact_id: str,
    output_root: Path,
    development_artifact: bool = False,
) -> Path:
    _validate_artifact_id(artifact_id)
    if kind not in {"signal", "shortcut"}:
        raise CompetitionExportError(f"unsupported policy kind: {kind}")
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise CompetitionExportError(f"checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise CompetitionExportError("checkpoint root must be a mapping")
    if checkpoint.get("schema_version") != 1 or checkpoint.get("policy_kind") != kind:
        raise CompetitionExportError("checkpoint kind or schema does not match")
    metrics = _load_test_metrics(checkpoint_path)
    qualified = _qualifies(kind, metrics)
    if not qualified and not development_artifact:
        raise CompetitionExportError(
            "test metrics do not satisfy the race export gate; use "
            "--development-artifact only for offline debugging"
        )
    model_config = _required_mapping(checkpoint, "model_config", "checkpoint")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping):
        raise CompetitionExportError("checkpoint.model_state must be a mapping")
    if kind == "signal":
        config = SignalModelConfig(
            backbone=_required_string(model_config, "backbone", "model_config"),
            pretrained=False,
            hidden_size=_required_int(model_config, "hidden_size", "model_config"),
            input_height=_required_int(
                model_config, "input_height", "model_config"
            ),
            input_width=_required_int(model_config, "input_width", "model_config"),
        )
        policy = SignalTemporalPolicy(config)
        policy.load_state_dict(model_state, strict=True)
        wrapper = SignalStepWrapper(policy.eval()).eval()
        inputs = (
            torch.zeros(1, 3, config.input_height, config.input_width),
            torch.zeros(1, 1, config.hidden_size),
        )
        output_contract = {
            "order": ["status_logits", "bbox", "progress", "next_hidden"],
            "shapes": [
                [1, len(SIGNAL_STATUS_NAMES)],
                [1, 4],
                [1],
                [1, 1, config.hidden_size],
            ],
        }
        preprocessing = {
            "color_space": "RGB",
            "geometry": "upper_two_thirds_bicubic_resize",
            "input_shape": [1, 3, config.input_height, config.input_width],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        labels: Mapping[str, Any] = {
            "status_order": list(SIGNAL_STATUS_NAMES),
            "lamp_encoding": "independent_multi_label_bits",
            "action_priority": ["red_or_yellow", "left_arrow", "straight_green"],
        }
    else:
        config = ShortcutModelConfig(
            backbone=_required_string(model_config, "backbone", "model_config"),
            pretrained=False,
            hidden_size=_required_int(model_config, "hidden_size", "model_config"),
            image_size=_required_int(model_config, "image_size", "model_config"),
            horizon_steps=_required_int(
                model_config, "horizon_steps", "model_config"
            ),
        )
        policy = ShortcutTemporalPolicy(config)
        policy.load_state_dict(model_state, strict=True)
        wrapper = ShortcutStepWrapper(policy.eval()).eval()
        inputs = (
            torch.zeros(1, 3, config.image_size, config.image_size),
            torch.zeros(1, 2),
            torch.zeros(1, 1, config.hidden_size),
        )
        output_contract = {
            "order": [
                "angle_logits",
                "speed_logits",
                "phase_logits",
                "handoff_logits",
                "next_hidden",
            ],
            "shapes": [
                [1, config.horizon_steps, 201],
                [1, config.horizon_steps, 201],
                [1, len(SHORTCUT_PHASES)],
                [1],
                [1, 1, config.hidden_size],
            ],
        }
        preprocessing = {
            "color_space": "RGB",
            "geometry": "full_frame_bicubic_resize_unwarped",
            "input_shape": [1, 3, config.image_size, config.image_size],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        labels = {
            "phase_order": list(SHORTCUT_PHASES),
            "command_classes": 201,
            "decode_mapping": "class_id - 100",
            "future_horizon_steps": config.horizon_steps,
        }
    with torch.inference_mode():
        eager = wrapper(*inputs)
        traced = torch.jit.trace(wrapper, inputs, strict=True)
        exported = traced(*inputs)
    _compare_outputs(eager, exported)
    output_root = output_root.expanduser().resolve()
    artifact_dir = output_root / artifact_id
    if artifact_dir.exists():
        raise CompetitionExportError(f"artifact already exists: {artifact_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".incoming-{artifact_id}-{os.getpid()}"
    if temporary.exists():
        raise CompetitionExportError(f"temporary artifact exists: {temporary}")
    temporary.mkdir()
    try:
        traced.save(str(temporary / MODEL_FILENAME))
        manifest = {
            "schema_version": 1,
            "artifact_kind": f"{kind}_temporal_policy",
            "artifact_id": artifact_id,
            "created_at": datetime.now(UTC).isoformat(),
            "race_qualified": qualified,
            "source": {
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
                "data_provenance": checkpoint.get("data_provenance", {}),
                "test_metrics": metrics,
            },
            "model": {
                "format": "torchscript",
                "file": MODEL_FILENAME,
                "config": dict(model_config),
                "output": output_contract,
            },
            "preprocessing": preprocessing,
            "labels": dict(labels),
        }
        (temporary / MANIFEST_FILENAME).write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )
        _write_checksums(temporary)
        verify_temporal_artifact(
            temporary,
            expected_kind=kind,
            expected_artifact_id=artifact_id,
        )
        temporary.rename(artifact_dir)
    except BaseException:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise
    return artifact_dir


def build_bundle(
    *,
    base_artifact: Path,
    signal_artifact: Path,
    shortcut_artifact: Path,
    artifact_id: str,
    output_root: Path,
) -> Path:
    _validate_artifact_id(artifact_id)
    base_artifact = base_artifact.expanduser().resolve()
    signal_artifact = signal_artifact.expanduser().resolve()
    shortcut_artifact = shortcut_artifact.expanduser().resolve()
    verify_artifact(base_artifact, require_schema_version=1)
    signal_manifest = verify_temporal_artifact(
        signal_artifact,
        expected_kind="signal",
        require_race_qualified=True,
    )
    shortcut_manifest = verify_temporal_artifact(
        shortcut_artifact,
        expected_kind="shortcut",
        require_race_qualified=True,
    )
    base_manifest = _load_yaml_mapping(base_artifact / MANIFEST_FILENAME)
    output_root = output_root.expanduser().resolve()
    final = output_root / artifact_id
    if final.exists():
        raise CompetitionExportError(f"bundle already exists: {final}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".incoming-{artifact_id}-{os.getpid()}"
    if temporary.exists():
        raise CompetitionExportError(f"temporary bundle exists: {temporary}")
    temporary.mkdir()
    try:
        provenance = temporary / "provenance"
        provenance.mkdir()
        shutil.copy2(base_artifact / "model.ts", temporary / "base_model.ts")
        shutil.copy2(signal_artifact / "model.ts", temporary / "signal_model.ts")
        shutil.copy2(
            shortcut_artifact / "model.ts",
            temporary / "shortcut_model.ts",
        )
        for name, source in (
            ("base_manifest.yaml", base_artifact / MANIFEST_FILENAME),
            ("signal_manifest.yaml", signal_artifact / MANIFEST_FILENAME),
            ("shortcut_manifest.yaml", shortcut_artifact / MANIFEST_FILENAME),
        ):
            shutil.copy2(source, provenance / name)
        manifest = {
            "schema_version": 1,
            "artifact_kind": "competition_bundle",
            "artifact_id": artifact_id,
            "created_at": datetime.now(UTC).isoformat(),
            "models": {
                "base": {
                    "file": "base_model.ts",
                    "artifact_id": base_manifest["artifact_id"],
                    "source_sha256s_sha256": sha256_file(
                        base_artifact / CHECKSUM_FILENAME
                    ),
                    "model": base_manifest["model"],
                    "preprocessing": base_manifest["preprocessing"],
                },
                "signal": {
                    "file": "signal_model.ts",
                    "artifact_id": signal_manifest["artifact_id"],
                    "config": signal_manifest["model"]["config"],
                    "preprocessing": signal_manifest["preprocessing"],
                    "labels": signal_manifest["labels"],
                },
                "shortcut": {
                    "file": "shortcut_model.ts",
                    "artifact_id": shortcut_manifest["artifact_id"],
                    "config": shortcut_manifest["model"]["config"],
                    "preprocessing": shortcut_manifest["preprocessing"],
                    "labels": shortcut_manifest["labels"],
                },
            },
            "mission": {
                "signal_probability_threshold": 0.5,
                "stop_votes": {"required": 2, "window": 3},
                "go_votes": {"required": 4, "window": 5},
                "decision_progress_deadline": 0.9,
                "handoff_probability_threshold": 0.9,
                "handoff_consecutive_frames": 5,
                "handoff_max_angle_difference": 25.0,
                "shortcut_timeout_sec": 12.0,
                "shortcut_once": True,
                "action_priority": ["STOP", "LEFT", "STRAIGHT"],
            },
            "runtime": {
                "control_rate_hz": 20.0,
                "maximum_forward_speed": 15.0,
                "all_models_preloaded": True,
            },
        }
        (temporary / MANIFEST_FILENAME).write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )
        _write_checksums(temporary)
        verify_bundle(temporary, expected_artifact_id=artifact_id)
        temporary.rename(final)
    except BaseException:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise
    return final


def verify_temporal_artifact(
    artifact_dir: Path,
    *,
    expected_kind: str,
    require_race_qualified: bool = False,
    expected_artifact_id: str | None = None,
) -> Mapping[str, Any]:
    root = _safe_artifact_root(artifact_dir)
    _verify_checksums(root)
    manifest = _load_yaml_mapping(root / MANIFEST_FILENAME)
    if manifest.get("schema_version") != 1:
        raise CompetitionExportError("temporal artifact schema must be 1")
    if manifest.get("artifact_kind") != f"{expected_kind}_temporal_policy":
        raise CompetitionExportError("temporal artifact kind mismatch")
    required_id = expected_artifact_id or root.name
    if manifest.get("artifact_id") != required_id:
        raise CompetitionExportError("temporal artifact id mismatch")
    if require_race_qualified and manifest.get("race_qualified") is not True:
        raise CompetitionExportError("temporal artifact is not race-qualified")
    model = _required_mapping(manifest, "model", "manifest")
    if model.get("format") != "torchscript" or model.get("file") != MODEL_FILENAME:
        raise CompetitionExportError("unsupported temporal model contract")
    return manifest


def verify_bundle(
    artifact_dir: Path,
    *,
    expected_artifact_id: str | None = None,
) -> Mapping[str, Any]:
    root = _safe_artifact_root(artifact_dir)
    _verify_checksums(root)
    manifest = _load_yaml_mapping(root / MANIFEST_FILENAME)
    if manifest.get("schema_version") != 1:
        raise CompetitionExportError("bundle schema_version must be 1")
    if manifest.get("artifact_kind") != "competition_bundle":
        raise CompetitionExportError("artifact is not a competition bundle")
    required_id = expected_artifact_id or root.name
    if manifest.get("artifact_id") != required_id:
        raise CompetitionExportError("bundle artifact id mismatch")
    models = _required_mapping(manifest, "models", "manifest")
    for name, filename in (
        ("base", "base_model.ts"),
        ("signal", "signal_model.ts"),
        ("shortcut", "shortcut_model.ts"),
    ):
        contract = _required_mapping(models, name, "models")
        if contract.get("file") != filename:
            raise CompetitionExportError(f"bundle {name} model filename mismatch")
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise CompetitionExportError(f"bundle model is missing: {filename}")
    mission = _required_mapping(manifest, "mission", "manifest")
    if mission.get("action_priority") != ["STOP", "LEFT", "STRAIGHT"]:
        raise CompetitionExportError("bundle action priority is unsupported")
    return manifest


def _qualifies(kind: str, metrics: Mapping[str, Any]) -> bool:
    try:
        if kind == "signal":
            return (
                float(metrics["stop_false_negative_rate"]) == 0.0
                and float(metrics["false_left_rate"]) == 0.0
            )
        return (
            float(metrics["early_handoff_rate"]) == 0.0
            and float(metrics["first_angle_mae"]) <= 10.0
        )
    except (KeyError, TypeError, ValueError):
        return False


def _load_test_metrics(checkpoint_path: Path) -> Mapping[str, Any]:
    path = checkpoint_path.parent / "test_metrics.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CompetitionExportError("test_metrics.json is invalid") from exc
    if not isinstance(payload, Mapping):
        raise CompetitionExportError("test metrics root must be a mapping")
    return payload


def _compare_outputs(eager: object, exported: object) -> None:
    if not isinstance(eager, tuple) or not isinstance(exported, tuple):
        raise CompetitionExportError("export wrapper must return tuples")
    if len(eager) != len(exported):
        raise CompetitionExportError("TorchScript output count changed")
    for expected, actual in zip(eager, exported, strict=True):
        if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
            raise CompetitionExportError("model output contains non-finite values")
        if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-5):
            raise CompetitionExportError("TorchScript output differs from eager")


def _write_checksums(root: Path) -> None:
    files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != CHECKSUM_FILENAME
    )
    (root / CHECKSUM_FILENAME).write_text(
        "".join(f"{sha256_file(root / path)}  {path}\n" for path in files),
        encoding="utf-8",
    )


def _verify_checksums(root: Path) -> None:
    checksum_path = root / CHECKSUM_FILENAME
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise CompetitionExportError("SHA256SUMS is missing or unsafe")
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise CompetitionExportError("invalid SHA256SUMS line")
        digest, relative = parts
        relative = relative.removeprefix("*")
        _validate_relative_path(relative)
        expected[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != CHECKSUM_FILENAME
    }
    if set(expected) != actual:
        raise CompetitionExportError("SHA256SUMS file list mismatch")
    for relative, digest in expected.items():
        path = root / relative
        if path.is_symlink() or sha256_file(path) != digest:
            raise CompetitionExportError(f"checksum mismatch: {relative}")


def _safe_artifact_root(path: Path) -> Path:
    path = path.expanduser()
    if path.is_symlink():
        raise CompetitionExportError("artifact root must not be a symlink")
    path = path.resolve()
    if not path.is_dir():
        raise CompetitionExportError(f"artifact root is missing: {path}")
    return path


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CompetitionExportError(f"missing or unsafe YAML: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CompetitionExportError(f"invalid YAML: {path}") from exc
    if not isinstance(payload, Mapping):
        raise CompetitionExportError(f"YAML root must be a mapping: {path}")
    return payload


def _required_mapping(
    payload: Mapping[str, Any],
    key: str,
    context: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise CompetitionExportError(f"{context}.{key} must be a mapping")
    return value


def _required_string(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise CompetitionExportError(f"{context}.{key} must be a string")
    return value


def _required_int(payload: Mapping[str, Any], key: str, context: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CompetitionExportError(f"{context}.{key} must be positive integer")
    return value


def _validate_artifact_id(value: str) -> None:
    if not ARTIFACT_ID_PATTERN.fullmatch(value):
        raise CompetitionExportError(f"invalid artifact id: {value}")


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or not re.fullmatch(r"[A-Za-z0-9._/-]+", value)
    ):
        raise CompetitionExportError(f"unsafe artifact path: {value}")
