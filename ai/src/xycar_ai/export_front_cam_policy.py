from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import yaml
from torch import nn

from xycar_ai.front_cam_policy_model import (
    AR_CONTROL_TOKEN_ARCHITECTURE,
    AutoregressiveControlTokenViTPolicy,
    TaskTokenViTPolicy,
)
from xycar_ai.steering_contract import (
    STEERING_CONTRACT_NAME,
    is_exact_steering_contract,
    steering_contract_mapping,
)

LEGACY_ARTIFACT_SCHEMA_VERSION = 1
AR_ARTIFACT_SCHEMA_VERSION = 3
ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_OUTPUT_ROOT = Path("artifacts/models")
MODEL_FILENAME = "model.ts"
MANIFEST_FILENAME = "manifest.yaml"
CHECKSUM_FILENAME = "SHA256SUMS"


class PolicyExportError(ValueError):
    """Raised when a checkpoint cannot produce a safe runtime artifact."""


class _TupleOutputPolicy(nn.Module):
    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.policy(images)
        return outputs["angle_logits"], outputs["speed_logits"]


class _TupleOutputARPolicy(nn.Module):
    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        images: torch.Tensor,
        history_class_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.policy(images, history_class_ids)
        return outputs["angle_logits"], outputs["speed_logits"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a front-camera policy checkpoint as TorchScript."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Artifact parent directory (default: artifacts/models).",
    )
    parser.add_argument(
        "--require-schema-version",
        type=int,
        choices=(LEGACY_ARTIFACT_SCHEMA_VERSION, AR_ARTIFACT_SCHEMA_VERSION),
        help="Reject export unless the artifact has this schema version.",
    )
    parser.add_argument(
        "--promotion-report",
        default="",
        help="Required passing promotion_gate.json for Guided generation 1+.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    artifact_dir = export_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        artifact_id=args.artifact_id,
        output_root=Path(args.output_root),
        require_schema_version=args.require_schema_version,
        promotion_report_path=(
            Path(args.promotion_report) if args.promotion_report else None
        ),
    )
    print(f"exported policy artifact: {artifact_dir}")
    return 0


def export_checkpoint(
    *,
    checkpoint_path: Path,
    artifact_id: str,
    output_root: Path,
    require_schema_version: int | None = None,
    promotion_report_path: Path | None = None,
) -> Path:
    _validate_artifact_id(artifact_id)
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    output_root = output_root.expanduser().resolve()
    artifact_dir = output_root / artifact_id
    if artifact_dir.exists():
        raise FileExistsError(f"artifact already exists: {artifact_dir}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint = _checkpoint_mapping(payload)
    checkpoint_config = _mapping(checkpoint, "config", "checkpoint")
    data_config = _mapping(checkpoint_config, "data", "checkpoint.config")
    if data_config.get("required_steering_contract") != STEERING_CONTRACT_NAME:
        raise PolicyExportError(
            "checkpoint data must require normalized_percent_v1 steering"
        )
    promotion = _validated_promotion_report(
        checkpoint_path=checkpoint_path,
        data_config=data_config,
        report_path=promotion_report_path,
    )
    model_config = _mapping(
        checkpoint_config, "model", "checkpoint.config"
    )
    model_name = _string(model_config, "name", "checkpoint.config.model")
    architecture = str(model_config.get("architecture", "task_tokens"))
    schema_version = (
        AR_ARTIFACT_SCHEMA_VERSION
        if architecture == AR_CONTROL_TOKEN_ARCHITECTURE
        else LEGACY_ARTIFACT_SCHEMA_VERSION
    )
    if (
        require_schema_version is not None
        and schema_version != require_schema_version
    ):
        raise PolicyExportError(
            "checkpoint would export artifact schema_version "
            f"{schema_version}, required {require_schema_version}"
        )
    image_size = _integer(
        model_config,
        "image_size",
        "checkpoint.config.model",
    )
    if image_size <= 0:
        raise PolicyExportError("checkpoint image_size must be positive")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping):
        raise PolicyExportError("checkpoint.model_state must be a mapping")

    history_frames = int(model_config.get("history_frames", 0))
    use_type_embedding = bool(model_config.get("control_token_type_embedding", False))
    if architecture == AR_CONTROL_TOKEN_ARCHITECTURE:
        if (
            model_config.get("history_initial_angle", 0),
            model_config.get("history_initial_speed", 25),
        ) != (0, 25):
            raise PolicyExportError(
                "AR checkpoint initial history command must be (0, 25)"
            )
        policy = AutoregressiveControlTokenViTPolicy(
            model_name=model_name,
            pretrained=False,
            image_size=image_size,
            history_frames=history_frames,
            use_control_type_embedding=use_type_embedding,
        )
    elif architecture == "task_tokens":
        policy = TaskTokenViTPolicy(
            model_name=model_name,
            pretrained=False,
            image_size=image_size,
        )
    else:
        raise PolicyExportError(
            f"unsupported checkpoint model architecture: {architecture}"
        )
    policy.load_state_dict(model_state, strict=True)
    policy.eval()
    image_sample = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    if architecture == AR_CONTROL_TOKEN_ARCHITECTURE:
        wrapper = _TupleOutputARPolicy(policy).eval()
        initial_angle = int(model_config.get("history_initial_angle", 0)) + 100
        initial_speed = int(model_config.get("history_initial_speed", 25)) + 100
        history_sample = torch.tensor(
            [[[initial_angle, initial_speed]] * history_frames],
            dtype=torch.long,
        )
        sample_inputs: tuple[torch.Tensor, ...] = (
            image_sample,
            history_sample,
        )
    else:
        wrapper = _TupleOutputPolicy(policy).eval()
        sample_inputs = (image_sample,)
    with torch.inference_mode():
        eager_outputs = wrapper(*sample_inputs)
        traced = torch.jit.trace(wrapper, sample_inputs, strict=True)
        traced_outputs = traced(*sample_inputs)
    _validate_outputs(eager_outputs)
    _validate_outputs(traced_outputs)
    for eager, exported in zip(eager_outputs, traced_outputs, strict=True):
        if not torch.allclose(eager, exported, atol=1e-5, rtol=1e-5):
            raise PolicyExportError("TorchScript output differs from eager output")

    output_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_root / f".incoming-{artifact_id}-{os.getpid()}"
    if temporary_dir.exists():
        raise FileExistsError(f"temporary artifact path exists: {temporary_dir}")
    temporary_dir.mkdir()
    try:
        model_path = temporary_dir / MODEL_FILENAME
        traced.save(str(model_path))
        reloaded = torch.jit.load(str(model_path), map_location="cpu").eval()
        with torch.inference_mode():
            reloaded_outputs = reloaded(*sample_inputs)
        _validate_outputs(reloaded_outputs)
        for eager, exported in zip(eager_outputs, reloaded_outputs, strict=True):
            if not torch.allclose(eager, exported, atol=1e-5, rtol=1e-5):
                raise PolicyExportError(
                    "reloaded TorchScript output differs from eager output"
                )

        manifest = _build_manifest(
            artifact_id=artifact_id,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            model_name=model_name,
            image_size=image_size,
            architecture=architecture,
            model_config=model_config,
            promotion=promotion,
        )
        manifest_path = temporary_dir / MANIFEST_FILENAME
        manifest_path.write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )
        checksums = [MODEL_FILENAME, MANIFEST_FILENAME]
        checksum_path = temporary_dir / CHECKSUM_FILENAME
        checksum_path.write_text(
            "".join(
                f"{sha256_file(temporary_dir / relative)}  {relative}\n"
                for relative in checksums
            ),
            encoding="utf-8",
        )
        verify_artifact(
            temporary_dir,
            require_schema_version=require_schema_version,
        )
        temporary_dir.rename(artifact_dir)
    except BaseException:
        if temporary_dir.is_dir():
            shutil.rmtree(temporary_dir)
        raise
    return artifact_dir


def verify_artifact(
    artifact_dir: Path, *, require_schema_version: int | None = None
) -> None:
    artifact_dir = artifact_dir.expanduser()
    if artifact_dir.is_symlink():
        raise PolicyExportError("artifact directory must not be a symlink")
    artifact_dir = artifact_dir.resolve()
    manifest_path = artifact_dir / MANIFEST_FILENAME
    checksum_path = artifact_dir / CHECKSUM_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise PolicyExportError("artifact manifest is missing or unsafe")
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise PolicyExportError("artifact checksum file is missing or unsafe")
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise PolicyExportError("invalid SHA256SUMS line")
        digest, relative = parts
        relative = relative.removeprefix("*")
        _validate_relative_artifact_path(relative)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise PolicyExportError("invalid SHA-256 digest")
        if relative in expected:
            raise PolicyExportError(f"duplicate checksum path: {relative}")
        expected[relative] = digest

    actual_files = {
        path.relative_to(artifact_dir).as_posix()
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.name != CHECKSUM_FILENAME
    }
    if actual_files != set(expected):
        raise PolicyExportError("SHA256SUMS file list does not match artifact contents")
    for relative, digest in expected.items():
        path = artifact_dir / relative
        if path.is_symlink() or not path.is_file():
            raise PolicyExportError(f"artifact file is missing or unsafe: {relative}")
        if sha256_file(path) != digest:
            raise PolicyExportError(f"artifact checksum mismatch: {relative}")
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PolicyExportError("artifact manifest is invalid YAML") from exc
    if not isinstance(manifest, Mapping):
        raise PolicyExportError("artifact manifest must be a mapping")
    steering_contract = manifest.get("steering_contract")
    if steering_contract is not None and not is_exact_steering_contract(
        steering_contract
    ):
        raise PolicyExportError("artifact steering contract is incompatible")
    if (
        require_schema_version is not None
        and manifest.get("schema_version") != require_schema_version
    ):
        raise PolicyExportError(
            "artifact schema_version differs from required "
            f"{require_schema_version}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_manifest(
    *,
    artifact_id: str,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    model_name: str,
    image_size: int,
    architecture: str,
    model_config: Mapping[str, Any],
    promotion: Mapping[str, object] | None,
) -> dict[str, object]:
    preprocessing = _mapping(checkpoint, "preprocessing", "checkpoint")
    label_contract = _mapping(checkpoint, "label_contract", "checkpoint")
    source = checkpoint.get("source")
    if not isinstance(source, Mapping):
        source = {"mgw_commit": "unknown", "dirty": True}
    dataset_stats = checkpoint.get("dataset_stats")
    dataset_snapshot = "unknown"
    if isinstance(dataset_stats, Mapping):
        snapshot = dataset_stats.get("dataset_snapshot")
        if isinstance(snapshot, str) and snapshot:
            dataset_snapshot = snapshot
    if architecture == AR_CONTROL_TOKEN_ARCHITECTURE:
        history_frames = int(model_config.get("history_frames", 0))
        initial_angle = int(model_config.get("history_initial_angle", 0))
        initial_speed = int(model_config.get("history_initial_speed", 25))
        model_input: dict[str, object] = {
            "kind": "tuple",
            "order": ["images", "history_class_ids"],
            "images": {
                "color_space": "RGB",
                "dtype": "float32",
                "shape": [1, 3, image_size, image_size],
            },
            "history_class_ids": {
                "dtype": "int64",
                "shape": [1, history_frames, 2],
            },
        }
        history_contract: dict[str, object] | None = {
            "frames": history_frames,
            "pair_order": ["angle_class_id", "speed_class_id"],
            "time_order": "oldest_to_newest",
            "initial_command": [initial_angle, initial_speed],
            "initial_class_ids": [initial_angle + 100, initial_speed + 100],
            "update": "externally_executed_commands",
        }
        schema_version = AR_ARTIFACT_SCHEMA_VERSION
    else:
        model_input = {
            "name": "images",
            "color_space": "RGB",
            "dtype": "float32",
            "shape": [1, 3, image_size, image_size],
        }
        history_contract = None
        schema_version = LEGACY_ARTIFACT_SCHEMA_VERSION
    manifest: dict[str, object] = {
        "schema_version": schema_version,
        "artifact_id": artifact_id,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            **dict(source),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
            "best_epoch": int(checkpoint.get("best_epoch", 0)),
            "best_score": float(checkpoint.get("best_score", float("inf"))),
        },
        "dataset": {"snapshot": dataset_snapshot},
        "model": {
            "format": "torchscript",
            "file": MODEL_FILENAME,
            "name": model_name,
            "architecture": architecture,
            "input": model_input,
            "output": {
                "kind": "tuple",
                "order": ["angle_logits", "speed_logits"],
                "shapes": [[1, 201], [1, 201]],
            },
        },
        "preprocessing": dict(preprocessing),
        "label_contract": dict(label_contract),
        "steering_contract": steering_contract_mapping(),
        "runtime": {"torch_num_threads": 8, "warmup_count": 3},
    }
    if history_contract is not None:
        manifest["history"] = history_contract
    if promotion is not None:
        manifest["promotion"] = dict(promotion)
    return manifest


def _validated_promotion_report(
    *,
    checkpoint_path: Path,
    data_config: Mapping[str, Any],
    report_path: Path | None,
) -> dict[str, object] | None:
    generation_value = data_config.get("current_generation", 0)
    if isinstance(generation_value, bool) or not isinstance(generation_value, int):
        raise PolicyExportError("checkpoint current_generation must be an integer")
    if generation_value <= 0:
        return None
    if report_path is None:
        raise PolicyExportError(
            "Guided generation 1+ export requires --promotion-report"
        )
    report_path = report_path.expanduser().resolve()
    if not report_path.is_file() or report_path.is_symlink():
        raise PolicyExportError("promotion report is missing or unsafe")
    try:
        report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PolicyExportError("promotion report is invalid") from exc
    if not isinstance(report, Mapping):
        raise PolicyExportError("promotion report must be a mapping")
    candidate = report.get("candidate")
    checks = report.get("checks")
    if (
        report.get("schema_version") != 1
        or report.get("status") != "passed"
        or report.get("generation") != generation_value
        or not isinstance(candidate, Mapping)
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise PolicyExportError("promotion report has not passed every gate")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if candidate.get("sha256") != checkpoint_sha256:
        raise PolicyExportError("promotion report candidate hash differs from checkpoint")
    parent = report.get("parent")
    if not isinstance(parent, Mapping) or not isinstance(parent.get("sha256"), str):
        raise PolicyExportError("promotion report parent hash is missing")
    return {
        "offline_gate": "passed",
        "generation": generation_value,
        "report_sha256": sha256_file(report_path),
        "parent_checkpoint_sha256": parent["sha256"],
        "candidate_checkpoint_sha256": checkpoint_sha256,
    }


def _checkpoint_mapping(payload: object) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise PolicyExportError("checkpoint must contain a mapping")
    required = {"config", "model_state", "preprocessing", "label_contract"}
    missing = required - set(payload)
    if missing:
        raise PolicyExportError(f"checkpoint fields are missing: {sorted(missing)}")
    return payload


def _mapping(
    payload: Mapping[str, Any],
    key: str,
    context: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise PolicyExportError(f"{context}.{key} must be a mapping")
    return value


def _string(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise PolicyExportError(f"{context}.{key} must be a non-empty string")
    return value


def _integer(payload: Mapping[str, Any], key: str, context: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise PolicyExportError(f"{context}.{key} must be an integer")
    return value


def _validate_outputs(outputs: object) -> None:
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise PolicyExportError("policy output must be a two-tensor tuple")
    for name, output in zip(
        ("angle_logits", "speed_logits"),
        outputs,
        strict=True,
    ):
        if not isinstance(output, torch.Tensor):
            raise PolicyExportError(f"{name} must be a tensor")
        if tuple(output.shape) != (1, 201):
            raise PolicyExportError(
                f"{name} must have shape [1, 201], got {tuple(output.shape)}"
            )
        if not torch.isfinite(output).all():
            raise PolicyExportError(f"{name} contains a non-finite value")


def _validate_artifact_id(artifact_id: str) -> None:
    if not ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
        raise PolicyExportError(f"invalid artifact id: {artifact_id}")


def _validate_relative_artifact_path(relative: str) -> None:
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or not re.fullmatch(r"[A-Za-z0-9._/-]+", relative)
    ):
        raise PolicyExportError(f"unsafe artifact path: {relative}")


if __name__ == "__main__":
    raise SystemExit(main())
