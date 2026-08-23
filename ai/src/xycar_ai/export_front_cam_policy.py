from __future__ import annotations

import argparse
import hashlib
import math
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

from xycar_ai.compact_control import (
    ANGLE_OUTPUT_CLASSES,
    COMPACT_CONTROL_ENCODING,
    LEGACY_CONTROL_ENCODING,
    SPEED_OUTPUT_CLASSES,
    executed_command_to_history_tokens,
    unknown_history_pair,
)
from xycar_ai.front_cam_policy_model import (
    AR_CONTROL_TOKEN_ARCHITECTURE,
    CATEGORICAL_PREDICTION_MODE,
    CONTINUOUS_REGRESSION_PREDICTION_MODE,
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
COMPACT_AR_ARTIFACT_SCHEMA_VERSION = 5
REGRESSION_AR_ARTIFACT_SCHEMA_VERSION = 6
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


class _RegressionTupleOutputARPolicy(nn.Module):
    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        images: torch.Tensor,
        history_class_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.policy(images, history_class_ids)
        return outputs["angle_driver"], outputs["speed"]


class _FixedSpeedTupleOutputARPolicy(nn.Module):
    def __init__(self, policy: nn.Module, fixed_speed_class_id: int) -> None:
        super().__init__()
        self.policy = policy
        self.fixed_speed_class_id = fixed_speed_class_id

    def forward(
        self,
        images: torch.Tensor,
        history_class_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.policy(images, history_class_ids)
        fixed_speed_logits = torch.full_like(outputs["speed_logits"], -100.0)
        fixed_speed_logits[:, self.fixed_speed_class_id] = 100.0
        return outputs["angle_logits"], fixed_speed_logits


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
        choices=(
            LEGACY_ARTIFACT_SCHEMA_VERSION,
            AR_ARTIFACT_SCHEMA_VERSION,
            COMPACT_AR_ARTIFACT_SCHEMA_VERSION,
            REGRESSION_AR_ARTIFACT_SCHEMA_VERSION,
        ),
        help="Reject export unless the artifact has this schema version.",
    )
    parser.add_argument(
        "--promotion-report",
        default="",
        help=(
            "Optional promotion_gate.json. Its passed/failed status is "
            "validated and recorded in the artifact but does not block export."
        ),
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
    label_contract = _mapping(checkpoint, "label_contract", "checkpoint")
    data_config = _mapping(checkpoint_config, "data", "checkpoint.config")
    if data_config.get("required_steering_contract") != STEERING_CONTRACT_NAME:
        raise PolicyExportError(
            "checkpoint data must require normalized_percent_v2 steering"
        )
    promotion = _validated_promotion_report(
        checkpoint_path=checkpoint_path,
        data_config=data_config,
        report_path=promotion_report_path,
    )
    model_config = _mapping(checkpoint_config, "model", "checkpoint.config")
    model_name = _string(model_config, "name", "checkpoint.config.model")
    architecture = str(model_config.get("architecture", "task_tokens"))
    control_encoding = str(
        model_config.get("control_encoding", LEGACY_CONTROL_ENCODING)
    )
    prediction_mode = str(
        model_config.get("prediction_mode", CATEGORICAL_PREDICTION_MODE)
    )
    speed_output_max_value = model_config.get("speed_output_max", 30.0)
    if (
        isinstance(speed_output_max_value, bool)
        or not isinstance(speed_output_max_value, (int, float))
        or not math.isfinite(float(speed_output_max_value))
        or not float(speed_output_max_value).is_integer()
        or not 0.0 < float(speed_output_max_value) <= 50.0
    ):
        raise PolicyExportError(
            "checkpoint.config.model.speed_output_max must be a whole number in [1, 50]"
        )
    speed_output_max = float(speed_output_max_value)
    if prediction_mode not in {
        CATEGORICAL_PREDICTION_MODE,
        CONTINUOUS_REGRESSION_PREDICTION_MODE,
    }:
        raise PolicyExportError(f"unsupported prediction mode: {prediction_mode}")
    if prediction_mode == CONTINUOUS_REGRESSION_PREDICTION_MODE and (
        architecture != AR_CONTROL_TOKEN_ARCHITECTURE
        or control_encoding != COMPACT_CONTROL_ENCODING
    ):
        raise PolicyExportError("continuous regression requires compact AR control")
    if prediction_mode != CONTINUOUS_REGRESSION_PREDICTION_MODE and (
        speed_output_max != 30.0
    ):
        raise PolicyExportError("categorical artifacts require speed_output_max=30")
    schema_version = (
        REGRESSION_AR_ARTIFACT_SCHEMA_VERSION
        if architecture == AR_CONTROL_TOKEN_ARCHITECTURE
        and control_encoding == COMPACT_CONTROL_ENCODING
        and prediction_mode == CONTINUOUS_REGRESSION_PREDICTION_MODE
        else COMPACT_AR_ARTIFACT_SCHEMA_VERSION
        if architecture == AR_CONTROL_TOKEN_ARCHITECTURE
        and control_encoding == COMPACT_CONTROL_ENCODING
        else AR_ARTIFACT_SCHEMA_VERSION
        if architecture == AR_CONTROL_TOKEN_ARCHITECTURE
        else LEGACY_ARTIFACT_SCHEMA_VERSION
    )
    if require_schema_version is not None and schema_version != require_schema_version:
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

    training_objective, fixed_speed_class_id = _training_objective_contract(
        checkpoint=checkpoint,
        label_contract=label_contract,
        architecture=architecture,
        control_encoding=control_encoding,
        speed_output_max=speed_output_max,
    )
    if prediction_mode == CONTINUOUS_REGRESSION_PREDICTION_MODE:
        regression_speed = _mapping(
            label_contract,
            "speed",
            "checkpoint.label_contract",
        )
        regression_history = _mapping(
            label_contract,
            "history",
            "checkpoint.label_contract",
        )
        if (
            label_contract.get("prediction_mode")
            != CONTINUOUS_REGRESSION_PREDICTION_MODE
            or training_objective is None
            or training_objective.get("mode") != "joint_angle_speed_regression"
            or training_objective.get("speed_output_trained") is not True
            or regression_speed.get("range") != [0.0, speed_output_max]
            or regression_history.get("actual_speed_token_range")
            != [50, 50 + int(speed_output_max)]
        ):
            raise PolicyExportError(
                "regression checkpoint label and training objective contracts differ"
            )

    history_frames = int(model_config.get("history_frames", 0))
    use_type_embedding = bool(model_config.get("control_token_type_embedding", False))
    if architecture == AR_CONTROL_TOKEN_ARCHITECTURE:
        if control_encoding == LEGACY_CONTROL_ENCODING and (
            model_config.get("history_initial_angle", 0),
            model_config.get("history_initial_speed", 25),
        ) != (0, 25):
            raise PolicyExportError(
                "AR checkpoint initial history command must be (0, 25)"
            )
        policy_arguments: dict[str, object] = {
            "model_name": model_name,
            "pretrained": False,
            "image_size": image_size,
            "history_frames": history_frames,
            "use_control_type_embedding": use_type_embedding,
        }
        if control_encoding == COMPACT_CONTROL_ENCODING:
            policy_arguments["control_encoding"] = control_encoding
        if prediction_mode == CONTINUOUS_REGRESSION_PREDICTION_MODE:
            policy_arguments["prediction_mode"] = prediction_mode
            policy_arguments["speed_output_max"] = speed_output_max
        policy = AutoregressiveControlTokenViTPolicy(
            **policy_arguments,
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
        if prediction_mode == CONTINUOUS_REGRESSION_PREDICTION_MODE:
            if fixed_speed_class_id is not None:
                raise PolicyExportError("regression artifact cannot use fixed speed")
            wrapper = _RegressionTupleOutputARPolicy(policy).eval()
        else:
            wrapper = (
                _TupleOutputARPolicy(policy)
                if fixed_speed_class_id is None
                else _FixedSpeedTupleOutputARPolicy(policy, fixed_speed_class_id)
            ).eval()
        if control_encoding == COMPACT_CONTROL_ENCODING:
            initial_angle, initial_speed, _ = _compact_history_initialization(
                label_contract,
                speed_output_max=speed_output_max,
            )
            expected_shapes = (
                ((1, 1), (1, 1))
                if prediction_mode == CONTINUOUS_REGRESSION_PREDICTION_MODE
                else ((1, ANGLE_OUTPUT_CLASSES), (1, SPEED_OUTPUT_CLASSES))
            )
        else:
            initial_angle = int(model_config.get("history_initial_angle", 0)) + 100
            initial_speed = int(model_config.get("history_initial_speed", 25)) + 100
            expected_shapes = ((1, 201), (1, 201))
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
        expected_shapes = ((1, 201), (1, 201))
    with torch.inference_mode():
        eager_outputs = wrapper(*sample_inputs)
        traced = torch.jit.trace(wrapper, sample_inputs, strict=True)
        traced_outputs = traced(*sample_inputs)
    _validate_outputs(eager_outputs, expected_shapes=expected_shapes)
    _validate_outputs(traced_outputs, expected_shapes=expected_shapes)
    _validate_regression_outputs(
        eager_outputs,
        prediction_mode=prediction_mode,
        speed_output_max=speed_output_max,
    )
    _validate_regression_outputs(
        traced_outputs,
        prediction_mode=prediction_mode,
        speed_output_max=speed_output_max,
    )
    _validate_fixed_speed_output(eager_outputs, fixed_speed_class_id)
    _validate_fixed_speed_output(traced_outputs, fixed_speed_class_id)
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
        _validate_outputs(reloaded_outputs, expected_shapes=expected_shapes)
        _validate_regression_outputs(
            reloaded_outputs,
            prediction_mode=prediction_mode,
            speed_output_max=speed_output_max,
        )
        _validate_fixed_speed_output(reloaded_outputs, fixed_speed_class_id)
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
            training_objective=training_objective,
            fixed_speed_class_id=fixed_speed_class_id,
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
            f"artifact schema_version differs from required {require_schema_version}"
        )
    _validate_manifest_training_objective(manifest)


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
    training_objective: Mapping[str, object] | None,
    fixed_speed_class_id: int | None,
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
        control_encoding = str(
            model_config.get("control_encoding", LEGACY_CONTROL_ENCODING)
        )
        prediction_mode = str(
            model_config.get("prediction_mode", CATEGORICAL_PREDICTION_MODE)
        )
        speed_output_max = float(model_config.get("speed_output_max", 30.0))
        compact = control_encoding == COMPACT_CONTROL_ENCODING
        history_input_name = "history_token_ids" if compact else "history_class_ids"
        model_input: dict[str, object] = {
            "kind": "tuple",
            "order": ["images", history_input_name],
            "images": {
                "color_space": "RGB",
                "dtype": "float32",
                "shape": [1, 3, image_size, image_size],
            },
            history_input_name: {
                "dtype": "int64",
                "shape": [1, history_frames, 2],
            },
        }
        if compact:
            initial_angle, initial_speed, initialization = (
                _compact_history_initialization(
                    label_contract,
                    speed_output_max=(
                        speed_output_max
                        if prediction_mode == CONTINUOUS_REGRESSION_PREDICTION_MODE
                        else 30.0
                    ),
                )
            )
            history_speed_output_max = (
                int(speed_output_max)
                if prediction_mode == CONTINUOUS_REGRESSION_PREDICTION_MODE
                else 30
            )
            history_contract = {
                "frames": history_frames,
                "pair_order": ["angle_token_id", "speed_token_id"],
                "time_order": "oldest_to_newest",
                "initialization": initialization,
                "initial_token_ids": [initial_angle, initial_speed],
                "actual_angle_token_range": [0, 100],
                "actual_speed_token_range": [
                    50,
                    50 + history_speed_output_max,
                ],
                "update": "externally_executed_commands",
            }
            if initialization == "canonical_initial_command":
                label_history = _mapping(
                    label_contract,
                    "history",
                    "checkpoint.label_contract",
                )
                history_contract["initial_command"] = list(
                    label_history["initial_command"]
                )
            if prediction_mode == CONTINUOUS_REGRESSION_PREDICTION_MODE:
                output_shapes = [[1, 1], [1, 1]]
                output_order = ["angle_driver", "speed"]
                output_values: list[dict[str, object]] | None = [
                    {
                        "name": "angle_driver",
                        "dtype": "float32",
                        "unit": "driver_angle",
                        "range": [-50.0, 50.0],
                        "runtime_normalized_mapping": "value * 2",
                    },
                    {
                        "name": "speed",
                        "dtype": "float32",
                        "unit": "motor_speed",
                        "range": [0.0, speed_output_max],
                    },
                ]
                schema_version = REGRESSION_AR_ARTIFACT_SCHEMA_VERSION
            else:
                output_shapes = [
                    [1, ANGLE_OUTPUT_CLASSES],
                    [1, SPEED_OUTPUT_CLASSES],
                ]
                output_order = ["angle_logits", "speed_logits"]
                output_values = None
                schema_version = COMPACT_AR_ARTIFACT_SCHEMA_VERSION
        else:
            initial_angle = int(model_config.get("history_initial_angle", 0))
            initial_speed = int(model_config.get("history_initial_speed", 25))
            history_contract = {
                "frames": history_frames,
                "pair_order": ["angle_class_id", "speed_class_id"],
                "time_order": "oldest_to_newest",
                "initial_command": [initial_angle, initial_speed],
                "initial_class_ids": [initial_angle + 100, initial_speed + 100],
                "update": "externally_executed_commands",
            }
            output_shapes = [[1, 201], [1, 201]]
            output_order = ["angle_logits", "speed_logits"]
            output_values = None
            schema_version = AR_ARTIFACT_SCHEMA_VERSION
    else:
        model_input = {
            "name": "images",
            "color_space": "RGB",
            "dtype": "float32",
            "shape": [1, 3, image_size, image_size],
        }
        history_contract = None
        output_shapes = [[1, 201], [1, 201]]
        output_order = ["angle_logits", "speed_logits"]
        output_values = None
        schema_version = LEGACY_ARTIFACT_SCHEMA_VERSION
        prediction_mode = CATEGORICAL_PREDICTION_MODE
    model_output: dict[str, object] = {
        "kind": "tuple",
        "order": output_order,
        "shapes": output_shapes,
    }
    if output_values is not None:
        model_output["values"] = output_values
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
            "control_encoding": str(
                model_config.get("control_encoding", LEGACY_CONTROL_ENCODING)
            ),
            "prediction_mode": prediction_mode,
            "input": model_input,
            "output": model_output,
        },
        "preprocessing": dict(preprocessing),
        "label_contract": dict(label_contract),
        "steering_contract": steering_contract_mapping(),
        "runtime": {"torch_num_threads": 8, "warmup_count": 3},
    }
    if history_contract is not None:
        manifest["history"] = history_contract
    if training_objective is not None:
        manifest["training_objective"] = dict(training_objective)
    if fixed_speed_class_id is not None:
        manifest["speed_output"] = {
            "mode": "fixed_class",
            "command": fixed_speed_class_id,
            "class_id": fixed_speed_class_id,
            "checkpoint_head_trained": False,
        }
    if promotion is not None:
        manifest["promotion"] = dict(promotion)
    return manifest


def _training_objective_contract(
    *,
    checkpoint: Mapping[str, Any],
    label_contract: Mapping[str, Any],
    architecture: str,
    control_encoding: str,
    speed_output_max: float,
) -> tuple[dict[str, object] | None, int | None]:
    raw_objective = checkpoint.get("training_objective")
    if raw_objective is None:
        return None, None
    if not isinstance(raw_objective, Mapping):
        raise PolicyExportError("checkpoint.training_objective must be a mapping")
    mode = _string(raw_objective, "mode", "checkpoint.training_objective")
    speed_output_trained = raw_objective.get("speed_output_trained")
    if not isinstance(speed_output_trained, bool):
        raise PolicyExportError(
            "checkpoint.training_objective.speed_output_trained must be a boolean"
        )
    speed_loss_weight = _finite_number(
        raw_objective,
        "speed_loss_weight",
        "checkpoint.training_objective",
    )
    validation_speed_mae_weight = _finite_number(
        raw_objective,
        "validation_speed_mae_weight",
        "checkpoint.training_objective",
    )
    objective = {
        "mode": mode,
        "speed_output_trained": speed_output_trained,
        "speed_loss_weight": speed_loss_weight,
        "validation_speed_mae_weight": validation_speed_mae_weight,
    }
    if mode == "joint_angle_speed_regression":
        if raw_objective.get("loss") != "smooth_l1_normalized":
            raise PolicyExportError("regression objective must use normalized SmoothL1")
        angle_normalization = _finite_number(
            raw_objective,
            "angle_normalization",
            "checkpoint.training_objective",
        )
        speed_normalization = _finite_number(
            raw_objective,
            "speed_normalization",
            "checkpoint.training_objective",
        )
        angle_beta = _finite_number(
            raw_objective,
            "angle_beta",
            "checkpoint.training_objective",
        )
        speed_beta = _finite_number(
            raw_objective,
            "speed_beta",
            "checkpoint.training_objective",
        )
        if (
            angle_normalization != 50.0
            or speed_normalization != speed_output_max
            or angle_beta <= 0.0
            or not math.isclose(
                speed_beta,
                1.0 / speed_output_max,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or speed_loss_weight <= 0.0
            or not speed_output_trained
        ):
            raise PolicyExportError("regression objective contract is invalid")
        objective.update(
            {
                "loss": "smooth_l1_normalized",
                "angle_normalization": angle_normalization,
                "speed_normalization": speed_normalization,
                "angle_beta": angle_beta,
                "speed_beta": speed_beta,
            }
        )
    if speed_output_trained:
        return objective, None
    if (
        mode != "angle_only"
        or speed_loss_weight != 0.0
        or validation_speed_mae_weight != 0.0
    ):
        raise PolicyExportError(
            "untrained speed output requires angle_only mode and zero speed weights"
        )
    if (
        architecture != AR_CONTROL_TOKEN_ARCHITECTURE
        or control_encoding != COMPACT_CONTROL_ENCODING
    ):
        raise PolicyExportError(
            "angle-only export requires compact AR control encoding"
        )

    dataset_stats = _mapping(checkpoint, "dataset_stats", "checkpoint")
    all_stats = _mapping(dataset_stats, "all", "checkpoint.dataset_stats")
    sample_count = all_stats.get("sample_count")
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count <= 0
    ):
        raise PolicyExportError(
            "angle-only dataset sample_count must be a positive integer"
        )
    speed_range = all_stats.get("speed_range")
    if (
        not isinstance(speed_range, list)
        or len(speed_range) != 2
        or any(isinstance(value, bool) for value in speed_range)
        or not all(isinstance(value, (int, float)) for value in speed_range)
        or not all(math.isfinite(float(value)) for value in speed_range)
        or float(speed_range[0]) != float(speed_range[1])
    ):
        raise PolicyExportError(
            "angle-only dataset speed_range must contain one fixed speed"
        )
    fixed_speed = float(speed_range[0])
    fixed_speed_class_id = round(fixed_speed)
    if (
        fixed_speed != fixed_speed_class_id
        or not 0 <= fixed_speed_class_id < SPEED_OUTPUT_CLASSES
    ):
        raise PolicyExportError(
            "angle-only fixed speed must map exactly to a compact speed class"
        )

    history = _mapping(label_contract, "history", "checkpoint.label_contract")
    initial_command = history.get("initial_command")
    if (
        history.get("initialization") != "canonical_initial_command"
        or not isinstance(initial_command, list)
        or len(initial_command) != 2
        or isinstance(initial_command[1], bool)
        or not isinstance(initial_command[1], (int, float))
        or float(initial_command[1]) != fixed_speed
    ):
        raise PolicyExportError(
            "angle-only canonical history speed must match dataset fixed speed"
        )
    return objective, fixed_speed_class_id


def _validate_manifest_training_objective(
    manifest: Mapping[str, Any],
) -> None:
    objective = manifest.get("training_objective")
    if objective is None:
        return
    if not isinstance(objective, Mapping):
        raise PolicyExportError("artifact training_objective must be a mapping")
    speed_output_trained = objective.get("speed_output_trained")
    if not isinstance(speed_output_trained, bool):
        raise PolicyExportError(
            "artifact training_objective speed_output_trained must be a boolean"
        )
    if speed_output_trained:
        return
    speed_output = manifest.get("speed_output")
    if not isinstance(speed_output, Mapping):
        raise PolicyExportError(
            "angle-only artifact must declare its fixed speed output"
        )
    class_id = speed_output.get("class_id")
    if (
        speed_output.get("mode") != "fixed_class"
        or speed_output.get("checkpoint_head_trained") is not False
        or not isinstance(class_id, int)
        or isinstance(class_id, bool)
        or not 0 <= class_id < SPEED_OUTPUT_CLASSES
        or speed_output.get("command") != class_id
    ):
        raise PolicyExportError("angle-only artifact fixed speed contract is invalid")


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
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if report_path is None:
        return {
            "offline_gate": "not_evaluated",
            "generation": generation_value,
            "candidate_checkpoint_sha256": checkpoint_sha256,
        }
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
    status = report.get("status")
    if (
        report.get("schema_version") != 1
        or status not in {"passed", "failed"}
        or report.get("generation") != generation_value
        or not isinstance(candidate, Mapping)
        or not isinstance(checks, Mapping)
        or not checks
        or not all(isinstance(value, bool) for value in checks.values())
    ):
        raise PolicyExportError("promotion report contract is invalid")
    checks_passed = all(value is True for value in checks.values())
    if (status == "passed") != checks_passed:
        raise PolicyExportError("promotion report status disagrees with checks")
    if candidate.get("sha256") != checkpoint_sha256:
        raise PolicyExportError(
            "promotion report candidate hash differs from checkpoint"
        )
    parent = report.get("parent")
    if (
        not isinstance(parent, Mapping)
        or not isinstance(parent.get("sha256"), str)
        or not parent["sha256"]
    ):
        raise PolicyExportError("promotion report parent hash is missing")
    return {
        "offline_gate": status,
        "generation": generation_value,
        "report_sha256": sha256_file(report_path),
        "parent_checkpoint_sha256": parent["sha256"],
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "failed_checks": sorted(
            str(name) for name, value in checks.items() if value is False
        ),
    }


def _checkpoint_mapping(payload: object) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise PolicyExportError("checkpoint must contain a mapping")
    required = {"config", "model_state", "preprocessing", "label_contract"}
    missing = required - set(payload)
    if missing:
        raise PolicyExportError(f"checkpoint fields are missing: {sorted(missing)}")
    return payload


def _compact_history_initialization(
    label_contract: Mapping[str, Any],
    *,
    speed_output_max: float = 30.0,
) -> tuple[int, int, str]:
    history = _mapping(label_contract, "history", "checkpoint.label_contract")
    initialization = history.get("initialization")
    initial_ids = history.get("initial_token_ids")
    if initialization == "learned_unknown_tokens":
        expected = unknown_history_pair()
    elif initialization == "canonical_initial_command":
        initial_command = history.get("initial_command")
        if (
            not isinstance(initial_command, list)
            or len(initial_command) != 2
            or any(isinstance(value, bool) for value in initial_command)
            or not all(isinstance(value, (int, float)) for value in initial_command)
        ):
            raise PolicyExportError(
                "compact canonical history initial_command must be numeric "
                "[angle, speed]"
            )
        expected = executed_command_to_history_tokens(
            float(initial_command[0]),
            float(initial_command[1]),
            speed_max=speed_output_max,
        )
    else:
        raise PolicyExportError(
            "compact history initialization must be learned UNKNOWN or canonical"
        )
    if initial_ids != list(expected):
        raise PolicyExportError(
            f"compact history initial_token_ids must encode {list(expected)}"
        )
    return expected[0], expected[1], str(initialization)


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


def _finite_number(payload: Mapping[str, Any], key: str, context: str) -> float:
    value = payload.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise PolicyExportError(f"{context}.{key} must be a finite number")
    return float(value)


def _validate_outputs(
    outputs: object,
    *,
    expected_shapes: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise PolicyExportError("policy output must be a two-tensor tuple")
    for name, output, expected_shape in zip(
        ("angle_logits", "speed_logits"),
        outputs,
        expected_shapes,
        strict=True,
    ):
        if not isinstance(output, torch.Tensor):
            raise PolicyExportError(f"{name} must be a tensor")
        if tuple(output.shape) != expected_shape:
            raise PolicyExportError(
                f"{name} must have shape {list(expected_shape)}, "
                f"got {tuple(output.shape)}"
            )
        if not torch.isfinite(output).all():
            raise PolicyExportError(f"{name} contains a non-finite value")


def _validate_regression_outputs(
    outputs: tuple[torch.Tensor, torch.Tensor],
    *,
    prediction_mode: str,
    speed_output_max: float,
) -> None:
    if prediction_mode != CONTINUOUS_REGRESSION_PREDICTION_MODE:
        return
    angle_driver, speed = outputs
    if bool((angle_driver < -50.0).any()) or bool((angle_driver > 50.0).any()):
        raise PolicyExportError("regression angle output is outside [-50, 50]")
    if bool((speed < 0.0).any()) or bool((speed > speed_output_max).any()):
        raise PolicyExportError(
            f"regression speed output is outside [0, {speed_output_max:g}]"
        )


def _validate_fixed_speed_output(
    outputs: object,
    fixed_speed_class_id: int | None,
) -> None:
    if fixed_speed_class_id is None:
        return
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise PolicyExportError("policy output must be a two-tensor tuple")
    speed_logits = outputs[1]
    if not isinstance(speed_logits, torch.Tensor):
        raise PolicyExportError("speed_logits must be a tensor")
    predicted = torch.argmax(speed_logits, dim=1)
    if not torch.all(predicted == fixed_speed_class_id):
        raise PolicyExportError(
            "angle-only artifact did not produce its declared fixed speed"
        )


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
