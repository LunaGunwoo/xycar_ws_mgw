"""Export the nice-shortcut ResNet18 angle checkpoint for vehicle runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import torch
import torchvision
import yaml
from torch import nn

from xycar_ai.steering_contract import steering_contract_mapping


SCHEMA_VERSION = 7
IMAGE_SIZE = 224
SPEED_DIVISOR = 25.0
MODEL_FILENAME = "model.ts"
MANIFEST_FILENAME = "manifest.yaml"
CHECKSUM_FILENAME = "SHA256SUMS"
ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ShortcutAngleExportError(ValueError):
    """Raised when the checkpoint or export contract is unsafe."""


class SteerNet(nn.Module):
    """Exact ResNet18 image+speed model used by xycar_train_sumin.ipynb."""

    def __init__(self) -> None:
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(513, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        image: torch.Tensor,
        speed_normalized: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(image).flatten(1)
        return self.head(torch.cat([features, speed_normalized], dim=1))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a square-warp shortcut angle checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--fixed-speed", type=float, default=23.0)
    parser.add_argument(
        "--warp-config",
        default="config/front_cam_policy_preprocess.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/models",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    artifact = export_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        artifact_id=args.artifact_id,
        fixed_speed=args.fixed_speed,
        warp_config_path=Path(args.warp_config),
        output_root=Path(args.output_root),
    )
    print(f"exported shortcut angle artifact: {artifact}")
    return 0


def export_checkpoint(
    *,
    checkpoint_path: Path,
    artifact_id: str,
    fixed_speed: float,
    warp_config_path: Path,
    output_root: Path,
) -> Path:
    if not ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
        raise ShortcutAngleExportError("artifact id is invalid")
    if not math.isfinite(fixed_speed) or not 0.0 <= fixed_speed <= 30.0:
        raise ShortcutAngleExportError("fixed speed must be in [0,30]")
    checkpoint_path = checkpoint_path.expanduser().resolve()
    warp_config_path = warp_config_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise FileNotFoundError(f"checkpoint is missing or unsafe: {checkpoint_path}")
    warp = _load_warp(warp_config_path)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ShortcutAngleExportError(
            "checkpoint must be a raw model state dict"
        )
    model = SteerNet().eval()
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ShortcutAngleExportError(
            f"checkpoint does not match the notebook model: {exc}"
        ) from exc

    image = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    speed = torch.tensor(
        [[fixed_speed / SPEED_DIVISOR]],
        dtype=torch.float32,
    )
    with torch.inference_mode():
        eager = model(image, speed)
        traced = torch.jit.trace(model, (image, speed), strict=True)
        traced_output = traced(image, speed)
    _validate_output(eager)
    _validate_output(traced_output)
    if not torch.allclose(eager, traced_output, atol=1e-6, rtol=1e-6):
        raise ShortcutAngleExportError("traced output differs from eager output")

    artifact_dir = output_root / artifact_id
    if artifact_dir.exists() or artifact_dir.is_symlink():
        raise FileExistsError(f"artifact already exists: {artifact_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_root / f".incoming-{artifact_id}-{os.getpid()}"
    if temporary_dir.exists() or temporary_dir.is_symlink():
        raise FileExistsError(f"temporary path exists: {temporary_dir}")
    temporary_dir.mkdir()
    try:
        model_path = temporary_dir / MODEL_FILENAME
        traced.save(str(model_path))
        reloaded = torch.jit.load(str(model_path), map_location="cpu").eval()
        with torch.inference_mode():
            reloaded_output = reloaded(image, speed)
        _validate_output(reloaded_output)
        if not torch.allclose(eager, reloaded_output, atol=1e-6, rtol=1e-6):
            raise ShortcutAngleExportError(
                "reloaded TorchScript output differs from eager output"
            )

        manifest = _manifest(
            artifact_id=artifact_id,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=_sha256_file(checkpoint_path),
            fixed_speed=float(fixed_speed),
            warp_config_path=warp_config_path,
            warp=warp,
        )
        manifest_path = temporary_dir / MANIFEST_FILENAME
        manifest_path.write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )
        checksum_path = temporary_dir / CHECKSUM_FILENAME
        checksum_path.write_text(
            "".join(
                f"{_sha256_file(temporary_dir / name)}  {name}\n"
                for name in (MODEL_FILENAME, MANIFEST_FILENAME)
            ),
            encoding="utf-8",
        )
        temporary_dir.rename(artifact_dir)
    except BaseException:
        if temporary_dir.is_dir():
            shutil.rmtree(temporary_dir)
        raise
    return artifact_dir


def _load_warp(path: Path) -> dict[str, float | int]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"warp config is missing or unsafe: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ShortcutAngleExportError("warp config schema must be version 1")
    warp = payload.get("warp")
    expected = {
        "top_y",
        "bottom_y",
        "top_left_x",
        "top_right_x",
        "bottom_left_x",
        "bottom_right_x",
        "bev_width",
        "bev_height",
        "dst_left_x",
        "dst_right_x",
    }
    if not isinstance(warp, Mapping) or set(warp) != expected:
        raise ShortcutAngleExportError("warp parameter keys are incompatible")
    result = dict(warp)
    if result["bev_width"] != IMAGE_SIZE or result["bev_height"] != IMAGE_SIZE:
        raise ShortcutAngleExportError("warp output must be 224x224")
    return result


def _manifest(
    *,
    artifact_id: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    fixed_speed: float,
    warp_config_path: Path,
    warp: Mapping[str, float | int],
) -> dict[str, object]:
    canonical_warp = json.dumps(
        {"schema_version": 1, "warp": dict(warp)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "training_run": checkpoint_path.parent.name,
        },
        "model": {
            "format": "torchscript",
            "file": MODEL_FILENAME,
            "architecture": "resnet18_speed_conditioned_angle",
            "prediction_mode": "angle_regression_fixed_speed",
            "input": {
                "kind": "tuple",
                "order": ["images", "speed_normalized"],
                "images": {
                    "color_space": "RGB",
                    "dtype": "float32",
                    "shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
                },
                "speed_normalized": {
                    "dtype": "float32",
                    "shape": [1, 1],
                    "unit": "motor_speed",
                    "normalization": (
                        "value / runtime.speed_normalization_divisor"
                    ),
                },
            },
            "output": {
                "kind": "tensor",
                "name": "angle_normalized",
                "dtype": "float32",
                "shape": [1, 1],
                "range": [-1.0, 1.0],
                "runtime_mapping": "value * 100",
            },
        },
        "preprocessing": {
            "geometry": "perspective_road_warp_then_bicubic_resize",
            "image_size": IMAGE_SIZE,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "road_warp": {
                "schema_version": 1,
                "parameters": dict(warp),
                "sha256": hashlib.sha256(canonical_warp).hexdigest(),
                "source_point_order": [
                    "bottom_left",
                    "top_left",
                    "top_right",
                    "bottom_right",
                ],
                "coordinate_space": "normalized_input_frame",
                "interpolation": "bilinear",
                "config_path": str(warp_config_path),
            },
            "training_augmentation": {
                "horizontal_flip_probability": 0.0,
            },
        },
        "runtime": {
            "fixed_speed": fixed_speed,
            "speed_normalization_divisor": SPEED_DIVISOR,
            "torch_num_threads": 8,
            "warmup_count": 3,
        },
        "label_contract": {
            "prediction_mode": "angle_regression_fixed_speed",
            "output_shape": [1, 1],
            "angle": {
                "unit": "normalized_percent",
                "range": [-100.0, 100.0],
                "model_output_range": [-1.0, 1.0],
                "runtime_mapping": "angle_normalized * 100",
            },
            "speed": {
                "unit": "motor_speed",
                "source": "runtime.fixed_speed",
                "range": [0.0, 30.0],
            },
        },
        "steering_contract": steering_contract_mapping(),
    }


def _validate_output(output: object) -> None:
    if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 1):
        raise ShortcutAngleExportError("model output must be tensor [1,1]")
    if not bool(torch.isfinite(output).all()):
        raise ShortcutAngleExportError("model output must be finite")
    if not -1.0 <= float(output.item()) <= 1.0:
        raise ShortcutAngleExportError("model output must be in [-1,1]")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
