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
    (directory / "test_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
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
        checkpoint_path=_write_checkpoint(
            tmp_path / "shortcut-run", kind="shortcut"
        ),
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
    with pytest.raises(
        ValueError, match="base artifact lacks normalized steering"
    ):
        build_bundle(
            base_artifact=base,
            signal_artifact=signal,
            shortcut_artifact=shortcut,
            artifact_id="legacy-base-rejected",
            output_root=model_root,
        )
