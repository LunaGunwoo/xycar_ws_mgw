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


def test_export_checkpoint_writes_verified_artifact(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(export_module, "TaskTokenViTPolicy", _TinyPolicy)
    checkpoint_path = tmp_path / "best.pt"
    model = _TinyPolicy(model_name="tiny", pretrained=False, image_size=16)
    torch.save(
        {
            "epoch": 6,
            "best_epoch": 6,
            "best_score": 12.5,
            "config": {"model": {"name": "tiny", "image_size": 16}},
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
    )

    assert {path.name for path in artifact.iterdir()} == {
        "model.ts",
        "manifest.yaml",
        "SHA256SUMS",
    }
    verify_artifact(artifact)
    manifest = yaml.safe_load((artifact / "manifest.yaml").read_text())
    assert manifest["artifact_id"] == "fixture-policy"
    assert manifest["source"]["best_epoch"] == 6
    assert manifest["model"]["input"]["shape"] == [1, 3, 16, 16]
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
