from __future__ import annotations

import pytest
import torch

from xycar_ai.front_cam_policy_metrics import (
    ClassificationMetricAccumulator,
    combine_policy_losses,
    ordinal_emd_loss,
    selection_score,
)
from xycar_ai.front_cam_policy_model import TaskTokenViTPolicy


def test_task_token_vit_tiny_output_shapes():
    model = TaskTokenViTPolicy(
        model_name="vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        pretrained=False,
        image_size=224,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(torch.zeros(2, 3, 224, 224))

    assert tuple(outputs["angle_logits"].shape) == (2, 201)
    assert tuple(outputs["speed_logits"].shape) == (2, 201)
    assert model.preprocessing_contract()["geometry"] == "full_frame_bicubic_resize"


def test_task_token_vit_small_output_shapes():
    model = TaskTokenViTPolicy(
        model_name="vit_small_patch16_224.augreg_in21k_ft_in1k",
        pretrained=False,
        image_size=224,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 224, 224))

    assert tuple(outputs["angle_logits"].shape) == (1, 201)
    assert tuple(outputs["speed_logits"].shape) == (1, 201)
    assert sum(parameter.numel() for parameter in model.parameters()) == 21_823_506


def test_ordinal_emd_tracks_class_distance_and_combines_losses():
    logits = torch.full((3, 201), -1000.0)
    logits[0, 100] = 1000.0
    logits[1, 101] = 1000.0
    logits[2, 150] = 1000.0
    target = torch.tensor([100, 100, 100])
    losses = [
        ordinal_emd_loss(logits[index : index + 1], target[index : index + 1])
        for index in range(3)
    ]
    assert losses[0].item() == pytest.approx(0.0)
    assert losses[1].item() == pytest.approx(1.0 / 201.0)
    assert losses[2].item() == pytest.approx(50.0 / 201.0)

    total, emd = combine_policy_losses(
        angle_loss=torch.tensor(2.0),
        speed_loss=torch.tensor(4.0),
        angle_emd_loss=torch.tensor(10.0),
        speed_emd_loss=torch.tensor(8.0),
        speed_loss_weight=0.5,
        emd_loss_weight=0.2,
    )
    assert emd.item() == pytest.approx(14.0)
    assert total.item() == pytest.approx(6.8)


def test_selection_score_prioritizes_angle():
    metrics = {"val_angle_mae": 4.0, "val_speed_mae": 8.0}
    assert selection_score(metrics) == pytest.approx(6.0)


def test_angle_bucket_metrics_and_horizontal_flip_rate():
    true_angles = torch.tensor([-100, -20, 0, 20, 100])
    true_classes = true_angles + 100
    logits = torch.full((5, 201), -1000.0)
    logits[:, 100] = 1000.0
    accumulator = ClassificationMetricAccumulator("val")
    accumulator.update(
        outputs={"angle_logits": logits, "speed_logits": logits},
        batch={
            "angle": true_angles,
            "speed": torch.zeros(5),
            "angle_class_id": true_classes,
            "speed_class_id": torch.full((5,), 100),
            "horizontal_flipped": torch.tensor([True, False, True, False, False]),
        },
        total_loss=torch.tensor(0.0),
        angle_loss=torch.tensor(0.0),
        speed_loss=torch.tensor(0.0),
        emd_loss=torch.tensor(0.0),
    )
    metrics = accumulator.compute()

    assert metrics["val_horizontal_flip_rate"] == pytest.approx(0.4)
    for bucket in ("hard_left", "left", "near_zero", "right", "hard_right"):
        assert metrics[f"val_angle_bucket_{bucket}_count"] == 1.0
    assert metrics["val_angle_bucket_hard_left_mae"] == pytest.approx(100.0)
    assert metrics["val_angle_bucket_near_zero_mae"] == pytest.approx(0.0)
    assert metrics["val_angle_bucket_hard_right_within_10_acc"] == 0.0
