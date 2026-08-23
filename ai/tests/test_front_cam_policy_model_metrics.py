from __future__ import annotations

import pytest
import torch

from xycar_ai.compact_control import COMPACT_CONTROL_ENCODING
from xycar_ai.front_cam_policy_metrics import (
    ClassificationMetricAccumulator,
    RegressionMetricAccumulator,
    combine_policy_losses,
    ordinal_emd_loss,
    selection_score,
)
from xycar_ai.front_cam_policy_model import (
    AutoregressiveControlTokenViTPolicy,
    CONTINUOUS_REGRESSION_PREDICTION_MODE,
    TaskTokenViTPolicy,
)


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


@pytest.mark.parametrize("use_type_embedding", [False, True])
def test_ar_control_token_vit_output_shapes_and_shared_projection(
    use_type_embedding: bool,
):
    model = AutoregressiveControlTokenViTPolicy(
        model_name="vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        pretrained=False,
        image_size=32,
        history_frames=4,
        use_control_type_embedding=use_type_embedding,
    )
    model.eval()
    images = torch.zeros(2, 3, 32, 32)
    history = torch.tensor(
        [
            [[100, 125], [100, 125], [101, 124], [102, 123]],
            [[90, 120], [91, 121], [92, 122], [93, 123]],
        ],
        dtype=torch.long,
    )
    with torch.no_grad():
        features = model.forward_features(images, history)
        outputs = model(images, history)
        expected = torch.nn.functional.linear(
            features[:, -2:],
            model.control_token_embedding.weight[:201],
        ) + model.output_bias.unsqueeze(0)

    assert tuple(outputs["angle_logits"].shape) == (2, 201)
    assert tuple(outputs["speed_logits"].shape) == (2, 201)
    assert torch.equal(outputs["angle_logits"], expected[:, 0])
    assert torch.equal(outputs["speed_logits"], expected[:, 1])
    assert (model.control_type_embedding is not None) is use_type_embedding
    assert model.query_token_ids.tolist() == [201, 202]
    assert model.control_type_ids.tolist() == [0, 1] * 5
    assert "query_token_ids" not in model.state_dict()
    assert "control_type_ids" not in model.state_dict()


def test_ar_control_token_vit_rejects_invalid_history():
    model = AutoregressiveControlTokenViTPolicy(
        model_name="vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        pretrained=False,
        image_size=32,
    )
    images = torch.zeros(1, 3, 32, 32)
    with pytest.raises(ValueError, match=r"shape \[B,4,2\]"):
        model(images, torch.zeros(1, 3, 2, dtype=torch.long))
    with pytest.raises(TypeError, match="torch.int64"):
        model(images, torch.zeros(1, 4, 2))
    with pytest.raises(ValueError, match=r"\[0,200\]"):
        model(images, torch.full((1, 4, 2), 201, dtype=torch.long))


def test_compact_ar_uses_unknown_tokens_and_unequal_shared_outputs():
    model = AutoregressiveControlTokenViTPolicy(
        model_name="vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        pretrained=False,
        image_size=32,
        history_frames=4,
        control_encoding=COMPACT_CONTROL_ENCODING,
    ).eval()
    images = torch.zeros(2, 3, 32, 32)
    history = torch.tensor(
        [
            [[101, 102], [101, 102], [0, 50], [100, 80]],
            [[101, 102], [50, 65], [51, 66], [49, 64]],
        ],
        dtype=torch.long,
    )
    with torch.no_grad():
        features = model.forward_features(images, history)
        outputs = model(images, history)
    expected_angle = torch.nn.functional.linear(
        features[:, -2],
        model.control_token_embedding.weight[:101],
        model.angle_output_bias,
    )
    expected_speed = torch.nn.functional.linear(
        features[:, -1],
        model.control_token_embedding.weight[50:81],
        model.speed_output_bias,
    )

    assert tuple(outputs["angle_logits"].shape) == (2, 101)
    assert tuple(outputs["speed_logits"].shape) == (2, 31)
    assert torch.equal(outputs["angle_logits"], expected_angle)
    assert torch.equal(outputs["speed_logits"], expected_speed)
    assert model.query_token_ids.tolist() == [103, 104]
    with pytest.raises(ValueError, match="invalid token id"):
        model(images[:1], torch.tensor([[[0, 49]] * 4], dtype=torch.long))


def test_compact_ar_continuous_regression_outputs_scalar_ranges():
    model = AutoregressiveControlTokenViTPolicy(
        model_name="vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        pretrained=False,
        image_size=32,
        history_frames=4,
        control_encoding=COMPACT_CONTROL_ENCODING,
        prediction_mode=CONTINUOUS_REGRESSION_PREDICTION_MODE,
    ).eval()
    history = torch.tensor([[[50, 75]] * 4] * 2, dtype=torch.long)
    with torch.no_grad():
        outputs = model(torch.zeros(2, 3, 32, 32), history)

    assert set(outputs) == {"angle_driver", "speed"}
    assert tuple(outputs["angle_driver"].shape) == (2, 1)
    assert tuple(outputs["speed"].shape) == (2, 1)
    assert bool((outputs["angle_driver"].abs() <= 50.0).all())
    assert bool(((0.0 <= outputs["speed"]) & (outputs["speed"] <= 30.0)).all())
    assert isinstance(model.angle_regression_head[0], torch.nn.LayerNorm)
    assert model.angle_regression_head[1].out_features == 128
    assert model.angle_regression_head[3].p == pytest.approx(0.3)


def test_compact_ar_continuous_regression_supports_speed_35():
    model = AutoregressiveControlTokenViTPolicy(
        model_name="vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        pretrained=False,
        image_size=32,
        history_frames=4,
        control_encoding=COMPACT_CONTROL_ENCODING,
        prediction_mode=CONTINUOUS_REGRESSION_PREDICTION_MODE,
        speed_output_max=35.0,
    ).eval()
    history = torch.tensor([[[50, 85]] * 4], dtype=torch.long)
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 32, 32), history)

    assert model.speed_output_max == 35.0
    assert bool(((0.0 <= outputs["speed"]) & (outputs["speed"] <= 35.0)).all())


def test_regression_metrics_use_driver_and_normalized_angle_units():
    accumulator = RegressionMetricAccumulator("test")
    accumulator.update(
        outputs={
            "angle_driver": torch.tensor([[-40.0], [45.0]]),
            "speed": torch.tensor([[20.0], [24.0]]),
        },
        batch={
            "angle_raw": torch.tensor([-100.0, 100.0]),
            "speed_raw": torch.tensor([20.0, 25.0]),
        },
        total_loss=torch.tensor(1.0),
        angle_loss=torch.tensor(0.5),
        speed_loss=torch.tensor(1.0),
        emd_loss=torch.tensor(0.0),
    )
    metrics = accumulator.compute()

    assert metrics["test_angle_driver_mae"] == pytest.approx(7.5)
    assert metrics["test_angle_mae"] == pytest.approx(15.0)
    assert metrics["test_speed_mae"] == pytest.approx(0.5)
    assert metrics["test_angle_sign_acc"] == 1.0


def test_regression_prediction_std_accumulates_amp_outputs_in_float32():
    accumulator = RegressionMetricAccumulator("train")
    accumulator.update(
        outputs={
            "angle_driver": torch.full((128, 1), 50.0, dtype=torch.float16),
            "speed": torch.full((128, 1), 30.0, dtype=torch.float16),
        },
        batch={
            "angle_raw": torch.full((128,), 100.0),
            "speed_raw": torch.full((128,), 30.0),
        },
        total_loss=torch.tensor(0.0),
        angle_loss=torch.tensor(0.0),
        speed_loss=torch.tensor(0.0),
        emd_loss=torch.tensor(0.0),
    )
    metrics = accumulator.compute()

    assert metrics["train_angle_prediction_std"] == 0.0
    assert metrics["train_speed_prediction_std"] == 0.0


def test_compact_metrics_decode_angle_back_to_normalized_units():
    angle_logits = torch.full((2, 101), -1000.0)
    speed_logits = torch.full((2, 31), -1000.0)
    angle_logits[0, 0] = angle_logits[1, 100] = 1000.0
    speed_logits[0, 0] = speed_logits[1, 30] = 1000.0
    accumulator = ClassificationMetricAccumulator(
        "test",
        control_encoding=COMPACT_CONTROL_ENCODING,
    )
    accumulator.update(
        outputs={"angle_logits": angle_logits, "speed_logits": speed_logits},
        batch={
            "angle": torch.tensor([-100.0, 100.0]),
            "speed": torch.tensor([0.0, 30.0]),
            "angle_class_id": torch.tensor([0, 100]),
            "speed_class_id": torch.tensor([0, 30]),
        },
        total_loss=torch.tensor(0.0),
        angle_loss=torch.tensor(0.0),
        speed_loss=torch.tensor(0.0),
        emd_loss=torch.tensor(0.0),
    )
    metrics = accumulator.compute()
    assert metrics["test_angle_mae"] == 0.0
    assert metrics["test_angle_driver_mae"] == 0.0
    assert metrics["test_speed_mae"] == 0.0


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

    angle_only_total, angle_only_emd = combine_policy_losses(
        angle_loss=torch.tensor(2.0),
        speed_loss=torch.tensor(4.0),
        angle_emd_loss=torch.tensor(10.0),
        speed_emd_loss=torch.tensor(8.0),
        speed_loss_weight=0.0,
        emd_loss_weight=0.2,
    )
    assert angle_only_emd.item() == pytest.approx(10.0)
    assert angle_only_total.item() == pytest.approx(4.0)


def test_selection_score_prioritizes_angle():
    metrics = {"val_angle_mae": 4.0, "val_speed_mae": 8.0}
    assert selection_score(metrics) == pytest.approx(6.0)
    assert selection_score(metrics, speed_mae_weight=0.0) == pytest.approx(4.0)


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


def test_source_generation_metrics_include_mae_within_10_and_speed():
    true_angles = torch.tensor([0, 20, -20, 40])
    true_speeds = torch.tensor([10, 10, 20, 20])
    logits = torch.full((4, 201), -1000.0)
    logits[:, 100] = 1000.0
    accumulator = ClassificationMetricAccumulator("val")
    accumulator.update(
        outputs={"angle_logits": logits, "speed_logits": logits},
        batch={
            "angle": true_angles,
            "speed": true_speeds,
            "angle_class_id": true_angles + 100,
            "speed_class_id": true_speeds + 100,
            "generation": torch.tensor([0, 0, 1, 2]),
            "source_id": ["manual", "manual", "guided", "guided"],
        },
        total_loss=torch.tensor(0.0),
        angle_loss=torch.tensor(0.0),
        speed_loss=torch.tensor(0.0),
        emd_loss=torch.tensor(0.0),
    )
    metrics = accumulator.compute()

    assert metrics["val_source_manual_generation_0_angle_mae"] == 10.0
    assert metrics["val_source_manual_generation_0_angle_within_10_acc"] == 0.5
    assert metrics["val_source_manual_generation_0_speed_mae"] == 10.0
    assert metrics["val_source_guided_generation_1_angle_mae"] == 20.0
    assert metrics["val_source_guided_generation_2_speed_mae"] == 20.0
