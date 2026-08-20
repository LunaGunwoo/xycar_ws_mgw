from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from xycar_ai.compact_control import (
    COMPACT_CONTROL_ENCODING,
    DRIVER_ANGLE_OFFSET,
    LEGACY_CONTROL_ENCODING,
    NORMALIZED_TO_DRIVER_SCALE,
)
from xycar_ai.front_cam_policy_data import COMMAND_OFFSET

ANGLE_BUCKETS = {
    "hard_left": (-100, -61),
    "left": (-60, -11),
    "near_zero": (-10, 10),
    "right": (11, 60),
    "hard_right": (61, 100),
}


@dataclass
class ClassificationMetricAccumulator:
    split_name: str
    control_encoding: str = LEGACY_CONTROL_ENCODING
    sample_count: int = 0
    total_loss_sum: float = 0.0
    angle_loss_sum: float = 0.0
    speed_loss_sum: float = 0.0
    emd_loss_sum: float = 0.0
    angle_exact_count: int = 0
    speed_exact_count: int = 0
    angle_within_5_count: int = 0
    speed_within_5_count: int = 0
    angle_within_10_count: int = 0
    speed_within_10_count: int = 0
    angle_abs_error_sum: float = 0.0
    speed_abs_error_sum: float = 0.0
    angle_expected_abs_error_sum: float = 0.0
    speed_expected_abs_error_sum: float = 0.0
    horizontal_flip_count: int = 0
    bucket_counts: dict[str, int] = field(default_factory=dict)
    bucket_abs_error_sums: dict[str, float] = field(default_factory=dict)
    bucket_within_10_counts: dict[str, int] = field(default_factory=dict)
    generation_counts: dict[int, int] = field(default_factory=dict)
    generation_angle_abs_error_sums: dict[int, float] = field(default_factory=dict)
    generation_speed_abs_error_sums: dict[int, float] = field(default_factory=dict)
    source_generation_counts: dict[tuple[str, int], int] = field(default_factory=dict)
    source_generation_angle_abs_error_sums: dict[tuple[str, int], float] = field(
        default_factory=dict
    )
    source_generation_speed_abs_error_sums: dict[tuple[str, int], float] = field(
        default_factory=dict
    )
    source_generation_angle_within_10_counts: dict[tuple[str, int], int] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for bucket in ANGLE_BUCKETS:
            self.bucket_counts.setdefault(bucket, 0)
            self.bucket_abs_error_sums.setdefault(bucket, 0.0)
            self.bucket_within_10_counts.setdefault(bucket, 0)

    @torch.no_grad()
    def update(
        self,
        *,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, object],
        total_loss: torch.Tensor,
        angle_loss: torch.Tensor,
        speed_loss: torch.Tensor,
        emd_loss: torch.Tensor,
    ) -> None:
        angle_true = _tensor(batch["angle"], outputs["angle_logits"].device).float()
        speed_true = _tensor(batch["speed"], outputs["speed_logits"].device).float()
        angle_class_true = _tensor(
            batch["angle_class_id"], outputs["angle_logits"].device
        ).long()
        speed_class_true = _tensor(
            batch["speed_class_id"], outputs["speed_logits"].device
        ).long()
        horizontal_flipped = _tensor(
            batch.get("horizontal_flipped", False), outputs["angle_logits"].device
        ).bool()
        generation = _tensor(
            batch.get("generation", 0), outputs["angle_logits"].device
        ).long()
        batch_size = int(angle_true.numel())

        angle_pred_class = outputs["angle_logits"].argmax(dim=1)
        speed_pred_class = outputs["speed_logits"].argmax(dim=1)
        if self.control_encoding == COMPACT_CONTROL_ENCODING:
            angle_pred = (
                angle_pred_class.float() - DRIVER_ANGLE_OFFSET
            ) / NORMALIZED_TO_DRIVER_SCALE
            speed_pred = speed_pred_class.float()
            angle_expected = (
                expected_class_id(outputs["angle_logits"])
                - DRIVER_ANGLE_OFFSET
            ) / NORMALIZED_TO_DRIVER_SCALE
            speed_expected = expected_class_id(outputs["speed_logits"])
        else:
            angle_pred = angle_pred_class.float() - COMMAND_OFFSET
            speed_pred = speed_pred_class.float() - COMMAND_OFFSET
            angle_expected = expected_command(outputs["angle_logits"])
            speed_expected = expected_command(outputs["speed_logits"])
        angle_error = (angle_pred - angle_true).abs()
        speed_error = (speed_pred - speed_true).abs()

        self.sample_count += batch_size
        self.total_loss_sum += float(total_loss.detach().cpu()) * batch_size
        self.angle_loss_sum += float(angle_loss.detach().cpu()) * batch_size
        self.speed_loss_sum += float(speed_loss.detach().cpu()) * batch_size
        self.emd_loss_sum += float(emd_loss.detach().cpu()) * batch_size
        self.angle_exact_count += int((angle_pred_class == angle_class_true).sum())
        self.speed_exact_count += int((speed_pred_class == speed_class_true).sum())
        self.angle_within_5_count += int((angle_error <= 5).sum())
        self.speed_within_5_count += int((speed_error <= 5).sum())
        self.angle_within_10_count += int((angle_error <= 10).sum())
        self.speed_within_10_count += int((speed_error <= 10).sum())
        self.angle_abs_error_sum += float(angle_error.sum().cpu())
        self.speed_abs_error_sum += float(speed_error.sum().cpu())
        self.angle_expected_abs_error_sum += float(
            (angle_expected - angle_true).abs().sum().cpu()
        )
        self.speed_expected_abs_error_sum += float(
            (speed_expected - speed_true).abs().sum().cpu()
        )
        self.horizontal_flip_count += int(horizontal_flipped.sum())
        for value in generation.unique().tolist():
            generation_id = int(value)
            mask = generation == generation_id
            generation_count = int(mask.sum())
            self.generation_counts[generation_id] = (
                self.generation_counts.get(generation_id, 0) + generation_count
            )
            self.generation_angle_abs_error_sums[generation_id] = (
                self.generation_angle_abs_error_sums.get(generation_id, 0.0)
                + float(angle_error[mask].sum().cpu())
            )
            self.generation_speed_abs_error_sums[generation_id] = (
                self.generation_speed_abs_error_sums.get(generation_id, 0.0)
                + float(speed_error[mask].sum().cpu())
            )
        raw_source_ids = batch.get("source_id")
        if raw_source_ids is not None:
            source_ids = (
                [raw_source_ids]
                if isinstance(raw_source_ids, str)
                else list(raw_source_ids)
            )
            if len(source_ids) != batch_size or not all(
                isinstance(source_id, str) and source_id for source_id in source_ids
            ):
                raise ValueError("batch source_id values must match the batch size")
            generation_values = generation.reshape(-1).tolist()
            angle_error_values = angle_error.reshape(-1).tolist()
            speed_error_values = speed_error.reshape(-1).tolist()
            for source_id, generation_id, angle_value, speed_value in zip(
                source_ids,
                generation_values,
                angle_error_values,
                speed_error_values,
                strict=True,
            ):
                key = (source_id, int(generation_id))
                self.source_generation_counts[key] = (
                    self.source_generation_counts.get(key, 0) + 1
                )
                self.source_generation_angle_abs_error_sums[key] = (
                    self.source_generation_angle_abs_error_sums.get(key, 0.0)
                    + float(angle_value)
                )
                self.source_generation_speed_abs_error_sums[key] = (
                    self.source_generation_speed_abs_error_sums.get(key, 0.0)
                    + float(speed_value)
                )
                self.source_generation_angle_within_10_counts[key] = (
                    self.source_generation_angle_within_10_counts.get(key, 0)
                    + int(angle_value <= 10)
                )
        for bucket, (low, high) in ANGLE_BUCKETS.items():
            mask = (angle_true >= low) & (angle_true <= high)
            bucket_count = int(mask.sum())
            self.bucket_counts[bucket] += bucket_count
            if bucket_count:
                self.bucket_abs_error_sums[bucket] += float(
                    angle_error[mask].sum().cpu()
                )
                self.bucket_within_10_counts[bucket] += int(
                    (angle_error[mask] <= 10).sum()
                )

    def compute(self) -> dict[str, float]:
        count = max(self.sample_count, 1)
        prefix = self.split_name
        metrics = {
            f"{prefix}_sample_count": float(self.sample_count),
            f"{prefix}_horizontal_flip_rate": self.horizontal_flip_count / count,
            f"{prefix}_loss": self.total_loss_sum / count,
            f"{prefix}_angle_loss": self.angle_loss_sum / count,
            f"{prefix}_speed_loss": self.speed_loss_sum / count,
            f"{prefix}_emd_loss": self.emd_loss_sum / count,
            f"{prefix}_angle_exact_acc": self.angle_exact_count / count,
            f"{prefix}_speed_exact_acc": self.speed_exact_count / count,
            f"{prefix}_angle_within_5_acc": self.angle_within_5_count / count,
            f"{prefix}_speed_within_5_acc": self.speed_within_5_count / count,
            f"{prefix}_angle_within_10_acc": self.angle_within_10_count / count,
            f"{prefix}_speed_within_10_acc": self.speed_within_10_count / count,
            f"{prefix}_angle_mae": self.angle_abs_error_sum / count,
            f"{prefix}_speed_mae": self.speed_abs_error_sum / count,
            f"{prefix}_angle_expected_mae": (self.angle_expected_abs_error_sum / count),
            f"{prefix}_speed_expected_mae": (self.speed_expected_abs_error_sum / count),
        }
        if self.control_encoding == COMPACT_CONTROL_ENCODING:
            metrics[f"{prefix}_angle_driver_mae"] = (
                self.angle_abs_error_sum
                / count
                * NORMALIZED_TO_DRIVER_SCALE
            )
        for bucket in ANGLE_BUCKETS:
            bucket_count = self.bucket_counts[bucket]
            bucket_prefix = f"{prefix}_angle_bucket_{bucket}"
            metrics[f"{bucket_prefix}_count"] = float(bucket_count)
            metrics[f"{bucket_prefix}_mae"] = (
                self.bucket_abs_error_sums[bucket] / bucket_count
                if bucket_count
                else 0.0
            )
            metrics[f"{bucket_prefix}_within_10_acc"] = (
                self.bucket_within_10_counts[bucket] / bucket_count
                if bucket_count
                else 0.0
            )
        for generation in sorted(self.generation_counts):
            generation_count = self.generation_counts[generation]
            generation_prefix = f"{prefix}_generation_{generation}"
            metrics[f"{generation_prefix}_sample_count"] = float(generation_count)
            metrics[f"{generation_prefix}_angle_mae"] = (
                self.generation_angle_abs_error_sums[generation] / generation_count
            )
            metrics[f"{generation_prefix}_speed_mae"] = (
                self.generation_speed_abs_error_sums[generation] / generation_count
            )
        for source_id, generation in sorted(self.source_generation_counts):
            key = (source_id, generation)
            source_generation_count = self.source_generation_counts[key]
            source_prefix = f"{prefix}_source_{source_id}_generation_{generation}"
            metrics[f"{source_prefix}_sample_count"] = float(
                source_generation_count
            )
            metrics[f"{source_prefix}_angle_mae"] = (
                self.source_generation_angle_abs_error_sums[key]
                / source_generation_count
            )
            metrics[f"{source_prefix}_angle_within_10_acc"] = (
                self.source_generation_angle_within_10_counts[key]
                / source_generation_count
            )
            metrics[f"{source_prefix}_speed_mae"] = (
                self.source_generation_speed_abs_error_sums[key]
                / source_generation_count
            )
        return metrics


@dataclass
class RegressionMetricAccumulator:
    """Accumulate scalar regression metrics in runtime command units."""

    split_name: str
    sample_count: int = 0
    total_loss_sum: float = 0.0
    angle_loss_sum: float = 0.0
    speed_loss_sum: float = 0.0
    angle_driver_abs_error_sum: float = 0.0
    speed_abs_error_sum: float = 0.0
    angle_within_5_count: int = 0
    angle_within_10_count: int = 0
    speed_within_5_count: int = 0
    speed_within_10_count: int = 0
    angle_sign_match_count: int = 0
    angle_prediction_sum: float = 0.0
    angle_prediction_square_sum: float = 0.0
    angle_target_sum: float = 0.0
    angle_target_square_sum: float = 0.0
    speed_prediction_sum: float = 0.0
    speed_prediction_square_sum: float = 0.0
    speed_target_sum: float = 0.0
    speed_target_square_sum: float = 0.0
    horizontal_flip_count: int = 0
    bucket_counts: dict[str, int] = field(default_factory=dict)
    bucket_abs_error_sums: dict[str, float] = field(default_factory=dict)
    bucket_within_10_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for bucket in ANGLE_BUCKETS:
            self.bucket_counts.setdefault(bucket, 0)
            self.bucket_abs_error_sums.setdefault(bucket, 0.0)
            self.bucket_within_10_counts.setdefault(bucket, 0)

    @torch.no_grad()
    def update(
        self,
        *,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, object],
        total_loss: torch.Tensor,
        angle_loss: torch.Tensor,
        speed_loss: torch.Tensor,
        emd_loss: torch.Tensor,
    ) -> None:
        angle_prediction = outputs["angle_driver"].reshape(-1)
        speed_prediction = outputs["speed"].reshape(-1)
        device = angle_prediction.device
        angle_target = _tensor(batch["angle_raw"], device).float().reshape(-1) * 0.5
        speed_target = _tensor(batch["speed_raw"], device).float().reshape(-1)
        if angle_prediction.shape != angle_target.shape:
            raise ValueError("regression angle output shape must match the target")
        if speed_prediction.shape != speed_target.shape:
            raise ValueError("regression speed output shape must match the target")
        if not bool(torch.isfinite(angle_prediction).all()) or not bool(
            torch.isfinite(speed_prediction).all()
        ):
            raise ValueError("regression output contains a non-finite value")

        driver_error = (angle_prediction - angle_target).abs()
        normalized_error = driver_error / NORMALIZED_TO_DRIVER_SCALE
        speed_error = (speed_prediction - speed_target).abs()
        normalized_target = angle_target / NORMALIZED_TO_DRIVER_SCALE
        horizontal_flipped = _tensor(
            batch.get("horizontal_flipped", False), device
        ).bool()
        batch_size = int(angle_target.numel())

        self.sample_count += batch_size
        self.total_loss_sum += float(total_loss.detach().cpu()) * batch_size
        self.angle_loss_sum += float(angle_loss.detach().cpu()) * batch_size
        self.speed_loss_sum += float(speed_loss.detach().cpu()) * batch_size
        self.angle_driver_abs_error_sum += float(driver_error.sum().cpu())
        self.speed_abs_error_sum += float(speed_error.sum().cpu())
        self.angle_within_5_count += int((normalized_error <= 5).sum())
        self.angle_within_10_count += int((normalized_error <= 10).sum())
        self.speed_within_5_count += int((speed_error <= 5).sum())
        self.speed_within_10_count += int((speed_error <= 10).sum())
        self.angle_sign_match_count += int(
            (torch.sign(angle_prediction) == torch.sign(angle_target)).sum()
        )
        self.horizontal_flip_count += int(horizontal_flipped.sum())
        for prediction, target, prefix in (
            (angle_prediction, angle_target, "angle"),
            (speed_prediction, speed_target, "speed"),
        ):
            # AMP may leave model outputs in float16. Squaring and summing a
            # normal 128-sample batch can overflow float16 even though every
            # scalar prediction is finite and in range.
            prediction = prediction.float()
            target = target.float()
            setattr(
                self,
                f"{prefix}_prediction_sum",
                getattr(self, f"{prefix}_prediction_sum")
                + float(prediction.sum().cpu()),
            )
            setattr(
                self,
                f"{prefix}_prediction_square_sum",
                getattr(self, f"{prefix}_prediction_square_sum")
                + float(prediction.square().sum().cpu()),
            )
            setattr(
                self,
                f"{prefix}_target_sum",
                getattr(self, f"{prefix}_target_sum") + float(target.sum().cpu()),
            )
            setattr(
                self,
                f"{prefix}_target_square_sum",
                getattr(self, f"{prefix}_target_square_sum")
                + float(target.square().sum().cpu()),
            )
        for bucket, (low, high) in ANGLE_BUCKETS.items():
            mask = (normalized_target >= low) & (normalized_target <= high)
            bucket_count = int(mask.sum())
            self.bucket_counts[bucket] += bucket_count
            if bucket_count:
                self.bucket_abs_error_sums[bucket] += float(
                    normalized_error[mask].sum().cpu()
                )
                self.bucket_within_10_counts[bucket] += int(
                    (normalized_error[mask] <= 10).sum()
                )

    def compute(self) -> dict[str, float]:
        count = max(self.sample_count, 1)
        prefix = self.split_name
        driver_mae = self.angle_driver_abs_error_sum / count
        metrics = {
            f"{prefix}_sample_count": float(self.sample_count),
            f"{prefix}_horizontal_flip_rate": self.horizontal_flip_count / count,
            f"{prefix}_loss": self.total_loss_sum / count,
            f"{prefix}_angle_loss": self.angle_loss_sum / count,
            f"{prefix}_speed_loss": self.speed_loss_sum / count,
            f"{prefix}_emd_loss": 0.0,
            f"{prefix}_angle_exact_acc": 0.0,
            f"{prefix}_speed_exact_acc": 0.0,
            f"{prefix}_angle_within_5_acc": self.angle_within_5_count / count,
            f"{prefix}_speed_within_5_acc": self.speed_within_5_count / count,
            f"{prefix}_angle_within_10_acc": self.angle_within_10_count / count,
            f"{prefix}_speed_within_10_acc": self.speed_within_10_count / count,
            f"{prefix}_angle_mae": driver_mae / NORMALIZED_TO_DRIVER_SCALE,
            f"{prefix}_angle_driver_mae": driver_mae,
            f"{prefix}_speed_mae": self.speed_abs_error_sum / count,
            f"{prefix}_angle_sign_acc": self.angle_sign_match_count / count,
            f"{prefix}_angle_prediction_std": self._std("angle", "prediction"),
            f"{prefix}_angle_target_std": self._std("angle", "target"),
            f"{prefix}_speed_prediction_std": self._std("speed", "prediction"),
            f"{prefix}_speed_target_std": self._std("speed", "target"),
        }
        for bucket in ANGLE_BUCKETS:
            bucket_count = self.bucket_counts[bucket]
            bucket_prefix = f"{prefix}_angle_bucket_{bucket}"
            metrics[f"{bucket_prefix}_count"] = float(bucket_count)
            metrics[f"{bucket_prefix}_mae"] = (
                self.bucket_abs_error_sums[bucket] / bucket_count
                if bucket_count
                else 0.0
            )
            metrics[f"{bucket_prefix}_within_10_acc"] = (
                self.bucket_within_10_counts[bucket] / bucket_count
                if bucket_count
                else 0.0
            )
        return metrics

    def _std(self, output: str, role: str) -> float:
        count = max(self.sample_count, 1)
        value_sum = getattr(self, f"{output}_{role}_sum")
        square_sum = getattr(self, f"{output}_{role}_square_sum")
        variance = max(square_sum / count - (value_sum / count) ** 2, 0.0)
        return variance**0.5


def selection_score(
    metrics: dict[str, float],
    *,
    split_name: str = "val",
    speed_mae_weight: float = 0.25,
) -> float:
    return (
        metrics[f"{split_name}_angle_mae"]
        + speed_mae_weight * metrics[f"{split_name}_speed_mae"]
    )


def expected_command(logits: torch.Tensor) -> torch.Tensor:
    values = torch.arange(
        -COMMAND_OFFSET,
        COMMAND_OFFSET + 1,
        device=logits.device,
        dtype=logits.dtype,
    )
    return (logits.softmax(dim=1) * values).sum(dim=1)


def expected_class_id(logits: torch.Tensor) -> torch.Tensor:
    values = torch.arange(
        logits.shape[1],
        device=logits.device,
        dtype=logits.dtype,
    )
    return (logits.softmax(dim=1) * values).sum(dim=1)


def ordinal_emd_loss(logits: torch.Tensor, target_class: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    if logits.ndim != 2:
        raise ValueError(f"ordinal EMD logits must be 2D, got {logits.shape}")
    target_class = target_class.long()
    if target_class.shape != (logits.shape[0],):
        raise ValueError("ordinal EMD target shape must match the batch size")
    probabilities = logits.softmax(dim=1)
    targets = F.one_hot(target_class, num_classes=logits.shape[1]).to(logits.dtype)
    return (probabilities.cumsum(dim=1) - targets.cumsum(dim=1)).abs().mean()


def combine_policy_losses(
    *,
    angle_loss: torch.Tensor,
    speed_loss: torch.Tensor,
    angle_emd_loss: torch.Tensor,
    speed_emd_loss: torch.Tensor,
    speed_loss_weight: float,
    emd_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    emd_loss = angle_emd_loss + speed_loss_weight * speed_emd_loss
    total_loss = (
        angle_loss + speed_loss_weight * speed_loss + emd_loss_weight * emd_loss
    )
    return total_loss, emd_loss


def _tensor(value: object, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return torch.as_tensor(value, device=device)
