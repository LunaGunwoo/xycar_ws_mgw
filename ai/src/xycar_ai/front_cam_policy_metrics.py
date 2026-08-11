from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

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
        batch_size = int(angle_true.numel())

        angle_pred_class = outputs["angle_logits"].argmax(dim=1)
        speed_pred_class = outputs["speed_logits"].argmax(dim=1)
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


def selection_score(metrics: dict[str, float], *, split_name: str = "val") -> float:
    return (
        metrics[f"{split_name}_angle_mae"] + 0.25 * metrics[f"{split_name}_speed_mae"]
    )


def expected_command(logits: torch.Tensor) -> torch.Tensor:
    values = torch.arange(
        -COMMAND_OFFSET,
        COMMAND_OFFSET + 1,
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
