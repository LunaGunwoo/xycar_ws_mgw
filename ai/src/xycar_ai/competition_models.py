"""Temporal traffic-signal and shortcut policies for competition driving."""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import nn

from xycar_ai.competition_data import LAMP_NAMES, SHORTCUT_PHASES
from xycar_ai.front_cam_policy_data import NUM_COMMAND_CLASSES


SIGNAL_STATUS_NAMES = (
    "approach",
    "visible",
    "readable",
    *LAMP_NAMES,
)
DEFAULT_SIGNAL_BACKBONE = "mobilenetv3_small_100.lamb_in1k"
DEFAULT_SHORTCUT_BACKBONE = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"


@dataclass(frozen=True)
class SignalModelConfig:
    backbone: str = DEFAULT_SIGNAL_BACKBONE
    pretrained: bool = True
    hidden_size: int = 256
    input_height: int = 160
    input_width: int = 320


@dataclass(frozen=True)
class ShortcutModelConfig:
    backbone: str = DEFAULT_SHORTCUT_BACKBONE
    pretrained: bool = True
    hidden_size: int = 256
    image_size: int = 224
    horizon_steps: int = 20


class SignalTemporalPolicy(nn.Module):
    """Encode an upper-frame ROI and temporally classify signal lamps."""

    def __init__(self, config: SignalModelConfig = SignalModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            num_classes=0,
            global_pool="avg",
        )
        # timm MobileNetV3 exposes ``num_features`` before its final conv head,
        # while ``forward_features``/the classifier-free forward path returns
        # ``head_hidden_size`` channels. Prefer it when present so
        # the exported model and training model agree on the actual tensor.
        head_hidden_size = getattr(self.encoder, "head_hidden_size", None)
        feature_size = int(head_hidden_size or self.encoder.num_features)
        self.embedding = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, config.hidden_size),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            config.hidden_size,
            config.hidden_size,
            batch_first=True,
        )
        self.status_head = nn.Linear(
            config.hidden_size,
            len(SIGNAL_STATUS_NAMES),
        )
        self.bbox_head = nn.Linear(config.hidden_size, 4)
        self.progress_head = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        images: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError("signal images must have shape [B,T,3,H,W]")
        batch, steps, channels, height, width = images.shape
        if (
            channels != 3
            or height != self.config.input_height
            or width != self.config.input_width
        ):
            raise ValueError("signal image shape does not match model config")
        features = self.encoder(images.reshape(batch * steps, 3, height, width))
        embeddings = self.embedding(features).reshape(batch, steps, -1)
        temporal, next_hidden = self.gru(embeddings, hidden)
        return {
            "status_logits": self.status_head(temporal),
            "bbox": self._decode_bbox(self.bbox_head(temporal)),
            "progress": torch.sigmoid(self.progress_head(temporal)).squeeze(-1),
            "hidden": next_hidden,
        }

    def step(
        self,
        image: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        result = self.forward(image.unsqueeze(1), hidden)
        return (
            result["status_logits"][:, -1],
            result["bbox"][:, -1],
            result["progress"][:, -1],
            result["hidden"],
        )

    @staticmethod
    def _decode_bbox(raw: torch.Tensor) -> torch.Tensor:
        values = torch.sigmoid(raw)
        center = values[..., :2]
        size = values[..., 2:] * 0.95
        minimum = (center - size / 2.0).clamp(0.0, 1.0)
        maximum = (center + size / 2.0).clamp(0.0, 1.0)
        return torch.cat((minimum, maximum), dim=-1)


class ShortcutTemporalPolicy(nn.Module):
    """Remember a left-signal decision through the complete shortcut."""

    def __init__(self, config: ShortcutModelConfig = ShortcutModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            num_classes=0,
            img_size=config.image_size,
        )
        feature_size = int(self.encoder.num_features)
        self.embedding = nn.Sequential(
            nn.LayerNorm(feature_size + 2),
            nn.Linear(feature_size + 2, config.hidden_size),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            config.hidden_size,
            config.hidden_size,
            batch_first=True,
        )
        action_outputs = config.horizon_steps * NUM_COMMAND_CLASSES
        self.angle_head = nn.Linear(config.hidden_size, action_outputs)
        self.speed_head = nn.Linear(config.hidden_size, action_outputs)
        self.phase_head = nn.Linear(config.hidden_size, len(SHORTCUT_PHASES))
        self.handoff_head = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        images: torch.Tensor,
        previous_commands: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError("shortcut images must have shape [B,T,3,H,W]")
        batch, steps, channels, height, width = images.shape
        if (
            channels != 3
            or height != self.config.image_size
            or width != self.config.image_size
        ):
            raise ValueError("shortcut image shape does not match model config")
        if tuple(previous_commands.shape) != (batch, steps, 2):
            raise ValueError("previous_commands must have shape [B,T,2]")
        features = self.encoder(images.reshape(batch * steps, 3, height, width))
        features = features.reshape(batch, steps, -1)
        embeddings = self.embedding(
            torch.cat((features, previous_commands / 100.0), dim=-1)
        )
        temporal, next_hidden = self.gru(embeddings, hidden)
        action_shape = (
            batch,
            steps,
            self.config.horizon_steps,
            NUM_COMMAND_CLASSES,
        )
        return {
            "angle_logits": self.angle_head(temporal).reshape(action_shape),
            "speed_logits": self.speed_head(temporal).reshape(action_shape),
            "phase_logits": self.phase_head(temporal),
            "handoff_logits": self.handoff_head(temporal).squeeze(-1),
            "hidden": next_hidden,
        }

    def step(
        self,
        image: torch.Tensor,
        previous_command: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        result = self.forward(
            image.unsqueeze(1),
            previous_command.unsqueeze(1),
            hidden,
        )
        return (
            result["angle_logits"][:, -1],
            result["speed_logits"][:, -1],
            result["phase_logits"][:, -1],
            result["handoff_logits"][:, -1],
            result["hidden"],
        )


class SignalStepWrapper(nn.Module):
    """Tuple-only TorchScript export boundary for one signal frame."""

    def __init__(self, policy: SignalTemporalPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        image: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy.step(image, hidden)


class ShortcutStepWrapper(nn.Module):
    """Tuple-only TorchScript export boundary for one shortcut frame."""

    def __init__(self, policy: ShortcutTemporalPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        image: torch.Tensor,
        previous_command: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return self.policy.step(image, previous_command, hidden)
