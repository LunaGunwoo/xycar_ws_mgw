from __future__ import annotations

from collections.abc import Mapping

import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn

from xycar_ai.front_cam_policy_data import NUM_COMMAND_CLASSES

DEFAULT_MODEL_NAME = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"


class TaskTokenViTPolicy(nn.Module):
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: bool = True,
        image_size: int = 224,
        num_classes: int = NUM_COMMAND_CLASSES,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = bool(pretrained)
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)

        # Loading pretrained weights is intentionally strict. A network or
        # cache failure must not silently turn this run into training from
        # random initialization.
        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=image_size,
        )
        if not isinstance(backbone, VisionTransformer):
            raise TypeError(
                "TaskTokenViTPolicy requires a timm VisionTransformer; "
                f"{model_name!r} produced {type(backbone).__name__}"
            )
        self._validate_backbone(backbone)
        self.backbone = backbone
        self.model_data_config = dict(timm.data.resolve_model_data_config(backbone))

        embed_dim = int(backbone.embed_dim)
        self.task_tokens = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.task_pos_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.angle_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)
        )
        self.speed_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)
        )
        nn.init.trunc_normal_(self.task_tokens, std=0.02)
        nn.init.trunc_normal_(self.task_pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(images)
        return {
            "angle_logits": self.angle_head(features[:, -2]),
            "speed_logits": self.speed_head(features[:, -1]),
        }

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        image_tokens = backbone.patch_embed(images)
        batch_size = image_tokens.shape[0]
        cls_token = backbone.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_token, image_tokens), dim=1)
        tokens = tokens + backbone.pos_embed
        task_tokens = self.task_tokens.expand(batch_size, -1, -1)
        task_tokens = task_tokens + self.task_pos_embed
        tokens = torch.cat((tokens, task_tokens), dim=1)
        tokens = backbone.pos_drop(tokens)
        tokens = backbone.patch_drop(tokens)
        tokens = backbone.norm_pre(tokens)
        tokens = backbone.blocks(tokens)
        return backbone.norm(tokens)

    def preprocessing_contract(self) -> dict[str, object]:
        return {
            "source": "timm.resolve_model_data_config",
            "geometry": "full_frame_bicubic_resize",
            "image_size": self.image_size,
            **_serializable_data_config(self.model_data_config),
        }

    @staticmethod
    def _validate_backbone(backbone: VisionTransformer) -> None:
        if getattr(backbone, "no_embed_class", False):
            raise TypeError("no_embed_class ViT backbones are unsupported")
        if getattr(backbone, "num_prefix_tokens", None) != 1:
            raise TypeError("the policy requires exactly one CLS prefix token")
        if getattr(backbone, "cls_token", None) is None:
            raise TypeError("the policy requires a CLS token")
        expected = 1 + int(backbone.patch_embed.num_patches)
        actual = int(backbone.pos_embed.shape[1])
        if actual != expected:
            raise TypeError(
                f"fixed ViT position embeddings are required: {actual} != {expected}"
            )


def _serializable_data_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    keys = (
        "input_size",
        "interpolation",
        "crop_pct",
        "crop_mode",
        "mean",
        "std",
    )
    result: dict[str, object] = {}
    for key in keys:
        value = config.get(key)
        if isinstance(value, tuple):
            result[key] = list(value)
        elif isinstance(value, (str, int, float, bool, list)) or value is None:
            result[key] = value
    return result
