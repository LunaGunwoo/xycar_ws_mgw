import torch

from xycar_ai.competition_models import (
    ShortcutModelConfig,
    ShortcutTemporalPolicy,
    SignalModelConfig,
    SignalTemporalPolicy,
)


def test_signal_step_contract():
    model = SignalTemporalPolicy(
        SignalModelConfig(
            backbone="mobilenetv3_small_050",
            pretrained=False,
            hidden_size=32,
            input_height=64,
            input_width=96,
        )
    ).eval()
    outputs = model.step(
        torch.zeros(1, 3, 64, 96),
        torch.zeros(1, 1, 32),
    )
    assert [tuple(value.shape) for value in outputs] == [
        (1, 7),
        (1, 4),
        (1,),
        (1, 1, 32),
    ]


def test_shortcut_step_contract():
    model = ShortcutTemporalPolicy(
        ShortcutModelConfig(
            backbone="vit_tiny_patch16_224",
            pretrained=False,
            hidden_size=32,
            image_size=224,
            horizon_steps=3,
        )
    ).eval()
    outputs = model.step(
        torch.zeros(1, 3, 224, 224),
        torch.zeros(1, 2),
        torch.zeros(1, 1, 32),
    )
    assert [tuple(value.shape) for value in outputs] == [
        (1, 3, 201),
        (1, 3, 201),
        (1, 6),
        (1,),
        (1, 1, 32),
    ]
