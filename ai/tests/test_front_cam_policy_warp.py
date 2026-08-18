from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from xycar_ai.front_cam_policy_data import FrontCamPolicyDataset, PolicySample
from xycar_ai.front_cam_policy_warp import (
    RoadWarpConfig,
    load_road_warp_config,
    save_road_warp_config,
    warp_image_array,
)
from xycar_ai.front_cam_policy_warp_tuner import WarpTunerState


def test_canonical_warp_defaults_match_tracked_yaml():
    config_path = Path(__file__).parents[1] / "config" / "front_cam_policy_preprocess.yaml"

    assert load_road_warp_config(config_path) == RoadWarpConfig(
        top_y=0.5,
        bottom_y=0.933,
        top_left_x=0.34,
        top_right_x=0.66,
        bottom_left_x=0.0,
        bottom_right_x=1.0,
        bev_width=224,
        bev_height=224,
        dst_left_x=0.0,
        dst_right_x=1.0,
    )
    assert load_road_warp_config(config_path) == RoadWarpConfig()


def test_warp_yaml_round_trip_and_strict_validation(tmp_path: Path):
    path = tmp_path / "warp.yaml"
    expected = _identity_warp()
    save_road_warp_config(path, expected)

    assert load_road_warp_config(path) == expected
    assert not any(tmp_path.glob(".*.tmp"))

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["warp"]["unknown"] = 1
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="warp keys mismatch"):
        load_road_warp_config(path)


def test_identity_warp_and_dataset_warp_before_flip(tmp_path: Path):
    image_array = np.zeros((60, 80, 3), dtype=np.uint8)
    image_array[:, :40] = (255, 0, 0)
    image_array[:, 40:] = (0, 0, 255)
    config = _identity_warp()
    warped = warp_image_array(image_array, config)
    assert warped.shape == image_array.shape
    assert np.array_equal(warped, image_array)

    image_path = tmp_path / "asymmetric.png"
    Image.fromarray(image_array).save(image_path)
    sample = PolicySample(
        session_id="session",
        image_path=image_path,
        relative_image="Images/asymmetric.png",
        angle_raw=-25.2,
        speed_raw=25.0,
        angle=-25,
        speed=25,
        angle_class_id=75,
        speed_class_id=125,
    )
    dataset = FrontCamPolicyDataset(
        [sample],
        transform=lambda image: np.asarray(image),
        horizontal_flip_probability=1.0,
        road_warp=config,
    )

    item = dataset[0]
    assert np.array_equal(item["image_tensor"], np.fliplr(image_array))
    assert item["angle_raw"] == 25.2
    assert item["angle"] == 25
    assert item["angle_class_id"] == 125
    assert item["speed"] == 25
    assert item["speed_class_id"] == 125


def test_tuner_only_writes_on_save_and_reset_discards_pending(tmp_path: Path):
    path = tmp_path / "warp.yaml"
    save_road_warp_config(path, _identity_warp())
    original_bytes = path.read_bytes()
    state = WarpTunerState(path)

    state.set_value("top_y", 0.2)
    assert state.has_pending_changes()
    assert path.read_bytes() == original_bytes
    state.reset()
    assert not state.has_pending_changes()
    assert path.read_bytes() == original_bytes

    state.set_value("top_y", 0.2)
    state.save()
    assert load_road_warp_config(path).top_y == 0.2
    assert path.read_bytes() != original_bytes


def _identity_warp() -> RoadWarpConfig:
    return RoadWarpConfig(
        top_y=0.0,
        bottom_y=1.0,
        top_left_x=0.0,
        top_right_x=1.0,
        bottom_left_x=0.0,
        bottom_right_x=1.0,
        bev_width=80,
        bev_height=60,
        dst_left_x=0.0,
        dst_right_x=1.0,
    )
