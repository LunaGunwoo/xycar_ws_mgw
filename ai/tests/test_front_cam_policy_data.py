from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
import yaml
from conftest import write_session, write_split_manifest
from PIL import Image
from torchvision import transforms

from xycar_ai.config import DataConfig
from xycar_ai.front_cam_policy_data import (
    FrontCamPolicyDataset,
    PolicyDatasetError,
    build_policy_data_splits,
    command_class_id,
    compute_sqrt_inverse_frequency_weights,
    discover_policy_sessions,
    policy_dataset_stats,
    quantize_command,
)


def test_gamepad_speed_filter_and_fixed_split(tmp_path: Path):
    data_root = tmp_path / "datasets" / "teleop"
    names = [
        "20260803_150534_000_session",
        "20260810_130735_027_session",
        "20260810_130818_255_session",
    ]
    for index, name in enumerate(names):
        write_session(data_root, name, labels=[(-10.4 + index, 25.0)])
    write_session(
        data_root,
        "20260803_120333_759_session",
        labels=[(0.0, 15.0)],
        max_forward_speed=15.0,
    )
    write_session(
        data_root,
        "20260810_131250_029_session",
        labels=[(0.0, 25.0)],
        complete=False,
    )
    write_session(
        data_root,
        "20260810_131214_144_session",
        labels=[(0.0, 25.0)],
        control_mode="terminal",
    )
    manifest = write_split_manifest(
        tmp_path / "config" / "split.yaml",
        train=[names[0]],
        val=[names[1]],
        test=[names[2]],
    )
    config = _data_config(data_root, manifest)

    sessions = discover_policy_sessions(config)
    splits = build_policy_data_splits(config)
    stats = policy_dataset_stats(splits)

    assert [session.session_id for session in sessions] == names
    assert splits.manifest()["splits"]["train"]["sample_count"] == 1
    assert stats["all"]["sample_count"] == 3
    assert set(splits.manifest()["splits"]) == {"train", "val", "test"}


def test_minimum_forward_speed_filter_includes_boundary(tmp_path: Path):
    data_root = tmp_path / "datasets" / "teleop"
    expected = [
        "20260810_130735_027_session",
        "20260810_130818_255_session",
    ]
    write_session(
        data_root,
        "20260803_120333_759_session",
        labels=[(0.0, 15.0)],
        max_forward_speed=19.999,
    )
    write_session(
        data_root,
        expected[0],
        labels=[(0.0, 20.0)],
        max_forward_speed=20.0,
    )
    write_session(
        data_root,
        expected[1],
        labels=[(0.0, 25.0)],
        max_forward_speed=30.0,
    )
    config = DataConfig(
        root=data_root,
        split_manifest=tmp_path / "missing-split.yaml",
        require_all_matching_sessions=True,
        control_mode="gamepad",
        max_forward_speed=None,
        min_forward_speed=20.0,
        num_workers=0,
    )

    assert [
        session.session_id for session in discover_policy_sessions(config)
    ] == expected


def test_split_manifest_rejects_overlap_and_unlisted_session(tmp_path: Path):
    data_root = tmp_path / "datasets" / "teleop"
    names = [
        "20260810_130735_027_session",
        "20260810_130818_255_session",
        "20260810_130857_726_session",
        "20260810_130938_647_session",
    ]
    for name in names:
        write_session(data_root, name, labels=[(0.0, 25.0)])
    manifest = write_split_manifest(
        tmp_path / "config" / "split.yaml",
        train=[names[0]],
        val=[names[1]],
        test=[names[1]],
    )

    with pytest.raises(PolicyDatasetError, match="more than one split"):
        build_policy_data_splits(_data_config(data_root, manifest))

    write_split_manifest(
        manifest,
        train=[names[0]],
        val=[names[1]],
        test=[names[2]],
    )
    with pytest.raises(PolicyDatasetError, match="absent from the split manifest"):
        build_policy_data_splits(_data_config(data_root, manifest))


@pytest.mark.parametrize("invalid_value", ["nan", "inf", "-inf", "101"])
def test_dataset_rejects_invalid_label(tmp_path: Path, invalid_value: str):
    data_root = tmp_path / "datasets" / "teleop"
    name = "20260810_130735_027_session"
    session = write_session(data_root, name, labels=[(0.0, 25.0)])
    csv_path = session / "samples.csv"
    rows = list(csv.reader(csv_path.open("r", encoding="utf-8")))
    rows[1][2] = invalid_value
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        csv.writer(csv_file).writerows(rows)

    with pytest.raises(PolicyDatasetError):
        discover_policy_sessions(
            _data_config(data_root, tmp_path / "missing-split.yaml")
        )


def test_dataset_rejects_unsafe_image_path(tmp_path: Path):
    data_root = tmp_path / "datasets" / "teleop"
    name = "20260810_130735_027_session"
    session = write_session(data_root, name, labels=[(0.0, 25.0)])
    csv_path = session / "samples.csv"
    rows = list(csv.reader(csv_path.open("r", encoding="utf-8")))
    rows[1][1] = "../outside.png"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        csv.writer(csv_file).writerows(rows)

    with pytest.raises(PolicyDatasetError, match="unsafe image path"):
        discover_policy_sessions(
            _data_config(data_root, tmp_path / "missing-split.yaml")
        )


def test_dataset_rejects_missing_csv_column(tmp_path: Path):
    data_root = tmp_path / "datasets" / "teleop"
    name = "20260810_130735_027_session"
    session = write_session(data_root, name, labels=[(0.0, 25.0)])
    csv_path = session / "samples.csv"
    rows = list(csv.reader(csv_path.open("r", encoding="utf-8")))
    speed_index = rows[0].index("speed")
    for row in rows:
        del row[speed_index]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        csv.writer(csv_file).writerows(rows)

    with pytest.raises(PolicyDatasetError, match="missing samples.csv fields"):
        discover_policy_sessions(
            _data_config(data_root, tmp_path / "missing-split.yaml")
        )


def test_dataset_rejects_metadata_sample_count_mismatch(tmp_path: Path):
    data_root = tmp_path / "datasets" / "teleop"
    name = "20260810_130735_027_session"
    session = write_session(data_root, name, labels=[(0.0, 25.0)])
    metadata_path = session / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["sample_count"] = 2
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=True), encoding="utf-8")

    with pytest.raises(PolicyDatasetError, match="sample_count mismatch"):
        discover_policy_sessions(
            _data_config(data_root, tmp_path / "missing-split.yaml")
        )


@pytest.mark.parametrize(
    ("missing_relative", "error_text"),
    [
        ("metadata.yaml", "missing metadata.yaml"),
        ("samples.csv", "missing samples.csv"),
        ("Images/1.png", "missing image"),
    ],
)
def test_dataset_rejects_missing_required_file(
    tmp_path: Path, missing_relative: str, error_text: str
):
    data_root = tmp_path / "datasets" / "teleop"
    name = "20260810_130735_027_session"
    session = write_session(data_root, name, labels=[(0.0, 25.0)])
    (session / missing_relative).unlink()

    with pytest.raises(PolicyDatasetError, match=error_text):
        discover_policy_sessions(
            _data_config(data_root, tmp_path / "missing-split.yaml")
        )


def test_quantization_and_class_weight_contract(tmp_path: Path):
    assert quantize_command(-100.4) == -100
    assert quantize_command(-1.5) == -2
    assert quantize_command(1.5) == 2
    assert command_class_id(-100.4) == 0
    assert command_class_id(100.4) == 200

    data_root = tmp_path / "datasets" / "teleop"
    name = "20260810_130735_027_session"
    write_session(
        data_root,
        name,
        labels=[(0.0, 25.0), (0.0, 25.0), (100.0, 15.0)],
    )
    samples = discover_policy_sessions(
        _data_config(data_root, tmp_path / "missing-split.yaml")
    )[0].samples
    weights = compute_sqrt_inverse_frequency_weights(
        samples,
        field="angle_class_id",
        min_weight=0.5,
        max_weight=3.0,
    )
    assert tuple(weights.shape) == (201,)
    assert float(weights.min()) >= 0.5
    assert float(weights.max()) <= 3.0
    assert weights[100] < weights[200]
    assert weights[0].item() == pytest.approx(3.0)

    baseline_weights = compute_sqrt_inverse_frequency_weights(
        samples,
        field="angle_class_id",
        min_weight=0.5,
        max_weight=3.0,
        mirror_probability=0.0,
    )
    mirrored_weights = compute_sqrt_inverse_frequency_weights(
        samples,
        field="angle_class_id",
        min_weight=0.5,
        max_weight=3.0,
        mirror_probability=0.5,
    )
    assert torch.equal(weights, baseline_weights)
    assert mirrored_weights[0].item() == pytest.approx(mirrored_weights[200].item())


def test_horizontal_flip_pairs_image_and_angle_labels(tmp_path: Path):
    data_root = tmp_path / "datasets" / "teleop"
    name = "20260810_130735_027_session"
    session = write_session(
        data_root,
        name,
        labels=[(20.4, 25.0), (0.0, 24.0)],
    )
    asymmetric = Image.new("RGB", (2, 1))
    asymmetric.putdata([(255, 0, 0), (0, 0, 255)])
    asymmetric.save(session / "Images" / "1.png")
    samples = discover_policy_sessions(
        _data_config(data_root, tmp_path / "missing-split.yaml")
    )[0].samples
    unflipped_dataset = FrontCamPolicyDataset(
        samples,
        transform=transforms.ToTensor(),
        horizontal_flip_probability=0.0,
    )
    flipped_dataset = FrontCamPolicyDataset(
        samples,
        transform=transforms.ToTensor(),
        horizontal_flip_probability=1.0,
    )

    unflipped = unflipped_dataset[0]
    flipped = flipped_dataset[0]
    assert torch.equal(
        flipped["image_tensor"], torch.flip(unflipped["image_tensor"], dims=[2])
    )
    assert unflipped["horizontal_flipped"] is False
    assert unflipped["angle_raw"] == pytest.approx(20.4)
    assert unflipped["angle"] == 20
    assert unflipped["angle_class_id"] == 120
    assert flipped["horizontal_flipped"] is True
    assert flipped["angle_raw"] == pytest.approx(-20.4)
    assert flipped["angle"] == -20
    assert flipped["angle_class_id"] == 80
    assert flipped["speed_raw"] == unflipped["speed_raw"]
    assert flipped["speed"] == unflipped["speed"]
    assert flipped["speed_class_id"] == unflipped["speed_class_id"]

    flipped_center = flipped_dataset[1]
    assert flipped_center["angle"] == 0
    assert flipped_center["angle_class_id"] == 100


def test_horizontal_flip_sequence_is_seeded(tmp_path: Path):
    data_root = tmp_path / "datasets" / "teleop"
    name = "20260810_130735_027_session"
    write_session(data_root, name, labels=[(10.0, 25.0)])
    samples = discover_policy_sessions(
        _data_config(data_root, tmp_path / "missing-split.yaml")
    )[0].samples
    dataset = FrontCamPolicyDataset(
        samples,
        transform=transforms.ToTensor(),
        horizontal_flip_probability=0.5,
    )

    torch.manual_seed(123)
    first = [dataset[0]["horizontal_flipped"] for _ in range(32)]
    torch.manual_seed(123)
    second = [dataset[0]["horizontal_flipped"] for _ in range(32)]
    assert first == second
    assert any(first)
    assert not all(first)


def _data_config(data_root: Path, manifest: Path) -> DataConfig:
    return DataConfig(
        root=data_root,
        split_manifest=manifest,
        require_all_matching_sessions=True,
        control_mode="gamepad",
        max_forward_speed=25.0,
        min_forward_speed=None,
        num_workers=0,
    )
