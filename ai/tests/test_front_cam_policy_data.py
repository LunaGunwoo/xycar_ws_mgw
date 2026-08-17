from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
import yaml
from conftest import write_session, write_split_manifest
from PIL import Image
from torchvision import transforms

from xycar_ai.config import DataConfig, DataSourceConfig
from xycar_ai.front_cam_policy_data import (
    FrontCamPolicyDataset,
    PolicyDatasetError,
    PolicySample,
    attach_executed_command_history,
    attach_training_teacher_forced_history,
    build_policy_data_splits,
    command_class_id,
    compute_sqrt_inverse_frequency_weights,
    discover_policy_sessions,
    generation_epoch_sample_count,
    generation_sampling_weights,
    policy_dataset_stats,
    quantize_command,
    smooth_training_angle_targets,
    validate_session_initial_classes,
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


def test_multi_source_split_qualifies_colliding_names_and_generation_contracts(
    tmp_path: Path,
):
    manual_root = tmp_path / "datasets" / "stateless_manual"
    guided_root = tmp_path / "datasets" / "stateless_guided"
    shared_name = "20260810_130735_027_session"
    val_name = "20260810_130818_255_session"
    test_name = "20260810_130857_726_session"
    write_session(
        manual_root,
        shared_name,
        labels=[(1.0, 7.0)],
        generation=9,
    )
    write_session(
        guided_root,
        shared_name,
        labels=[(2.0, 9.0)],
        control_mode="guided_policy",
        generation=1,
    )
    write_session(
        manual_root,
        val_name,
        labels=[(3.0, 7.0)],
    )
    write_session(
        guided_root,
        test_name,
        labels=[(4.0, 9.0)],
        control_mode="guided_policy",
        generation=1,
    )
    manifest = write_split_manifest(
        tmp_path / "config" / "split.yaml",
        train=[f"manual/{shared_name}", f"guided/{shared_name}"],
        val=[f"manual/{val_name}"],
        test=[f"guided/{test_name}"],
        schema_version=2,
    )
    config = _multi_source_data_config(manual_root, guided_root, manifest)

    splits = build_policy_data_splits(config)

    assert [sample.session_id for sample in splits.train_samples] == [
        f"manual/{shared_name}",
        f"guided/{shared_name}",
    ]
    assert [sample.source_id for sample in splits.train_samples] == [
        "manual",
        "guided",
    ]
    assert [sample.generation for sample in splits.train_samples] == [0, 1]
    assert splits.manifest()["schema_version"] == 2
    assert splits.train_samples[0].relative_image.startswith("manual/")
    assert splits.train_samples[1].relative_image.startswith("guided/")


def test_multi_source_guided_requires_generation_and_source_manifest_schema(
    tmp_path: Path,
):
    manual_root = tmp_path / "datasets" / "stateless_manual"
    guided_root = tmp_path / "datasets" / "stateless_guided"
    manual_name = "20260810_130735_027_session"
    guided_name = "20260810_130818_255_session"
    test_name = "20260810_130857_726_session"
    write_session(manual_root, manual_name, labels=[(0.0, 7.0)])
    write_session(
        guided_root,
        guided_name,
        labels=[(0.0, 9.0)],
        control_mode="guided_policy",
    )
    write_session(manual_root, test_name, labels=[(0.0, 7.0)])
    manifest = write_split_manifest(
        tmp_path / "config" / "split.yaml",
        train=[f"manual/{manual_name}"],
        val=[f"guided/{guided_name}"],
        test=[f"manual/{test_name}"],
        schema_version=2,
    )
    config = _multi_source_data_config(manual_root, guided_root, manifest)

    with pytest.raises(PolicyDatasetError, match="curriculum.generation is required"):
        build_policy_data_splits(config)

    metadata_path = guided_root / guided_name / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["curriculum"] = {"generation": 1}
    metadata_path.write_text(
        yaml.safe_dump(metadata, sort_keys=True), encoding="utf-8"
    )
    write_split_manifest(
        manifest,
        train=[f"manual/{manual_name}"],
        val=[f"guided/{guided_name}"],
        test=[f"manual/{test_name}"],
        schema_version=1,
    )
    with pytest.raises(PolicyDatasetError, match="schema_version must be 2"):
        build_policy_data_splits(config)

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


def test_training_angle_mean_uses_session_edge_padding_and_preserves_other_targets(
    tmp_path: Path,
):
    data_root = tmp_path / "datasets" / "teleop"
    names = [
        "20260810_130735_027_session",
        "20260810_130818_255_session",
        "20260810_130857_726_session",
        "20260810_130938_647_session",
    ]
    for name in names[:3]:
        write_session(
            data_root,
            name,
            labels=[(10.0, 21.0), (20.0, 22.0), (30.0, 23.0)],
        )
    write_session(data_root, names[3], labels=[(100.0, 24.0), (100.0, 25.0)])
    manifest = write_split_manifest(
        tmp_path / "config" / "split.yaml",
        train=[names[0], names[3]],
        val=[names[1]],
        test=[names[2]],
    )
    splits = build_policy_data_splits(_data_config(data_root, manifest))

    smoothed = smooth_training_angle_targets(splits, 5)

    first_train = smoothed.train_sessions[0].samples
    second_train = smoothed.train_sessions[1].samples
    assert [sample.angle_raw for sample in first_train] == pytest.approx(
        [16.0, 20.0, 24.0]
    )
    assert [sample.angle for sample in first_train] == [16, 20, 24]
    assert [sample.angle_class_id for sample in first_train] == [116, 120, 124]
    assert [sample.speed_raw for sample in first_train] == [21.0, 22.0, 23.0]
    assert [sample.angle_raw for sample in second_train] == [100.0, 100.0]
    assert smoothed.val_samples == splits.val_samples
    assert smoothed.test_samples == splits.test_samples
    assert smooth_training_angle_targets(splits, 1) is splits


def test_teacher_forced_history_uses_smoothed_angle_raw_speed_and_session_padding(
    tmp_path: Path,
):
    data_root = tmp_path / "datasets" / "teleop"
    names = [
        "20260810_130735_027_session",
        "20260810_130818_255_session",
        "20260810_130857_726_session",
        "20260810_130938_647_session",
    ]
    write_session(
        data_root,
        names[0],
        labels=[(0.0, 25.0), (10.0, 24.0), (20.0, 23.0), (30.0, 22.0)],
    )
    write_session(
        data_root,
        names[1],
        labels=[(0.0, 25.0), (-10.0, 24.0)],
    )
    for name in names[2:]:
        write_session(data_root, name, labels=[(0.0, 25.0), (80.0, 20.0)])
    manifest = write_split_manifest(
        tmp_path / "config" / "split.yaml",
        train=names[:2],
        val=[names[2]],
        test=[names[3]],
    )
    splits = build_policy_data_splits(_data_config(data_root, manifest))
    validate_session_initial_classes(
        splits,
        angle_class_id=100,
        speed_class_id=125,
    )

    with_history = attach_training_teacher_forced_history(
        smooth_training_angle_targets(splits, 5),
        4,
    )
    first = with_history.train_sessions[0].samples
    first_pair = (first[0].angle_class_id, first[0].speed_class_id)
    assert first[0].history_class_ids == (first_pair,) * 4
    assert first[2].history_class_ids == (
        first_pair,
        first_pair,
        first_pair,
        (first[1].angle_class_id, first[1].speed_class_id),
    )
    assert first[1].history_class_ids[-1][0] == first[0].angle_class_id
    assert first[1].history_class_ids[-1][1] == 125
    second_first = with_history.train_sessions[1].samples[0]
    assert (
        second_first.history_class_ids
        == ((second_first.angle_class_id, second_first.speed_class_id),) * 4
    )
    assert all(sample.history_class_ids is None for sample in with_history.val_samples)
    assert all(sample.history_class_ids is None for sample in with_history.test_samples)

    flipped_dataset = FrontCamPolicyDataset(
        [first[2]],
        transform=transforms.ToTensor(),
        horizontal_flip_probability=1.0,
    )
    flipped = flipped_dataset[0]
    expected_history = torch.tensor(first[2].history_class_ids, dtype=torch.long)
    expected_history[:, 0] = 200 - expected_history[:, 0]
    assert torch.equal(flipped["history_class_ids"], expected_history)
    assert flipped["angle_class_id"] == 200 - first[2].angle_class_id
    assert flipped["speed_class_id"] == first[2].speed_class_id


def test_external_history_uses_session_seed_and_raw_executed_commands(
    tmp_path: Path,
):
    data_root = tmp_path / "datasets" / "teleop"
    names = [
        "20260810_130735_027_session",
        "20260810_130818_255_session",
        "20260810_130857_726_session",
    ]
    seed = [[90, 121], [91, 122], [92, 123], [93, 124]]
    for name in names:
        write_session(
            data_root,
            name,
            labels=[(10.0, 7.0), (20.0, 6.0)],
            control_mode="guided_policy",
            generation=1,
            initial_history_class_ids=seed,
        )
    manifest = write_split_manifest(
        tmp_path / "config" / "split.yaml",
        train=[names[0]],
        val=[names[1]],
        test=[names[2]],
    )
    config = DataConfig(
        root=data_root,
        split_manifest=manifest,
        require_all_matching_sessions=True,
        control_mode="",
        control_modes=("guided_policy",),
        max_forward_speed=None,
        min_forward_speed=None,
        num_workers=0,
        current_generation=1,
        generation_decay=0.5,
        ema_sampling=True,
    )

    attached = attach_executed_command_history(
        build_policy_data_splits(config),
        history_frames=4,
    )
    for samples in (
        attached.train_samples,
        attached.val_samples,
        attached.test_samples,
    ):
        assert samples[0].history_class_ids == tuple(map(tuple, seed))
        assert samples[1].history_class_ids == (
            (91, 122),
            (92, 123),
            (93, 124),
            (110, 107),
        )


def test_generation_weights_assign_ema_mass_uniformly_within_generation():
    samples = [
        _policy_sample(generation=0, angle_class_id=100),
        _policy_sample(generation=0, angle_class_id=101),
        _policy_sample(generation=1, angle_class_id=102),
        _policy_sample(generation=1, angle_class_id=103),
        _policy_sample(generation=1, angle_class_id=104),
        _policy_sample(generation=1, angle_class_id=105),
    ]

    weights = generation_sampling_weights(
        samples,
        current_generation=1,
        generation_decay=0.5,
    )

    assert sum(weights[:2]) == pytest.approx(0.5)
    assert sum(weights[2:]) == pytest.approx(1.0)
    assert weights[:2] == pytest.approx((0.25, 0.25))
    assert weights[2:] == pytest.approx((0.25, 0.25, 0.25, 0.25))
    assert (
        generation_epoch_sample_count(
            samples,
            current_generation=1,
            generation_decay=0.5,
        )
        == 6
    )


def test_generation_weights_require_current_training_samples():
    samples = [_policy_sample(generation=0, angle_class_id=100)]

    with pytest.raises(PolicyDatasetError, match="no samples for current_generation=1"):
        generation_sampling_weights(
            samples,
            current_generation=1,
            generation_decay=0.5,
        )


def test_initial_class_validation_rejects_a_session_mismatch(tmp_path: Path):
    data_root = tmp_path / "datasets" / "teleop"
    names = [
        "20260810_130735_027_session",
        "20260810_130818_255_session",
        "20260810_130857_726_session",
    ]
    write_session(data_root, names[0], labels=[(0.0, 25.0)])
    write_session(data_root, names[1], labels=[(1.0, 25.0)])
    write_session(data_root, names[2], labels=[(0.0, 25.0)])
    manifest = write_split_manifest(
        tmp_path / "config" / "split.yaml",
        train=[names[0]],
        val=[names[1]],
        test=[names[2]],
    )
    splits = build_policy_data_splits(_data_config(data_root, manifest))
    with pytest.raises(PolicyDatasetError, match=names[1]):
        validate_session_initial_classes(
            splits,
            angle_class_id=100,
            speed_class_id=125,
        )


@pytest.mark.parametrize("window_size", [0, -1, 2, 4])
def test_training_angle_mean_rejects_non_positive_or_even_window(
    tmp_path: Path,
    window_size: int,
):
    data_root = tmp_path / "datasets" / "teleop"
    names = [
        "20260810_130735_027_session",
        "20260810_130818_255_session",
        "20260810_130857_726_session",
    ]
    for name in names:
        write_session(data_root, name, labels=[(10.0, 25.0)])
    manifest = write_split_manifest(
        tmp_path / "config" / "split.yaml",
        train=[names[0]],
        val=[names[1]],
        test=[names[2]],
    )
    splits = build_policy_data_splits(_data_config(data_root, manifest))

    with pytest.raises(ValueError, match="positive odd integer"):
        smooth_training_angle_targets(splits, window_size)


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


def _multi_source_data_config(
    manual_root: Path, guided_root: Path, manifest: Path
) -> DataConfig:
    return DataConfig(
        root=None,
        split_manifest=manifest,
        require_all_matching_sessions=True,
        control_mode="",
        max_forward_speed=None,
        min_forward_speed=None,
        num_workers=0,
        current_generation=1,
        generation_decay=0.5,
        ema_sampling=True,
        sources=(
            DataSourceConfig(
                source_id="manual",
                root=manual_root,
                control_modes=("gamepad",),
                fixed_generation=0,
            ),
            DataSourceConfig(
                source_id="guided",
                root=guided_root,
                control_modes=("guided_policy",),
                require_curriculum_generation=True,
            ),
        ),
    )


def _policy_sample(*, generation: int, angle_class_id: int) -> PolicySample:
    angle = angle_class_id - 100
    return PolicySample(
        session_id=f"generation-{generation}",
        image_path=Path("unused.png"),
        relative_image="unused.png",
        angle_raw=float(angle),
        speed_raw=7.0,
        angle=angle,
        speed=7,
        angle_class_id=angle_class_id,
        speed_class_id=107,
        generation=generation,
    )
