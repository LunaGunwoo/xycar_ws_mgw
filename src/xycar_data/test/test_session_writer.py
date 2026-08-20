# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import csv
from datetime import datetime
import re
import time

import cv2
import numpy as np
import yaml

from xycar_data.class_image_writer import (
    AsyncClassImageWriter,
    ClassImageSample,
)
from xycar_data.session_writer import (
    AsyncSessionWriter,
    CameraSample,
    _unique_path,
)


def _sample(index: int) -> CameraSample:
    return CameraSample(
        image=np.full((4, 6, 3), index, dtype=np.uint8),
        camera_sequence=index,
        camera_stamp_sec=100 + index,
        camera_stamp_nanosec=index,
        camera_received_monotonic=float(index),
        camera_received_wall_time_ns=1000 + index,
        angle=float(index),
        speed=3.5,
        input_key='gamepad',
        lidar=None,
        lidar_skew_sec=None,
    )


def _wait_for_result(writer: AsyncSessionWriter):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        results = writer.poll_results()
        if results:
            return results[0]
        time.sleep(0.01)
    raise AssertionError('timed out waiting for dataset writer')


def _writer(root, *, image_format='png', jpeg_quality=95):
    return AsyncSessionWriter(
        root,
        png_compression=0,
        queue_size=64,
        min_free_space_mb=0,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
    )


def _wait_for_class_images(writer, expected_count):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if sum(writer.counts.values()) == expected_count:
            return
        if writer.failure is not None:
            raise AssertionError(writer.failure)
        time.sleep(0.01)
    raise AssertionError('timed out waiting for class images')


def test_finish_writes_buffered_final_samples_in_order(tmp_path):
    writer = _writer(tmp_path / 'teleop')
    try:
        token = writer.start_session({'control_mode': 'gamepad'})
        assert token is not None
        for index in range(1, 6):
            assert writer.submit(token, _sample(index))
        assert writer.finish(
            token,
            'b_button',
            final_samples=tuple(_sample(index) for index in range(6, 21)),
            extra_metadata={'emergency_discard_count': 0},
        )

        result = _wait_for_result(writer)

        assert result.completed
        assert result.sample_count == 20
        assert result.reason == 'b_button'
        assert result.path is not None
        assert re.fullmatch(
            r'\d{8}_\d{6}_\d{3}_session',
            result.path.name,
        )
        images = sorted((result.path / 'Images').glob('*.png'))
        assert len(images) == 20
        assert {path.name for path in images} == {
            f'{index}.png' for index in range(1, 21)
        }
        with (result.path / 'samples.csv').open(
            encoding='utf-8',
            newline='',
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert [int(row['camera_sequence']) for row in rows] == list(
            range(1, 21)
        )
        assert rows[-1]['angle'] == '20.000000'
    finally:
        writer.shutdown()


def test_jpeg_session_writes_jpg_paths_that_opencv_can_read(tmp_path):
    writer = _writer(
        tmp_path / 'teleop',
        image_format='jpeg',
        jpeg_quality=95,
    )
    try:
        token = writer.start_session({'control_mode': 'gamepad'})
        assert token is not None
        assert writer.submit(token, _sample(1))
        assert writer.finish(token, 'b_button')

        result = _wait_for_result(writer)

        assert result.completed
        assert result.path is not None
        image_path = result.path / 'Images' / '1.jpg'
        assert image_path.is_file()
        assert cv2.imread(str(image_path)) is not None
        with (result.path / 'samples.csv').open(
            encoding='utf-8',
            newline='',
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert [row['image'] for row in rows] == ['Images/1.jpg']
    finally:
        writer.shutdown()


def test_emergency_finish_keeps_prefix_and_records_discard_count(tmp_path):
    writer = _writer(tmp_path / 'teleop')
    try:
        token = writer.start_session({'control_mode': 'gamepad'})
        assert token is not None
        for index in range(1, 6):
            assert writer.submit(token, _sample(index))
        assert writer.finish(
            token,
            'speed_nonpositive',
            extra_metadata={
                'emergency_discard_count': 15,
                'emergency_discard_frames': 15,
            },
        )

        result = _wait_for_result(writer)

        assert result.completed
        assert result.sample_count == 5
        assert result.path is not None
        metadata = yaml.safe_load(
            (result.path / 'metadata.yaml').read_text(encoding='utf-8')
        )
        assert metadata['stop_reason'] == 'speed_nonpositive'
        assert metadata['emergency_discard_count'] == 15
        assert len(list((result.path / 'Images').glob('*.png'))) == 5
    finally:
        writer.shutdown()


def test_empty_finished_session_creates_no_directory(tmp_path):
    root = tmp_path / 'teleop'
    writer = _writer(root)
    try:
        token = writer.start_session({'control_mode': 'gamepad'})
        assert token is not None
        assert writer.finish(
            token,
            'speed_nonpositive',
            extra_metadata={'emergency_discard_count': 10},
        )

        result = _wait_for_result(writer)

        assert result.completed
        assert result.path is None
        assert result.sample_count == 0
        assert not root.exists()
    finally:
        writer.shutdown()


def test_discard_deletes_entire_active_session(tmp_path):
    root = tmp_path / 'guided'
    writer = _writer(root)
    try:
        token = writer.start_session({'control_mode': 'guided_policy'})
        assert token is not None
        for index in range(1, 6):
            assert writer.submit(token, _sample(index))
        assert writer.discard(token, 'x_button')

        result = _wait_for_result(writer)

        assert not result.completed
        assert result.discarded
        assert result.path is None
        assert result.sample_count == 5
        assert not list(root.glob('_recording_*'))
        assert not list(root.glob('*_session*'))
        assert not list(root.glob('*_incomplete*'))
    finally:
        writer.shutdown()


def test_unique_path_adds_suffix_for_same_timestamp(tmp_path):
    base = tmp_path / (
        datetime(2026, 7, 23, 12, 34, 56, 789000).strftime(
            '%Y%m%d_%H%M%S_%f'
        )[:-3]
        + '_session'
    )
    base.mkdir()
    second = base.with_name(f'{base.name}_2')
    second.mkdir()

    assert _unique_path(base) == base.with_name(f'{base.name}_3')


def test_class_image_writer_keeps_flat_jpeg_class_directories(tmp_path):
    root = tmp_path / 'traffic_signal_images'
    classes = ('red', 'yellow', 'straight_green', 'left_green')
    writer = AsyncClassImageWriter(
        root,
        class_names=classes,
        jpeg_quality=95,
        queue_size=16,
        min_free_space_mb=0,
    )
    try:
        for sequence, class_name in enumerate(classes, start=1):
            assert writer.submit(
                ClassImageSample(
                    class_name=class_name,
                    image=np.full(
                        (8, 12, 3),
                        sequence * 20,
                        dtype=np.uint8,
                    ),
                    sequence=sequence,
                    received_wall_time_ns=1_800_000_000_000_000_000,
                )
            )
        _wait_for_class_images(writer, len(classes))
    finally:
        assert writer.shutdown()

    assert {path.name for path in root.iterdir()} == set(classes)
    for class_name in classes:
        entries = list((root / class_name).iterdir())
        assert len(entries) == 1
        assert entries[0].suffix == '.jpg'
        assert cv2.imread(str(entries[0])).shape == (8, 12, 3)


def test_class_image_writer_never_overwrites_a_colliding_filename(tmp_path):
    root = tmp_path / 'traffic_signal_images'
    writer = AsyncClassImageWriter(
        root,
        class_names=('red',),
        jpeg_quality=95,
        queue_size=4,
        min_free_space_mb=0,
    )
    sample = ClassImageSample(
        class_name='red',
        image=np.zeros((4, 6, 3), dtype=np.uint8),
        sequence=1,
        received_wall_time_ns=1_800_000_000_000_000_000,
    )
    try:
        assert writer.submit(sample)
        assert writer.submit(sample)
        _wait_for_class_images(writer, 2)
    finally:
        assert writer.shutdown()

    images = sorted((root / 'red').glob('*.jpg'))
    assert len(images) == 2
    assert images[0].name != images[1].name
