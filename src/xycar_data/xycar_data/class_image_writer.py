# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0
"""Asynchronously write flat class-labelled image datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import queue
import re
import shutil
from threading import Event, Lock, Thread
import time
from typing import Optional, Sequence

import cv2
import numpy as np


_CLASS_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class ClassImageSample:
    """One original camera frame assigned to exactly one class."""

    class_name: str
    image: np.ndarray
    sequence: int
    received_wall_time_ns: int


@dataclass(frozen=True)
class _WriteJob:
    sample: ClassImageSample


@dataclass(frozen=True)
class _ShutdownJob:
    pass


class AsyncClassImageWriter:
    """Write JPEG images on one bounded worker without callback disk I/O."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        class_names: Sequence[str],
        jpeg_quality: int,
        queue_size: int,
        min_free_space_mb: int,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.class_names = tuple(str(name) for name in class_names)
        self.jpeg_quality = int(jpeg_quality)
        self.min_free_space_bytes = int(min_free_space_mb) * 1024 * 1024
        self._validate(queue_size, min_free_space_mb)

        self._counts_lock = Lock()
        self._counts = {name: 0 for name in self.class_names}
        self._failure_lock = Lock()
        self._failure: Optional[str] = None
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_size)
        self._stopped = Event()

        self.root_dir.mkdir(parents=True, exist_ok=True)
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if class_dir.exists() and not class_dir.is_dir():
                raise ValueError(
                    f'class dataset path is not a directory: {class_dir}'
                )
            class_dir.mkdir(exist_ok=True)
        self._ensure_free_space(self.root_dir)

        self._thread = Thread(
            target=self._run,
            name='xycar-class-image-writer',
            daemon=True,
        )
        self._thread.start()

    @property
    def failure(self) -> Optional[str]:
        with self._failure_lock:
            return self._failure

    @property
    def counts(self) -> dict[str, int]:
        with self._counts_lock:
            return dict(self._counts)

    def submit(self, sample: ClassImageSample) -> bool:
        if self.failure is not None:
            return False
        if sample.class_name not in self._counts:
            raise ValueError(f'unknown image class: {sample.class_name}')
        if (
            not isinstance(sample.image, np.ndarray)
            or sample.image.dtype != np.uint8
            or sample.image.ndim != 3
            or sample.image.shape[2] != 3
        ):
            raise ValueError(
                'class image must be uint8 BGR with three channels'
            )
        if sample.sequence < 1 or sample.received_wall_time_ns < 0:
            raise ValueError(
                'image sequence must be positive and wall time non-negative'
            )
        try:
            self._queue.put_nowait(_WriteJob(sample))
            return True
        except queue.Full:
            return False

    def shutdown(self, timeout_sec: float = 15.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_sec)
        if not self._stopped.is_set():
            try:
                self._queue.put(
                    _ShutdownJob(),
                    timeout=max(0.0, deadline - time.monotonic()),
                )
            except queue.Full:
                return False
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return not self._thread.is_alive()

    def _validate(self, queue_size: int, min_free_space_mb: int) -> None:
        if not self.class_names:
            raise ValueError('at least one image class is required')
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError('image class names must be distinct')
        for class_name in self.class_names:
            if _CLASS_NAME_PATTERN.fullmatch(class_name) is None:
                raise ValueError(
                    'image class names must contain lowercase letters, '
                    f'digits, or underscores: {class_name!r}'
                )
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in [1, 100]')
        if queue_size < 1:
            raise ValueError('queue_size must be positive')
        if min_free_space_mb < 0:
            raise ValueError('min_free_space_mb must be non-negative')

    def _run(self) -> None:
        try:
            while True:
                job = self._queue.get()
                if isinstance(job, _ShutdownJob):
                    return
                if not isinstance(job, _WriteJob):
                    raise RuntimeError(
                        f'unknown writer job: {type(job).__name__}'
                    )
                self._write(job.sample)
        except Exception as exc:
            with self._failure_lock:
                self._failure = f'class image writer failure: {exc}'
        finally:
            self._stopped.set()

    def _write(self, sample: ClassImageSample) -> None:
        class_dir = self.root_dir / sample.class_name
        self._ensure_free_space(class_dir)
        destination = _unique_image_path(class_dir, sample)
        temporary = destination.with_name(
            f'.{destination.stem}.{os.getpid()}.tmp.jpg'
        )
        try:
            if not cv2.imwrite(
                str(temporary),
                sample.image,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            ):
                raise RuntimeError(f'failed to write image: {destination}')
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        with self._counts_lock:
            self._counts[sample.class_name] += 1

    def _ensure_free_space(self, path: Path) -> None:
        available = shutil.disk_usage(path).free
        if available < self.min_free_space_bytes:
            available_mb = math.floor(available / (1024 * 1024))
            minimum_mb = math.ceil(
                self.min_free_space_bytes / (1024 * 1024)
            )
            raise RuntimeError(
                f'low disk space: {available_mb} MiB available, '
                f'{minimum_mb} MiB required'
            )


def _unique_image_path(
    class_dir: Path,
    sample: ClassImageSample,
) -> Path:
    seconds, nanoseconds = divmod(sample.received_wall_time_ns, 1_000_000_000)
    timestamp = datetime.fromtimestamp(seconds).strftime('%Y%m%d_%H%M%S')
    stem = f'{timestamp}_{nanoseconds:09d}_{sample.sequence:09d}'
    candidate = class_dir / f'{stem}.jpg'
    suffix = 2
    while candidate.exists():
        candidate = class_dir / f'{stem}_{suffix}.jpg'
        suffix += 1
    return candidate
