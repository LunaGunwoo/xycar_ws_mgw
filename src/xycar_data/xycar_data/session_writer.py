# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0
"""Asynchronous, crash-tolerant writer for camera-first teleop sessions."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import queue
import shutil
from threading import Event, Lock, Thread
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
import yaml


CSV_FIELDS = (
    "sample_index",
    "image",
    "angle",
    "speed",
    "input_key",
    "camera_sequence",
    "camera_stamp_sec",
    "camera_stamp_nanosec",
    "camera_received_wall_time_ns",
    "lidar_valid",
    "lidar",
    "lidar_sequence",
    "lidar_stamp_sec",
    "lidar_stamp_nanosec",
    "lidar_received_wall_time_ns",
    "lidar_skew_sec",
    "model_angle",
    "model_speed",
    "steering_axis",
    "steering_residual",
    "lt_depth",
    "rt_depth",
    "speed_delta",
    "human_correction",
    "inference_ms",
)


@dataclass(frozen=True)
class LidarSnapshot:
    sequence: int
    ranges: np.ndarray
    intensities: np.ndarray
    angle_min: float
    angle_max: float
    angle_increment: float
    time_increment: float
    scan_time: float
    range_min: float
    range_max: float
    frame_id: str
    stamp_sec: int
    stamp_nanosec: int
    received_monotonic: float
    received_wall_time_ns: int


@dataclass(frozen=True)
class CameraSample:
    image: np.ndarray
    camera_sequence: int
    camera_stamp_sec: int
    camera_stamp_nanosec: int
    camera_received_monotonic: float
    camera_received_wall_time_ns: int
    angle: float
    speed: float
    input_key: str
    lidar: Optional[LidarSnapshot]
    lidar_skew_sec: Optional[float]
    model_angle: Optional[float] = None
    model_speed: Optional[float] = None
    steering_axis: Optional[float] = None
    steering_residual: Optional[float] = None
    lt_depth: Optional[float] = None
    rt_depth: Optional[float] = None
    speed_delta: Optional[float] = None
    human_correction: Optional[bool] = None
    inference_ms: Optional[float] = None


@dataclass(frozen=True)
class SessionResult:
    token: int
    completed: bool
    path: Optional[Path]
    sample_count: int
    lidar_linked_count: int
    lidar_missing_count: int
    reason: str


@dataclass(frozen=True)
class _StartJob:
    token: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _SampleJob:
    token: int
    sample: CameraSample


@dataclass(frozen=True)
class _FinishJob:
    token: int
    reason: str
    complete: bool
    extra_metadata: Mapping[str, Any]
    final_samples: tuple[CameraSample, ...]


@dataclass(frozen=True)
class _ShutdownJob:
    pass


@dataclass
class _OpenSession:
    token: int
    metadata: Mapping[str, Any]
    started_at: datetime = field(default_factory=datetime.now)
    temp_dir: Optional[Path] = None
    images_dir: Optional[Path] = None
    lidar_dir: Optional[Path] = None
    csv_file: Any = None
    csv_writer: Any = None
    sample_count: int = 0
    lidar_linked_count: int = 0
    lidar_missing_count: int = 0
    saved_lidar_paths: dict[int, str] = field(default_factory=dict)


class AsyncSessionWriter:
    """Own disk I/O on one worker thread; callbacks only enqueue immutable data."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        png_compression: int,
        queue_size: int,
        min_free_space_mb: int,
        image_format: str = "png",
        jpeg_quality: int = 95,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.png_compression = int(png_compression)
        self.image_format = str(image_format).strip().lower()
        self.jpeg_quality = int(jpeg_quality)
        if self.image_format not in {"jpeg", "png"}:
            raise ValueError("image_format must be jpeg or png")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self.min_free_space_bytes = int(min_free_space_mb) * 1024 * 1024
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_size)
        self._results: queue.SimpleQueue[SessionResult] = queue.SimpleQueue()
        self._token = 0
        self._token_lock = Lock()
        self._failure_lock = Lock()
        self._failure: Optional[str] = None
        self._stopped = Event()
        self._thread = Thread(
            target=self._run,
            name="xycar-teleop-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def failure(self) -> Optional[str]:
        with self._failure_lock:
            return self._failure

    def start_session(self, metadata: Mapping[str, Any]) -> Optional[int]:
        if self.failure is not None:
            return None
        with self._token_lock:
            self._token += 1
            token = self._token
        if not self._enqueue(_StartJob(token, dict(metadata))):
            return None
        return token

    def submit(self, token: int, sample: CameraSample) -> bool:
        if self.failure is not None:
            return False
        return self._enqueue(_SampleJob(token, sample))

    def finish(
        self,
        token: int,
        reason: str,
        *,
        complete: bool = True,
        extra_metadata: Optional[Mapping[str, Any]] = None,
        final_samples: Sequence[CameraSample] = (),
    ) -> bool:
        if self.failure is not None:
            return False
        return self._enqueue(
            _FinishJob(
                token,
                reason,
                complete,
                dict(extra_metadata or {}),
                tuple(final_samples),
            )
        )

    def poll_results(self) -> list[SessionResult]:
        results: list[SessionResult] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def shutdown(self, timeout_sec: float = 15.0) -> bool:
        if not self._stopped.is_set():
            try:
                self._queue.put(_ShutdownJob(), timeout=1.0)
            except queue.Full:
                return False
        self._thread.join(timeout=max(0.0, timeout_sec))
        return not self._thread.is_alive()

    def _enqueue(self, job: object) -> bool:
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        current: Optional[_OpenSession] = None
        try:
            while True:
                job = self._queue.get()
                if isinstance(job, _ShutdownJob):
                    if current is not None:
                        self._finish_session(current, "process shutdown", complete=False)
                    return
                if self.failure is not None:
                    continue
                if isinstance(job, _StartJob):
                    if current is not None:
                        raise RuntimeError("attempted to start a second active session")
                    current = _OpenSession(job.token, job.metadata)
                elif isinstance(job, _SampleJob):
                    if current is None or current.token != job.token:
                        raise RuntimeError("sample does not belong to the active session")
                    self._write_sample(current, job.sample)
                elif isinstance(job, _FinishJob):
                    if current is None or current.token != job.token:
                        raise RuntimeError("finish does not belong to the active session")
                    for sample in job.final_samples:
                        self._write_sample(current, sample)
                    self._finish_session(
                        current,
                        job.reason,
                        complete=job.complete,
                        extra_metadata=job.extra_metadata,
                    )
                    current = None
                else:
                    raise RuntimeError(f"unknown writer job: {type(job).__name__}")
        except Exception as exc:
            message = f"dataset writer failure: {exc}"
            with self._failure_lock:
                self._failure = message
            if current is not None:
                try:
                    self._finish_session(current, message, complete=False)
                except Exception:
                    pass
        finally:
            self._stopped.set()

    def _write_sample(self, session: _OpenSession, sample: CameraSample) -> None:
        self._ensure_session_open(session)
        self._ensure_free_space(session.temp_dir)
        if session.temp_dir is None or session.images_dir is None:
            raise RuntimeError("session storage was not initialized")

        sample_index = session.sample_count + 1
        image_extension = ".jpg" if self.image_format == "jpeg" else ".png"
        image_relative = f"Images/{sample_index}{image_extension}"
        image_path = session.temp_dir / image_relative
        self._write_image(image_path, sample.image)

        lidar_relative = ""
        lidar = sample.lidar
        if lidar is not None:
            lidar_relative = session.saved_lidar_paths.get(lidar.sequence, "")
            if not lidar_relative:
                if session.lidar_dir is None:
                    raise RuntimeError("LiDAR storage was not initialized")
                lidar_relative = f"Lidar/{lidar.sequence:06d}.npz"
                self._write_lidar(session.temp_dir / lidar_relative, lidar)
                session.saved_lidar_paths[lidar.sequence] = lidar_relative
            session.lidar_linked_count += 1
        else:
            session.lidar_missing_count += 1

        if session.csv_writer is None or session.csv_file is None:
            raise RuntimeError("sample manifest was not initialized")
        session.csv_writer.writerow(
            {
                "sample_index": sample_index,
                "image": image_relative,
                "angle": f"{sample.angle:.6f}",
                "speed": f"{sample.speed:.6f}",
                "input_key": sample.input_key,
                "camera_sequence": sample.camera_sequence,
                "camera_stamp_sec": sample.camera_stamp_sec,
                "camera_stamp_nanosec": sample.camera_stamp_nanosec,
                "camera_received_wall_time_ns": sample.camera_received_wall_time_ns,
                "lidar_valid": str(lidar is not None).lower(),
                "lidar": lidar_relative,
                "lidar_sequence": "" if lidar is None else lidar.sequence,
                "lidar_stamp_sec": "" if lidar is None else lidar.stamp_sec,
                "lidar_stamp_nanosec": "" if lidar is None else lidar.stamp_nanosec,
                "lidar_received_wall_time_ns": ""
                if lidar is None
                else lidar.received_wall_time_ns,
                "lidar_skew_sec": ""
                if sample.lidar_skew_sec is None
                else f"{sample.lidar_skew_sec:.6f}",
                "model_angle": _optional_float(sample.model_angle),
                "model_speed": _optional_float(sample.model_speed),
                "steering_axis": _optional_float(sample.steering_axis),
                "steering_residual": _optional_float(
                    sample.steering_residual
                ),
                "lt_depth": _optional_float(sample.lt_depth),
                "rt_depth": _optional_float(sample.rt_depth),
                "speed_delta": _optional_float(sample.speed_delta),
                "human_correction": ""
                if sample.human_correction is None
                else str(sample.human_correction).lower(),
                "inference_ms": _optional_float(sample.inference_ms),
            }
        )
        session.csv_file.flush()
        session.sample_count = sample_index

    def _ensure_session_open(self, session: _OpenSession) -> None:
        if session.temp_dir is not None:
            return
        self.root_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = _unique_path(
            self.root_dir / f"_recording_{_session_timestamp(session.started_at)}"
        )
        images_dir = temp_dir / "Images"
        lidar_dir = temp_dir / "Lidar"
        images_dir.mkdir(parents=True, exist_ok=False)
        lidar_dir.mkdir(parents=True, exist_ok=False)
        csv_file = (temp_dir / "samples.csv").open("w", encoding="utf-8", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()
        csv_file.flush()
        session.temp_dir = temp_dir
        session.images_dir = images_dir
        session.lidar_dir = lidar_dir
        session.csv_file = csv_file
        session.csv_writer = csv_writer

    def _ensure_free_space(self, path: Optional[Path]) -> None:
        if path is None:
            raise RuntimeError("session storage was not initialized")
        free_bytes = shutil.disk_usage(path).free
        if free_bytes < self.min_free_space_bytes:
            minimum_mb = self.min_free_space_bytes // (1024 * 1024)
            available_mb = free_bytes // (1024 * 1024)
            raise RuntimeError(
                f"low disk space: {available_mb} MiB available, "
                f"{minimum_mb} MiB required"
            )

    def _write_image(self, path: Path, image: np.ndarray) -> None:
        temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
        if self.image_format == "jpeg":
            parameters = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        else:
            parameters = [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression]
        if not cv2.imwrite(
            str(temporary),
            image,
            parameters,
        ):
            raise RuntimeError(f"failed to write image: {path}")
        os.replace(temporary, path)

    def _write_lidar(self, path: Path, lidar: LidarSnapshot) -> None:
        temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
        np.savez_compressed(
            str(temporary),
            ranges=lidar.ranges,
            intensities=lidar.intensities,
            angle_min=np.float32(lidar.angle_min),
            angle_max=np.float32(lidar.angle_max),
            angle_increment=np.float32(lidar.angle_increment),
            time_increment=np.float32(lidar.time_increment),
            scan_time=np.float32(lidar.scan_time),
            range_min=np.float32(lidar.range_min),
            range_max=np.float32(lidar.range_max),
            frame_id=np.asarray(lidar.frame_id),
            stamp_sec=np.int64(lidar.stamp_sec),
            stamp_nanosec=np.int64(lidar.stamp_nanosec),
            received_wall_time_ns=np.int64(lidar.received_wall_time_ns),
        )
        os.replace(temporary, path)

    def _finish_session(
        self,
        session: _OpenSession,
        reason: str,
        *,
        complete: bool,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if session.csv_file is not None and not session.csv_file.closed:
            session.csv_file.close()
        if session.temp_dir is None:
            self._results.put(
                SessionResult(
                    token=session.token,
                    completed=complete,
                    path=None,
                    sample_count=0,
                    lidar_linked_count=0,
                    lidar_missing_count=0,
                    reason=reason,
                )
            )
            return

        stopped_at = datetime.now()
        metadata = {
            **dict(session.metadata),
            "format_version": 1,
            "start_time": session.started_at.isoformat(timespec="milliseconds"),
            "stop_time": stopped_at.isoformat(timespec="milliseconds"),
            "sample_count": session.sample_count,
            "lidar_linked_count": session.lidar_linked_count,
            "lidar_missing_count": session.lidar_missing_count,
            "stop_reason": reason,
            "complete": complete,
            **dict(extra_metadata or {}),
        }
        metadata_path = session.temp_dir / "metadata.yaml"
        metadata_path.write_text(
            yaml.safe_dump(metadata, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )
        state_suffix = "session" if complete else "incomplete"
        final_dir = _unique_path(
            self.root_dir
            / f"{_session_timestamp(stopped_at)}_{state_suffix}"
        )
        os.replace(session.temp_dir, final_dir)
        self._results.put(
            SessionResult(
                token=session.token,
                completed=complete,
                path=final_dir,
                sample_count=session.sample_count,
                lidar_linked_count=session.lidar_linked_count,
                lidar_missing_count=session.lidar_missing_count,
                reason=reason,
            )
        )


def _session_timestamp(value: datetime) -> str:
    return value.strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _optional_float(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def _unique_path(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(2, 10000):
        candidate = base.with_name(f"{base.name}_{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate unique dataset path from {base}")
