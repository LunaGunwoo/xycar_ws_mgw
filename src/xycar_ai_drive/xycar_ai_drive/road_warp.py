"""Shared road-warp geometry and tuning-config helpers."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from xycar_ai_drive.artifact import RoadWarpParameters

ROAD_WARP_SCHEMA_VERSION = 1
ROAD_WARP_PARAMETER_NAMES = (
    'top_y',
    'bottom_y',
    'top_left_x',
    'top_right_x',
    'bottom_left_x',
    'bottom_right_x',
    'bev_width',
    'bev_height',
    'dst_left_x',
    'dst_right_x',
)
ROAD_WARP_RATIO_NAMES = (
    'top_y',
    'bottom_y',
    'top_left_x',
    'top_right_x',
    'bottom_left_x',
    'bottom_right_x',
    'dst_left_x',
    'dst_right_x',
)


def road_warp_from_mapping(
    values: Mapping[str, object],
) -> RoadWarpParameters:
    """Validate a strict parameter mapping and return runtime parameters."""
    if set(values) != set(ROAD_WARP_PARAMETER_NAMES):
        raise ValueError('warp parameter keys are incompatible')
    normalized: dict[str, float | int] = {}
    for name in ROAD_WARP_RATIO_NAMES:
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f'warp.{name} must be a finite number')
        normalized[name] = float(value)
    for name in ('bev_width', 'bev_height'):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'warp.{name} must be an integer')
        normalized[name] = value

    result = RoadWarpParameters(**normalized)
    ratios = tuple(float(getattr(result, name)) for name in ROAD_WARP_RATIO_NAMES)
    if not all(0.0 <= value <= 1.0 for value in ratios):
        raise ValueError('warp ratios must be in [0,1]')
    if result.bottom_y - result.top_y < 0.02:
        raise ValueError('warp.bottom_y must be at least 0.02 below top_y')
    if result.top_right_x - result.top_left_x < 0.02:
        raise ValueError('warp top edge must be at least 0.02 wide')
    if result.bottom_right_x - result.bottom_left_x < 0.02:
        raise ValueError('warp bottom edge must be at least 0.02 wide')
    if not 80 <= result.bev_width <= 1920:
        raise ValueError('warp.bev_width must be in [80, 1920]')
    if not 60 <= result.bev_height <= 1080:
        raise ValueError('warp.bev_height must be in [60, 1080]')
    if not 0.0 <= result.dst_left_x <= 0.49:
        raise ValueError('warp.dst_left_x must be in [0, 0.49]')
    if not 0.51 <= result.dst_right_x <= 1.0:
        raise ValueError('warp.dst_right_x must be in [0.51, 1]')
    return result


def road_warp_values(
    config: RoadWarpParameters,
) -> dict[str, float | int]:
    """Return a stable YAML-compatible mapping for a validated config."""
    validated = road_warp_from_mapping(asdict(config))
    return {
        name: getattr(validated, name)
        for name in ROAD_WARP_PARAMETER_NAMES
    }


def load_road_warp_config(path: str | Path) -> RoadWarpParameters:
    """Load the standalone training warp YAML schema."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'road warp config is missing: {config_path}')
    try:
        payload = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    except yaml.YAMLError as exc:
        raise ValueError(f'invalid road warp YAML: {config_path}') from exc
    if not isinstance(payload, Mapping):
        raise ValueError('road warp YAML root must be a mapping')
    if set(payload) != {'schema_version', 'warp'}:
        raise ValueError('road warp YAML keys are incompatible')
    if payload.get('schema_version') != ROAD_WARP_SCHEMA_VERSION:
        raise ValueError('unsupported road warp schema_version')
    warp = payload.get('warp')
    if not isinstance(warp, Mapping):
        raise ValueError('warp must be a mapping')
    return road_warp_from_mapping(warp)


def save_road_warp_config(
    path: str | Path,
    config: RoadWarpParameters,
) -> Path:
    """Atomically save a config without following an output-file symlink."""
    output_path = Path(path).expanduser()
    if output_path.is_symlink():
        raise ValueError(f'refusing to replace config symlink: {output_path}')
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f'.{output_path.name}.tmp')
    payload = {
        'schema_version': ROAD_WARP_SCHEMA_VERSION,
        'warp': road_warp_values(config),
    }
    try:
        temporary.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding='utf-8',
        )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def warp_road_image(
    image: np.ndarray,
    config: RoadWarpParameters,
) -> np.ndarray:
    """Apply the same bilinear perspective warp used by policy runtime."""
    _validate_image(image)
    config = road_warp_from_mapping(asdict(config))
    height, width = image.shape[:2]
    transform = cv2.getPerspectiveTransform(
        source_points(config, width=width, height=height),
        destination_points(config),
    )
    return cv2.warpPerspective(
        image,
        transform,
        (config.bev_width, config.bev_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def draw_road_warp_overlay(
    image: np.ndarray,
    config: RoadWarpParameters,
) -> np.ndarray:
    """Draw the normalized source trapezoid on a BGR preview image."""
    _validate_image(image)
    config = road_warp_from_mapping(asdict(config))
    result = image.copy()
    height, width = result.shape[:2]
    points = np.rint(
        source_points(config, width=width, height=height)
    ).astype(np.int32)
    cv2.polylines(result, [points], True, (0, 255, 255), 3, cv2.LINE_AA)
    for point, label in zip(
        points,
        ('BL', 'TL', 'TR', 'BR'),
        strict=True,
    ):
        location = tuple(int(value) for value in point)
        cv2.circle(result, location, 7, (0, 120, 255), -1, cv2.LINE_AA)
        cv2.putText(
            result,
            label,
            (location[0] + 8, location[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return result


def source_points(
    config: RoadWarpParameters,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    if width < 2 or height < 2:
        raise ValueError('road warp requires an image of at least 2x2')
    max_x = float(width - 1)
    max_y = float(height - 1)
    return np.asarray(
        [
            [config.bottom_left_x * max_x, config.bottom_y * max_y],
            [config.top_left_x * max_x, config.top_y * max_y],
            [config.top_right_x * max_x, config.top_y * max_y],
            [config.bottom_right_x * max_x, config.bottom_y * max_y],
        ],
        dtype=np.float32,
    )


def destination_points(config: RoadWarpParameters) -> np.ndarray:
    max_x = float(config.bev_width - 1)
    max_y = float(config.bev_height - 1)
    left = config.dst_left_x * max_x
    right = config.dst_right_x * max_x
    return np.asarray(
        [
            [left, max_y],
            [left, 0.0],
            [right, 0.0],
            [right, max_y],
        ],
        dtype=np.float32,
    )


def _validate_image(image: np.ndarray) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] < 2
        or image.shape[1] < 2
    ):
        raise ValueError('warp input must be a uint8 HxWx3 image of at least 2x2')
