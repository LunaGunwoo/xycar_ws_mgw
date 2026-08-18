from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

ROAD_WARP_SCHEMA_VERSION = 1
ROAD_WARP_GEOMETRY = "perspective_road_warp_then_bicubic_resize"


@dataclass(frozen=True)
class RoadWarpConfig:
    top_y: float = 0.5
    bottom_y: float = 0.933
    top_left_x: float = 0.34
    top_right_x: float = 0.66
    bottom_left_x: float = 0.0
    bottom_right_x: float = 1.0
    bev_width: int = 224
    bev_height: int = 224
    dst_left_x: float = 0.0
    dst_right_x: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "top_y",
            "bottom_y",
            "top_left_x",
            "top_right_x",
            "bottom_left_x",
            "bottom_right_x",
            "dst_left_x",
            "dst_right_x",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"warp.{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"warp.{name} must be finite")
        for name in ("bev_width", "bev_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"warp.{name} must be an integer")

        ratios = (
            "top_y",
            "bottom_y",
            "top_left_x",
            "top_right_x",
            "bottom_left_x",
            "bottom_right_x",
            "dst_left_x",
            "dst_right_x",
        )
        for name in ratios:
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"warp.{name} must be in [0, 1]")
        if self.bottom_y - self.top_y < 0.02:
            raise ValueError("warp.bottom_y must be at least 0.02 below top_y")
        if self.top_right_x - self.top_left_x < 0.02:
            raise ValueError("warp top edge must be at least 0.02 wide")
        if self.bottom_right_x - self.bottom_left_x < 0.02:
            raise ValueError("warp bottom edge must be at least 0.02 wide")
        if self.bev_width < 80 or self.bev_width > 1920:
            raise ValueError("warp.bev_width must be in [80, 1920]")
        if self.bev_height < 60 or self.bev_height > 1080:
            raise ValueError("warp.bev_height must be in [60, 1080]")
        if not 0.0 <= self.dst_left_x <= 0.49:
            raise ValueError("warp.dst_left_x must be in [0, 0.49]")
        if not 0.51 <= self.dst_right_x <= 1.0:
            raise ValueError("warp.dst_right_x must be in [0.51, 1]")

    def serializable(self) -> dict[str, float | int]:
        return asdict(self)

    def source_points(self, width: int, height: int) -> np.ndarray:
        _validate_image_size(width, height)
        max_x = float(width - 1)
        max_y = float(height - 1)
        return np.asarray(
            [
                [self.bottom_left_x * max_x, self.bottom_y * max_y],
                [self.top_left_x * max_x, self.top_y * max_y],
                [self.top_right_x * max_x, self.top_y * max_y],
                [self.bottom_right_x * max_x, self.bottom_y * max_y],
            ],
            dtype=np.float32,
        )

    def destination_points(self) -> np.ndarray:
        max_x = float(self.bev_width - 1)
        max_y = float(self.bev_height - 1)
        left = self.dst_left_x * max_x
        right = self.dst_right_x * max_x
        return np.asarray(
            [[left, max_y], [left, 0.0], [right, 0.0], [right, max_y]],
            dtype=np.float32,
        )

    def contract(self, *, source_path: Path | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": ROAD_WARP_SCHEMA_VERSION,
            "parameters": self.serializable(),
            "sha256": road_warp_sha256(self),
            "source_point_order": [
                "bottom_left",
                "top_left",
                "top_right",
                "bottom_right",
            ],
            "coordinate_space": "normalized_input_frame",
            "interpolation": "bilinear",
        }
        if source_path is not None:
            payload["config_path"] = str(source_path)
        return payload


def road_warp_payload(config: RoadWarpConfig) -> dict[str, object]:
    return {
        "schema_version": ROAD_WARP_SCHEMA_VERSION,
        "warp": config.serializable(),
    }


def load_road_warp_config(path: str | Path) -> RoadWarpConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"road warp config does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid road warp YAML: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("road warp YAML root must be a mapping")
    expected_root = {"schema_version", "warp"}
    if set(payload) != expected_root:
        raise ValueError(
            "road warp config keys mismatch; "
            f"expected={sorted(expected_root)}, actual={sorted(payload)}"
        )
    if payload.get("schema_version") != ROAD_WARP_SCHEMA_VERSION:
        raise ValueError("unsupported road warp schema_version")
    warp_payload = payload.get("warp")
    if not isinstance(warp_payload, Mapping):
        raise TypeError("warp must be a mapping")
    expected_fields = {field.name for field in fields(RoadWarpConfig)}
    if set(warp_payload) != expected_fields:
        raise ValueError(
            "warp keys mismatch; "
            f"expected={sorted(expected_fields)}, actual={sorted(warp_payload)}"
        )
    return RoadWarpConfig(**dict(warp_payload))


def save_road_warp_config(path: str | Path, config: RoadWarpConfig) -> None:
    config_path = Path(path).expanduser().resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config_path.with_name(f".{config_path.name}.tmp")
    temporary.write_text(
        yaml.safe_dump(
            road_warp_payload(config),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, config_path)


def road_warp_sha256(config: RoadWarpConfig) -> str:
    canonical = json.dumps(
        road_warp_payload(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def warp_image_array(image: np.ndarray, config: RoadWarpConfig) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] < 2
        or image.shape[1] < 2
    ):
        raise ValueError("warp input must be a non-empty uint8 HxWx3 image")
    height, width = image.shape[:2]
    transform = cv2.getPerspectiveTransform(
        config.source_points(width, height),
        config.destination_points(),
    )
    return cv2.warpPerspective(
        image,
        transform,
        (config.bev_width, config.bev_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def warp_pil_image(image: Image.Image, config: RoadWarpConfig) -> Image.Image:
    rgb = image.convert("RGB")
    warped = warp_image_array(np.asarray(rgb, dtype=np.uint8), config)
    return Image.fromarray(warped)


def draw_warp_overlay(
    image: np.ndarray,
    config: RoadWarpConfig,
) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError("overlay input must be a uint8 HxWx3 image")
    result = image.copy()
    height, width = result.shape[:2]
    points = np.rint(config.source_points(width, height)).astype(np.int32)
    cv2.polylines(result, [points], True, (0, 255, 255), 3, cv2.LINE_AA)
    labels = ("BL", "TL", "TR", "BR")
    for point, label in zip(points, labels, strict=True):
        location = tuple(int(value) for value in point)
        cv2.circle(result, location, 7, (255, 80, 0), -1, cv2.LINE_AA)
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


def _validate_image_size(width: int, height: int) -> None:
    if width < 2 or height < 2:
        raise ValueError("image width and height must both be at least 2")
