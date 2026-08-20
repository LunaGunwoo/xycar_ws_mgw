"""Compact AR control vocabulary shared by training and export."""

from __future__ import annotations

import math

LEGACY_CONTROL_ENCODING = "legacy_command_201"
COMPACT_CONTROL_ENCODING = "driver_compact_v2"
CONTROL_ENCODINGS = {LEGACY_CONTROL_ENCODING, COMPACT_CONTROL_ENCODING}

DRIVER_ANGLE_MIN = -50
DRIVER_ANGLE_MAX = 50
DRIVER_ANGLE_OFFSET = 50
ANGLE_OUTPUT_CLASSES = 101

SPEED_MIN = 0
SPEED_MAX = 30
SPEED_OUTPUT_CLASSES = 31

NUMERIC_TOKEN_MIN = -50
NUMERIC_TOKEN_MAX = 50
NUMERIC_TOKEN_OFFSET = 50
NUMERIC_TOKEN_COUNT = 101
UNKNOWN_ANGLE_TOKEN_ID = 101
UNKNOWN_SPEED_TOKEN_ID = 102
ANGLE_QUERY_TOKEN_ID = 103
SPEED_QUERY_TOKEN_ID = 104
CONTROL_TOKEN_COUNT = 105

NORMALIZED_TO_DRIVER_SCALE = 0.5


def normalized_angle_to_driver(value: float) -> int:
    numeric = _finite(value, "normalized angle")
    return round(
        max(
            DRIVER_ANGLE_MIN,
            min(DRIVER_ANGLE_MAX, numeric * NORMALIZED_TO_DRIVER_SCALE),
        )
    )


def driver_angle_to_normalized(value: int | float) -> float:
    numeric = _finite(value, "driver angle")
    if not DRIVER_ANGLE_MIN <= numeric <= DRIVER_ANGLE_MAX:
        raise ValueError("driver angle must be in [-50,50]")
    return numeric / NORMALIZED_TO_DRIVER_SCALE


def angle_target_class_id(normalized_angle: float) -> int:
    return normalized_angle_to_driver(normalized_angle) + DRIVER_ANGLE_OFFSET


def speed_target_class_id(speed: float) -> int:
    numeric = _finite(speed, "speed")
    return round(max(SPEED_MIN, min(SPEED_MAX, numeric)))


def angle_class_id_to_normalized(class_id: int) -> float:
    _validate_int_range(class_id, 0, ANGLE_OUTPUT_CLASSES - 1, "angle class id")
    return driver_angle_to_normalized(class_id - DRIVER_ANGLE_OFFSET)


def speed_class_id_to_command(class_id: int) -> float:
    _validate_int_range(class_id, 0, SPEED_OUTPUT_CLASSES - 1, "speed class id")
    return float(class_id)


def angle_class_id_to_history_token(class_id: int) -> int:
    _validate_int_range(class_id, 0, ANGLE_OUTPUT_CLASSES - 1, "angle class id")
    return class_id


def speed_class_id_to_history_token(class_id: int) -> int:
    _validate_int_range(class_id, 0, SPEED_OUTPUT_CLASSES - 1, "speed class id")
    return class_id + NUMERIC_TOKEN_OFFSET


def executed_command_to_history_tokens(
    normalized_angle: float,
    speed: float,
) -> tuple[int, int]:
    return (
        angle_target_class_id(normalized_angle),
        speed_class_id_to_history_token(speed_target_class_id(speed)),
    )


def unknown_history_pair() -> tuple[int, int]:
    return UNKNOWN_ANGLE_TOKEN_ID, UNKNOWN_SPEED_TOKEN_ID


def flip_angle_history_token(token_id: int) -> int:
    if token_id == UNKNOWN_ANGLE_TOKEN_ID:
        return token_id
    _validate_int_range(token_id, 0, ANGLE_OUTPUT_CLASSES - 1, "angle token id")
    return ANGLE_OUTPUT_CLASSES - 1 - token_id


def validate_history_token_pair(angle_token_id: int, speed_token_id: int) -> None:
    if angle_token_id != UNKNOWN_ANGLE_TOKEN_ID:
        _validate_int_range(
            angle_token_id,
            0,
            ANGLE_OUTPUT_CLASSES - 1,
            "angle history token id",
        )
    if speed_token_id != UNKNOWN_SPEED_TOKEN_ID:
        _validate_int_range(
            speed_token_id,
            NUMERIC_TOKEN_OFFSET + SPEED_MIN,
            NUMERIC_TOKEN_OFFSET + SPEED_MAX,
            "speed history token id",
        )


def _finite(value: int | float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _validate_int_range(value: int, low: int, high: int, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not low <= value <= high
    ):
        raise ValueError(f"{label} must be in [{low},{high}]")
