"""Canonical normalized steering contract for training and export."""

from __future__ import annotations

from collections.abc import Mapping
import math

STEERING_CONTRACT_NAME = "normalized_percent_v1"


def steering_contract_mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": STEERING_CONTRACT_NAME,
        "command_min": -100.0,
        "command_max": 100.0,
        "driver_min": -40.0,
        "driver_max": 40.0,
        "mapping": "linear_scale_0.4",
    }


def session_steering_contract_mapping() -> dict[str, object]:
    return {
        **steering_contract_mapping(),
        "motor_topic": "/xycar_motor",
        "driver_topic": "/xycar_motor_safe",
    }


def validate_required_contract_name(value: str | None) -> None:
    if value is not None and value != STEERING_CONTRACT_NAME:
        raise ValueError(
            "required_steering_contract must be normalized_percent_v1"
        )


def is_exact_steering_contract(
    raw: object,
    *,
    include_topics: bool = False,
) -> bool:
    if not isinstance(raw, Mapping):
        return False
    expected = (
        session_steering_contract_mapping()
        if include_topics
        else steering_contract_mapping()
    )
    if set(raw) != set(expected):
        return False
    schema_version = raw.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        return False
    for key in ("command_min", "command_max", "driver_min", "driver_max"):
        value = raw.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != expected[key]
        ):
            return False
    return all(
        isinstance(raw.get(key), str) and raw.get(key) == expected[key]
        for key in set(expected) - {
            "schema_version",
            "command_min",
            "command_max",
            "driver_min",
            "driver_max",
        }
    )


def metadata_has_required_steering_contract(
    metadata: Mapping[str, object],
    required_name: str | None,
) -> bool:
    if required_name is None:
        return True
    if required_name != STEERING_CONTRACT_NAME:
        return False
    return is_exact_steering_contract(
        metadata.get("steering_contract"),
        include_topics=True,
    )
