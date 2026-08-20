"""Strict normalized steering contract shared by vehicle AI runtimes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

STEERING_CONTRACT_NAME = 'normalized_percent_v2'


@dataclass(frozen=True)
class SteeringContract:
    schema_version: int
    name: str
    command_min: float
    command_max: float
    driver_min: float
    driver_max: float
    mapping: str


NORMALIZED_STEERING_CONTRACT = SteeringContract(
    schema_version=1,
    name=STEERING_CONTRACT_NAME,
    command_min=-100.0,
    command_max=100.0,
    driver_min=-50.0,
    driver_max=50.0,
    mapping='linear_scale_0.5',
)


def steering_contract_mapping() -> dict[str, object]:
    return asdict(NORMALIZED_STEERING_CONTRACT)


def session_steering_contract(
    *,
    motor_topic: str,
    driver_topic: str = '/xycar_motor_safe',
) -> dict[str, object]:
    return {
        **steering_contract_mapping(),
        'motor_topic': motor_topic,
        'driver_topic': driver_topic,
    }


def parse_steering_contract(
    raw: object,
    *,
    context: str,
) -> SteeringContract | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f'{context} must be a mapping')
    expected_keys = set(steering_contract_mapping())
    if set(raw) != expected_keys:
        raise ValueError(f'{context} keys are incompatible')
    numeric_keys = (
        'command_min',
        'command_max',
        'driver_min',
        'driver_max',
    )
    values: dict[str, float] = {}
    for key in numeric_keys:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{context}.{key} must be numeric')
        values[key] = float(value)
        if not math.isfinite(values[key]):
            raise ValueError(f'{context}.{key} must be finite')
    schema_version = raw.get('schema_version')
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError(f'{context}.schema_version must be an integer')
    name = raw.get('name')
    mapping = raw.get('mapping')
    if not isinstance(name, str) or not isinstance(mapping, str):
        raise ValueError(f'{context} name and mapping must be strings')
    return SteeringContract(
        schema_version=schema_version,
        name=name,
        command_min=values['command_min'],
        command_max=values['command_max'],
        driver_min=values['driver_min'],
        driver_max=values['driver_max'],
        mapping=mapping,
    )


def require_normalized_steering_contract(
    contract: SteeringContract | None,
    *,
    context: str,
) -> SteeringContract:
    if contract != NORMALIZED_STEERING_CONTRACT:
        raise ValueError(
            f'{context} must use {STEERING_CONTRACT_NAME}'
        )
    return contract


def require_steering_contract_name(value: object) -> str:
    if value != STEERING_CONTRACT_NAME:
        raise ValueError(
            'steering_contract must be normalized_percent_v2'
        )
    return STEERING_CONTRACT_NAME
