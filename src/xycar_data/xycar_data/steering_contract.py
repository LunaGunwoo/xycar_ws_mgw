"""Canonical steering metadata for newly collected vehicle sessions."""

from __future__ import annotations

STEERING_CONTRACT_NAME = 'normalized_percent_v2'


def require_steering_contract_name(value: object) -> str:
    if value != STEERING_CONTRACT_NAME:
        raise ValueError(
            'steering_contract must be normalized_percent_v2'
        )
    return STEERING_CONTRACT_NAME


def session_steering_contract(
    *,
    motor_topic: str,
    driver_topic: str = '/xycar_motor_safe',
) -> dict[str, object]:
    return {
        'schema_version': 1,
        'name': STEERING_CONTRACT_NAME,
        'command_min': -100.0,
        'command_max': 100.0,
        'driver_min': -50.0,
        'driver_max': 50.0,
        'mapping': 'linear_scale_0.5',
        'motor_topic': motor_topic,
        'driver_topic': driver_topic,
    }
