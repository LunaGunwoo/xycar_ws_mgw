"""Fail-closed validation for external Guided collection profiles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from xycar_ai_drive.steering_contract import require_steering_contract_name


def validate_collection_profile(configured_path: str) -> None:
    if not configured_path:
        raise ValueError('collection_profile_path must not be empty')
    path = Path(configured_path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(
            'collection_profile_path must be an existing absolute file'
        )
    try:
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(
            f'collection profile must be valid UTF-8 YAML: {path}'
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError('collection profile root must be a mapping')
    node_config = payload.get('guided_policy_collector')
    if not isinstance(node_config, Mapping):
        raise ValueError(
            'collection profile must configure guided_policy_collector'
        )
    parameters = node_config.get('ros__parameters')
    if not isinstance(parameters, Mapping):
        raise ValueError(
            'guided_policy_collector.ros__parameters must be a mapping'
        )
    if 'residual_gain' in parameters:
        raise ValueError(
            'legacy residual_gain is not supported; replace it with '
            'max_steering_angle'
        )
    if 'max_steering_angle' not in parameters:
        raise ValueError(
            'collection profile must explicitly set max_steering_angle'
        )
    if 'steering_takeover_button' not in parameters:
        raise ValueError(
            'collection profile must explicitly set '
            'steering_takeover_button'
        )
    require_steering_contract_name(parameters.get('steering_contract'))
