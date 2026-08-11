"""Validate and load the deployed front-camera policy artifact contract."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

ARTIFACT_SCHEMA_VERSION = 1
CHECKSUM_FILENAME = 'SHA256SUMS'
MANIFEST_FILENAME = 'manifest.yaml'


class ArtifactContractError(ValueError):
    """Raised when a deployed model artifact violates its contract."""


@dataclass(frozen=True)
class PolicyArtifact:
    root: Path
    artifact_id: str
    model_path: Path
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def load_policy_artifact(root: str | Path) -> PolicyArtifact:
    root = Path(root).expanduser()
    if root.is_symlink():
        raise ArtifactContractError(
            f'artifact directory must not be a symlink: {root}'
        )
    root = root.resolve()
    if not root.is_dir():
        raise ArtifactContractError(f'artifact directory is missing: {root}')
    _verify_checksums(root)
    manifest = _load_mapping(root / MANIFEST_FILENAME)
    if manifest.get('schema_version') != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactContractError('unsupported artifact schema_version')
    artifact_id = _required_string(manifest, 'artifact_id', 'manifest')
    if artifact_id != root.name:
        raise ArtifactContractError(
            f'artifact id does not match directory name: {artifact_id}'
        )

    model = _required_mapping(manifest, 'model', 'manifest')
    if model.get('format') != 'torchscript':
        raise ArtifactContractError('model.format must be torchscript')
    model_relative = _required_string(model, 'file', 'model')
    _validate_relative_path(model_relative)
    model_path = root / model_relative
    if not model_path.is_file() or model_path.is_symlink():
        raise ArtifactContractError(f'model file is missing or unsafe: {model_path}')

    model_input = _required_mapping(model, 'input', 'model')
    if model_input.get('color_space') != 'RGB':
        raise ArtifactContractError('model input color_space must be RGB')
    if model_input.get('dtype') != 'float32':
        raise ArtifactContractError('model input dtype must be float32')
    shape = model_input.get('shape')
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or shape[0] != 1
        or shape[1] != 3
        or not isinstance(shape[2], int)
        or isinstance(shape[2], bool)
        or shape[2] <= 0
        or shape[3] != shape[2]
    ):
        raise ArtifactContractError('model input shape must be [1,3,N,N]')
    image_size = shape[2]

    model_output = _required_mapping(model, 'output', 'model')
    if model_output.get('kind') != 'tuple':
        raise ArtifactContractError('model output kind must be tuple')
    if model_output.get('order') != ['angle_logits', 'speed_logits']:
        raise ArtifactContractError('model output order is unsupported')
    if model_output.get('shapes') != [[1, 201], [1, 201]]:
        raise ArtifactContractError('model output shapes must both be [1,201]')

    preprocessing = _required_mapping(
        manifest,
        'preprocessing',
        'manifest',
    )
    if preprocessing.get('geometry') != 'full_frame_bicubic_resize':
        raise ArtifactContractError('unsupported preprocessing geometry')
    if preprocessing.get('image_size') != image_size:
        raise ArtifactContractError('preprocessing image_size mismatch')
    mean = _three_finite_floats(preprocessing, 'mean')
    std = _three_finite_floats(preprocessing, 'std')
    if any(value <= 0.0 for value in std):
        raise ArtifactContractError('preprocessing std values must be positive')

    label_contract = _required_mapping(
        manifest,
        'label_contract',
        'manifest',
    )
    if label_contract.get('num_classes') != 201:
        raise ArtifactContractError('label contract must use 201 classes')
    if label_contract.get('decode_mapping') != 'class_id - 100':
        raise ArtifactContractError('unsupported label decode mapping')

    return PolicyArtifact(
        root=root,
        artifact_id=artifact_id,
        model_path=model_path,
        image_size=image_size,
        mean=mean,
        std=std,
    )


def _verify_checksums(root: Path) -> None:
    checksum_path = root / CHECKSUM_FILENAME
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise ArtifactContractError('SHA256SUMS is missing or unsafe')
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding='utf-8').splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ArtifactContractError('invalid SHA256SUMS line')
        digest, relative = parts
        relative = relative.removeprefix('*')
        _validate_relative_path(relative)
        if not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ArtifactContractError('invalid SHA-256 digest')
        if relative in expected:
            raise ArtifactContractError(f'duplicate checksum path: {relative}')
        expected[relative] = digest

    for path in root.rglob('*'):
        if path.is_symlink():
            raise ArtifactContractError(f'artifact contains a symlink: {path}')
        if not path.is_dir() and not path.is_file():
            raise ArtifactContractError(f'artifact contains a special file: {path}')
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob('*')
        if path.is_file() and path.name != CHECKSUM_FILENAME
    }
    if actual_files != set(expected):
        raise ArtifactContractError(
            'SHA256SUMS file list does not match artifact contents'
        )
    for relative, expected_digest in expected.items():
        path = root / relative
        if _sha256_file(path) != expected_digest:
            raise ArtifactContractError(f'checksum mismatch: {relative}')


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path) -> Mapping[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ArtifactContractError(f'manifest is missing or unsafe: {path}')
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(payload, Mapping):
        raise ArtifactContractError('manifest must contain a mapping')
    return payload


def _required_mapping(
    payload: Mapping[str, object],
    key: str,
    context: str,
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ArtifactContractError(f'{context}.{key} must be a mapping')
    return value


def _required_string(
    payload: Mapping[str, object],
    key: str,
    context: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ArtifactContractError(
            f'{context}.{key} must be a non-empty string'
        )
    return value


def _three_finite_floats(
    payload: Mapping[str, object],
    key: str,
) -> tuple[float, float, float]:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != 3:
        raise ArtifactContractError(f'preprocessing.{key} must have 3 values')
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(
            f'preprocessing.{key} must be numeric'
        ) from exc
    if not all(math.isfinite(item) for item in result):
        raise ArtifactContractError(
            f'preprocessing.{key} must contain finite values'
        )
    return result[0], result[1], result[2]


def _validate_relative_path(relative: str) -> None:
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or '..' in path.parts
        or not re.fullmatch(r'[A-Za-z0-9._/-]+', relative)
    ):
        raise ArtifactContractError(f'unsafe artifact path: {relative}')
