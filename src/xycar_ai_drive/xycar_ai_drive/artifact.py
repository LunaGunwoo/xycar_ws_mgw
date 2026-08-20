"""Validate and load the deployed front-camera policy artifact contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from xycar_ai_drive.steering_contract import (
    SteeringContract,
    parse_steering_contract,
)

SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = {1, 2, 3, 5, 6}
LEGACY_CONTROL_ENCODING = 'legacy_command_201'
COMPACT_CONTROL_ENCODING = 'driver_compact_v2'
CATEGORICAL_PREDICTION_MODE = 'categorical'
CONTINUOUS_REGRESSION_PREDICTION_MODE = 'continuous_regression'
CHECKSUM_FILENAME = 'SHA256SUMS'
MANIFEST_FILENAME = 'manifest.yaml'


class ArtifactContractError(ValueError):
    """Raised when a deployed model artifact violates its contract."""


@dataclass(frozen=True)
class RoadWarpParameters:
    top_y: float
    bottom_y: float
    top_left_x: float
    top_right_x: float
    bottom_left_x: float
    bottom_right_x: float
    bev_width: int
    bev_height: int
    dst_left_x: float
    dst_right_x: float


@dataclass(frozen=True)
class PolicyHistoryContract:
    frames: int
    initial_class_ids: tuple[int, int]
    update: str
    control_encoding: str = LEGACY_CONTROL_ENCODING

    def valid_pair(self, pair: object) -> bool:
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in pair)
        ):
            return False
        angle_id, speed_id = pair
        if self.control_encoding == COMPACT_CONTROL_ENCODING:
            return pair == [101, 102] or pair == (101, 102) or (
                0 <= angle_id <= 100 and 50 <= speed_id <= 80
            )
        return 0 <= angle_id <= 200 and 0 <= speed_id <= 200


@dataclass(frozen=True)
class PolicyArtifact:
    root: Path
    artifact_id: str
    model_path: Path
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    road_warp: RoadWarpParameters | None
    history: PolicyHistoryContract | None
    steering_contract: SteeringContract | None
    control_encoding: str = LEGACY_CONTROL_ENCODING
    output_shapes: tuple[tuple[int, int], tuple[int, int]] = (
        (1, 201),
        (1, 201),
    )
    prediction_mode: str = CATEGORICAL_PREDICTION_MODE


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
    schema_version = manifest.get('schema_version')
    if schema_version not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
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
        raise ArtifactContractError(
            f'model file is missing or unsafe: {model_path}'
        )

    model_input = _required_mapping(model, 'input', 'model')
    history = None
    control_encoding = (
        COMPACT_CONTROL_ENCODING
        if schema_version in {5, 6}
        else LEGACY_CONTROL_ENCODING
    )
    if schema_version == 1:
        shape = _validate_image_input(model_input)
    else:
        if model.get('architecture') != 'ar_control_tokens':
            raise ArtifactContractError('AR schema requires ar_control_tokens')
        if model_input.get('kind') != 'tuple':
            raise ArtifactContractError(
                'AR schema model input must be a tuple'
            )
        history_input_name = (
            'history_token_ids'
            if schema_version in {5, 6}
            else 'history_class_ids'
        )
        if model_input.get('order') != ['images', history_input_name]:
            raise ArtifactContractError(
                'AR schema model input order is unsupported'
            )
        shape = _validate_image_input(
            _required_mapping(model_input, 'images', 'model.input')
        )
        history_input = _required_mapping(
            model_input,
            history_input_name,
            'model.input',
        )
        if history_input.get('dtype') != 'int64':
            raise ArtifactContractError('history input dtype must be int64')
        if history_input.get('shape') != [1, 4, 2]:
            raise ArtifactContractError('history input shape must be [1,4,2]')
        history = _history_contract(manifest, schema_version=schema_version)
    image_size = shape[2]

    model_output = _required_mapping(model, 'output', 'model')
    if model_output.get('kind') != 'tuple':
        raise ArtifactContractError('model output kind must be tuple')
    expected_output_order = (
        ['angle_driver', 'speed']
        if schema_version == 6
        else ['angle_logits', 'speed_logits']
    )
    if model_output.get('order') != expected_output_order:
        raise ArtifactContractError('model output order is unsupported')
    expected_output_shapes = (
        [[1, 1], [1, 1]]
        if schema_version == 6
        else
        [[1, 101], [1, 31]]
        if schema_version == 5
        else [[1, 201], [1, 201]]
    )
    if model_output.get('shapes') != expected_output_shapes:
        raise ArtifactContractError(
            f'model output shapes must be {expected_output_shapes}'
        )
    prediction_mode = str(
        model.get('prediction_mode', CATEGORICAL_PREDICTION_MODE)
    )
    expected_prediction_mode = (
        CONTINUOUS_REGRESSION_PREDICTION_MODE
        if schema_version == 6
        else CATEGORICAL_PREDICTION_MODE
    )
    if prediction_mode != expected_prediction_mode:
        raise ArtifactContractError('model prediction_mode is incompatible')
    if schema_version == 6:
        _validate_regression_output_values(model_output)

    preprocessing = _required_mapping(
        manifest,
        'preprocessing',
        'manifest',
    )
    geometry = preprocessing.get('geometry')
    if geometry not in {
        'full_frame_bicubic_resize',
        'perspective_road_warp_then_bicubic_resize',
    }:
        raise ArtifactContractError('unsupported preprocessing geometry')
    if preprocessing.get('image_size') != image_size:
        raise ArtifactContractError('preprocessing image_size mismatch')
    mean = _three_finite_floats(preprocessing, 'mean')
    std = _three_finite_floats(preprocessing, 'std')
    if any(value <= 0.0 for value in std):
        raise ArtifactContractError(
            'preprocessing std values must be positive'
        )
    road_warp = None
    if geometry == 'perspective_road_warp_then_bicubic_resize':
        road_warp = _road_warp_parameters(preprocessing)
    elif 'road_warp' in preprocessing:
        raise ArtifactContractError(
            'full-frame preprocessing must not define road_warp'
        )

    label_contract = _required_mapping(
        manifest,
        'label_contract',
        'manifest',
    )
    if schema_version in {5, 6}:
        _validate_compact_label_contract(
            label_contract,
            regression=schema_version == 6,
        )
    else:
        if label_contract.get('num_classes') != 201:
            raise ArtifactContractError('label contract must use 201 classes')
        if label_contract.get('decode_mapping') != 'class_id - 100':
            raise ArtifactContractError('unsupported label decode mapping')

    try:
        steering_contract = parse_steering_contract(
            manifest.get('steering_contract'),
            context='manifest.steering_contract',
        )
    except ValueError as exc:
        raise ArtifactContractError(str(exc)) from exc

    return PolicyArtifact(
        root=root,
        artifact_id=artifact_id,
        model_path=model_path,
        image_size=image_size,
        mean=mean,
        std=std,
        road_warp=road_warp,
        history=history,
        steering_contract=steering_contract,
        control_encoding=control_encoding,
        output_shapes=tuple(tuple(value) for value in expected_output_shapes),
        prediction_mode=prediction_mode,
    )


def _validate_image_input(model_input: Mapping[str, object]) -> list[object]:
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
    return shape


def _history_contract(
    manifest: Mapping[str, object],
    *,
    schema_version: int,
) -> PolicyHistoryContract:
    history = _required_mapping(manifest, 'history', 'manifest')
    if history.get('frames') != 4:
        raise ArtifactContractError('history.frames must be 4')
    expected_pair_order = (
        ['angle_token_id', 'speed_token_id']
        if schema_version in {5, 6}
        else ['angle_class_id', 'speed_class_id']
    )
    if history.get('pair_order') != expected_pair_order:
        raise ArtifactContractError('history pair order is unsupported')
    if history.get('time_order') != 'oldest_to_newest':
        raise ArtifactContractError('history time order is unsupported')
    expected_update = (
        'externally_executed_commands'
        if schema_version in {3, 5, 6}
        else 'predicted_argmax'
    )
    if history.get('update') != expected_update:
        raise ArtifactContractError('history update mode is unsupported')
    initial_key = (
        'initial_token_ids'
        if schema_version in {5, 6}
        else 'initial_class_ids'
    )
    initial_ids = history.get(initial_key)
    if (
        not isinstance(initial_ids, list)
        or len(initial_ids) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= (102 if schema_version in {5, 6} else 200)
            for value in initial_ids
        )
    ):
        raise ArtifactContractError('history initial class ids are invalid')
    initialization = history.get('initialization')
    if schema_version in {5, 6} and initialization == 'learned_unknown_tokens':
        expected_initial = [101, 102]
    elif (
        schema_version in {5, 6}
        and initialization == 'canonical_initial_command'
    ):
        initial_command = history.get('initial_command')
        if (
            not isinstance(initial_command, list)
            or len(initial_command) != 2
            or any(isinstance(value, bool) for value in initial_command)
            or not all(isinstance(value, (int, float)) for value in initial_command)
            or not all(math.isfinite(float(value)) for value in initial_command)
            or not -100 <= float(initial_command[0]) <= 100
            or not 0 <= float(initial_command[1]) <= 30
        ):
            raise ArtifactContractError(
                'schema v5 canonical history command is invalid'
            )
        expected_initial = [
            round(float(initial_command[0]) * 0.5) + 50,
            round(float(initial_command[1])) + 50,
        ]
    else:
        expected_initial = [100, 125]
    if initial_ids != expected_initial:
        raise ArtifactContractError(
            f'history initial ids must be {expected_initial}'
        )
    if schema_version in {5, 6}:
        if initialization not in {
            'learned_unknown_tokens',
            'canonical_initial_command',
        }:
            raise ArtifactContractError(
                'schema v5 history initialization is unsupported'
            )
        if history.get('actual_angle_token_range') != [0, 100]:
            raise ArtifactContractError('schema v5 angle token range is invalid')
        if history.get('actual_speed_token_range') != [50, 80]:
            raise ArtifactContractError('schema v5 speed token range is invalid')
    else:
        initial_command = history.get('initial_command')
        if initial_command != [0, 25]:
            raise ArtifactContractError('history initial command must be [0,25]')
    return PolicyHistoryContract(
        frames=4,
        initial_class_ids=(initial_ids[0], initial_ids[1]),
        update=expected_update,
        control_encoding=(
            COMPACT_CONTROL_ENCODING
            if schema_version in {5, 6}
            else LEGACY_CONTROL_ENCODING
        ),
    )


def _validate_compact_label_contract(
    contract: Mapping[str, object],
    *,
    regression: bool,
) -> None:
    if contract.get('control_encoding') != COMPACT_CONTROL_ENCODING:
        raise ArtifactContractError('compact control encoding is unsupported')
    expected_shapes = (
        {'angle_driver': [1, 1], 'speed': [1, 1]}
        if regression
        else {'angle_logits': [1, 101], 'speed_logits': [1, 31]}
    )
    if contract.get('output_shapes') != expected_shapes:
        raise ArtifactContractError('compact label output shapes are invalid')
    angle = _required_mapping(contract, 'angle', 'label_contract')
    speed = _required_mapping(contract, 'speed', 'label_contract')
    if regression:
        if contract.get('prediction_mode') != CONTINUOUS_REGRESSION_PREDICTION_MODE:
            raise ArtifactContractError('schema v6 prediction mode is invalid')
        if (
            angle.get('unit') != 'driver_angle'
            or angle.get('range') != [-50.0, 50.0]
            or angle.get('runtime_normalized_mapping') != 'angle_driver * 2'
            or speed.get('unit') != 'motor_speed'
            or speed.get('range') != [0.0, 30.0]
        ):
            raise ArtifactContractError('schema v6 scalar label contract is invalid')
        return
    vocabulary = _required_mapping(
        contract,
        'shared_numeric_vocabulary',
        'label_contract',
    )
    if angle.get('num_classes') != 101 or angle.get('driver_range') != [-50, 50]:
        raise ArtifactContractError('schema v5 angle label contract is invalid')
    if speed.get('num_classes') != 31 or speed.get('command_range') != [0, 30]:
        raise ArtifactContractError('schema v5 speed label contract is invalid')
    if (
        vocabulary.get('numeric_range') != [-50, 50]
        or vocabulary.get('unknown_angle_token_id') != 101
        or vocabulary.get('unknown_speed_token_id') != 102
        or vocabulary.get('angle_query_token_id') != 103
        or vocabulary.get('speed_query_token_id') != 104
        or vocabulary.get('vocabulary_size') != 105
    ):
        raise ArtifactContractError('schema v5 numeric vocabulary is invalid')


def _validate_regression_output_values(
    output: Mapping[str, object],
) -> None:
    expected = [
        {
            'name': 'angle_driver',
            'dtype': 'float32',
            'unit': 'driver_angle',
            'range': [-50.0, 50.0],
            'runtime_normalized_mapping': 'value * 2',
        },
        {
            'name': 'speed',
            'dtype': 'float32',
            'unit': 'motor_speed',
            'range': [0.0, 30.0],
        },
    ]
    if output.get('values') != expected:
        raise ArtifactContractError('schema v6 output value contract is invalid')


def _road_warp_parameters(
    preprocessing: Mapping[str, object],
) -> RoadWarpParameters:
    contract = _required_mapping(preprocessing, 'road_warp', 'preprocessing')
    if contract.get('schema_version') != 1:
        raise ArtifactContractError('unsupported road_warp schema_version')
    if contract.get('source_point_order') != [
        'bottom_left',
        'top_left',
        'top_right',
        'bottom_right',
    ]:
        raise ArtifactContractError('unsupported road_warp source point order')
    if contract.get('coordinate_space') != 'normalized_input_frame':
        raise ArtifactContractError('unsupported road_warp coordinate space')
    if contract.get('interpolation') != 'bilinear':
        raise ArtifactContractError('unsupported road_warp interpolation')
    parameters = _required_mapping(contract, 'parameters', 'road_warp')
    expected = {
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
    }
    if set(parameters) != expected:
        raise ArtifactContractError(
            'road_warp parameter keys are incompatible'
        )
    expected_sha256 = contract.get('sha256')
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r'[0-9a-f]{64}', expected_sha256
    ):
        raise ArtifactContractError('road_warp sha256 is invalid')
    try:
        canonical = json.dumps(
            {'schema_version': 1, 'warp': dict(parameters)},
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        ).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(
            'road_warp parameters are not serializable'
        ) from exc
    if hashlib.sha256(canonical).hexdigest() != expected_sha256:
        raise ArtifactContractError('road_warp sha256 mismatch')
    result = RoadWarpParameters(
        top_y=_finite_number(parameters, 'top_y'),
        bottom_y=_finite_number(parameters, 'bottom_y'),
        top_left_x=_finite_number(parameters, 'top_left_x'),
        top_right_x=_finite_number(parameters, 'top_right_x'),
        bottom_left_x=_finite_number(parameters, 'bottom_left_x'),
        bottom_right_x=_finite_number(parameters, 'bottom_right_x'),
        bev_width=_positive_integer(parameters, 'bev_width'),
        bev_height=_positive_integer(parameters, 'bev_height'),
        dst_left_x=_finite_number(parameters, 'dst_left_x'),
        dst_right_x=_finite_number(parameters, 'dst_right_x'),
    )
    ratios = (
        result.top_y,
        result.bottom_y,
        result.top_left_x,
        result.top_right_x,
        result.bottom_left_x,
        result.bottom_right_x,
        result.dst_left_x,
        result.dst_right_x,
    )
    if not all(0.0 <= value <= 1.0 for value in ratios):
        raise ArtifactContractError('road_warp ratios must be in [0,1]')
    if result.bottom_y - result.top_y < 0.02:
        raise ArtifactContractError('road_warp vertical range is too small')
    if result.top_right_x - result.top_left_x < 0.02:
        raise ArtifactContractError('road_warp top edge is too narrow')
    if result.bottom_right_x - result.bottom_left_x < 0.02:
        raise ArtifactContractError('road_warp bottom edge is too narrow')
    if not 80 <= result.bev_width <= 1920:
        raise ArtifactContractError('road_warp bev_width is outside limits')
    if not 60 <= result.bev_height <= 1080:
        raise ArtifactContractError('road_warp bev_height is outside limits')
    if not 0.0 <= result.dst_left_x <= 0.49:
        raise ArtifactContractError('road_warp dst_left_x is outside limits')
    if not 0.51 <= result.dst_right_x <= 1.0:
        raise ArtifactContractError('road_warp dst_right_x is outside limits')
    return result


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
            raise ArtifactContractError(
                f'artifact contains a special file: {path}'
            )
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


def _finite_number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactContractError(f'road_warp.{key} must be numeric')
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactContractError(f'road_warp.{key} must be finite')
    return result


def _positive_integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactContractError(
            f'road_warp.{key} must be a positive integer'
        )
    return value


def _validate_relative_path(relative: str) -> None:
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or '..' in path.parts
        or not re.fullmatch(r'[A-Za-z0-9._/-]+', relative)
    ):
        raise ArtifactContractError(f'unsafe artifact path: {relative}')
