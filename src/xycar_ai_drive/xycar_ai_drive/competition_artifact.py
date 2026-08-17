"""Strict loader for a preloaded three-model competition bundle."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from xycar_ai_drive.artifact import RoadWarpParameters
from xycar_ai_drive.steering_contract import (
    SteeringContract,
    parse_steering_contract,
    require_normalized_steering_contract,
)


class CompetitionArtifactError(ValueError):
    """Raised when a competition bundle violates its deployment contract."""


@dataclass(frozen=True)
class ImageContract:
    model_path: Path
    width: int
    height: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    geometry: str
    road_warp: RoadWarpParameters | None = None
    hidden_size: int = 0
    horizon_steps: int = 0


@dataclass(frozen=True)
class MissionContract:
    probability_threshold: float
    stop_votes_required: int
    stop_votes_window: int
    go_votes_required: int
    go_votes_window: int
    decision_progress_deadline: float
    handoff_probability_threshold: float
    handoff_consecutive_frames: int
    handoff_max_angle_difference: float
    shortcut_timeout_sec: float
    maximum_forward_speed: float


@dataclass(frozen=True)
class CompetitionBundle:
    root: Path
    artifact_id: str
    digest: str
    base: ImageContract
    signal: ImageContract
    shortcut: ImageContract
    mission: MissionContract
    steering_contract: SteeringContract


def load_competition_bundle(root: str | Path) -> CompetitionBundle:
    bundle_root = Path(root).expanduser()
    if bundle_root.is_symlink():
        raise CompetitionArtifactError("bundle root must not be a symlink")
    bundle_root = bundle_root.resolve()
    if not bundle_root.is_dir():
        raise CompetitionArtifactError(f"bundle root is missing: {bundle_root}")
    _verify_checksums(bundle_root)
    manifest = _load_mapping(bundle_root / "manifest.yaml")
    if manifest.get("schema_version") != 1:
        raise CompetitionArtifactError("bundle schema_version must be 1")
    if manifest.get("artifact_kind") != "competition_bundle":
        raise CompetitionArtifactError("artifact is not a competition bundle")
    try:
        steering_contract = require_normalized_steering_contract(
            parse_steering_contract(
                manifest.get("steering_contract"),
                context="manifest.steering_contract",
            ),
            context="competition bundle steering contract",
        )
    except ValueError as exc:
        raise CompetitionArtifactError(str(exc)) from exc
    artifact_id = _required_string(manifest, "artifact_id", "manifest")
    if artifact_id != bundle_root.name:
        raise CompetitionArtifactError("bundle artifact_id does not match directory")
    models = _required_mapping(manifest, "models", "manifest")
    base = _base_contract(bundle_root, _required_mapping(models, "base", "models"))
    signal = _temporal_contract(
        bundle_root,
        _required_mapping(models, "signal", "models"),
        expected_file="signal_model.ts",
        expected_geometry="upper_two_thirds_bicubic_resize",
        require_horizon=False,
    )
    shortcut = _temporal_contract(
        bundle_root,
        _required_mapping(models, "shortcut", "models"),
        expected_file="shortcut_model.ts",
        expected_geometry="full_frame_bicubic_resize_unwarped",
        require_horizon=True,
    )
    mission_raw = _required_mapping(manifest, "mission", "manifest")
    runtime_raw = _required_mapping(manifest, "runtime", "manifest")
    if mission_raw.get("action_priority") != ["STOP", "LEFT", "STRAIGHT"]:
        raise CompetitionArtifactError("unsupported mission action priority")
    mission = MissionContract(
        probability_threshold=_probability(
            mission_raw, "signal_probability_threshold"
        ),
        stop_votes_required=_positive_int(
            _required_mapping(mission_raw, "stop_votes", "mission"),
            "required",
        ),
        stop_votes_window=_positive_int(
            _required_mapping(mission_raw, "stop_votes", "mission"),
            "window",
        ),
        go_votes_required=_positive_int(
            _required_mapping(mission_raw, "go_votes", "mission"),
            "required",
        ),
        go_votes_window=_positive_int(
            _required_mapping(mission_raw, "go_votes", "mission"),
            "window",
        ),
        decision_progress_deadline=_probability(
            mission_raw, "decision_progress_deadline"
        ),
        handoff_probability_threshold=_probability(
            mission_raw, "handoff_probability_threshold"
        ),
        handoff_consecutive_frames=_positive_int(
            mission_raw, "handoff_consecutive_frames"
        ),
        handoff_max_angle_difference=_finite_positive(
            mission_raw, "handoff_max_angle_difference"
        ),
        shortcut_timeout_sec=_finite_positive(
            mission_raw, "shortcut_timeout_sec"
        ),
        maximum_forward_speed=_finite_positive(
            runtime_raw, "maximum_forward_speed"
        ),
    )
    if mission.stop_votes_required > mission.stop_votes_window:
        raise CompetitionArtifactError("stop vote requirement exceeds window")
    if mission.go_votes_required > mission.go_votes_window:
        raise CompetitionArtifactError("go vote requirement exceeds window")
    checksum_path = bundle_root / "SHA256SUMS"
    digest = hashlib.sha256(checksum_path.read_bytes()).hexdigest()
    return CompetitionBundle(
        root=bundle_root,
        artifact_id=artifact_id,
        digest=digest,
        base=base,
        signal=signal,
        shortcut=shortcut,
        mission=mission,
        steering_contract=steering_contract,
    )


def _base_contract(root: Path, raw: Mapping[str, Any]) -> ImageContract:
    model_path = _model_path(root, raw, "base_model.ts")
    model = _required_mapping(raw, "model", "base")
    model_input = _required_mapping(model, "input", "base.model")
    shape = model_input.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or shape[:2] != [1, 3]
        or shape[2] != shape[3]
        or not isinstance(shape[2], int)
    ):
        raise CompetitionArtifactError("base input shape must be [1,3,N,N]")
    preprocessing = _required_mapping(raw, "preprocessing", "base")
    geometry = _required_string(preprocessing, "geometry", "base.preprocessing")
    if geometry not in {
        "full_frame_bicubic_resize",
        "perspective_road_warp_then_bicubic_resize",
    }:
        raise CompetitionArtifactError("unsupported base preprocessing geometry")
    road_warp = None
    if geometry == "perspective_road_warp_then_bicubic_resize":
        road_warp = _road_warp(preprocessing)
    return ImageContract(
        model_path=model_path,
        width=shape[3],
        height=shape[2],
        mean=_three_floats(preprocessing, "mean"),
        std=_three_floats(preprocessing, "std", positive=True),
        geometry=geometry,
        road_warp=road_warp,
    )


def _temporal_contract(
    root: Path,
    raw: Mapping[str, Any],
    *,
    expected_file: str,
    expected_geometry: str,
    require_horizon: bool,
) -> ImageContract:
    model_path = _model_path(root, raw, expected_file)
    config = _required_mapping(raw, "config", expected_file)
    preprocessing = _required_mapping(raw, "preprocessing", expected_file)
    if preprocessing.get("geometry") != expected_geometry:
        raise CompetitionArtifactError(
            f"unexpected preprocessing geometry for {expected_file}"
        )
    shape = preprocessing.get("input_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or shape[:2] != [1, 3]
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
    ):
        raise CompetitionArtifactError(f"invalid input shape for {expected_file}")
    horizon = _positive_int(config, "horizon_steps") if require_horizon else 0
    return ImageContract(
        model_path=model_path,
        width=shape[3],
        height=shape[2],
        mean=_three_floats(preprocessing, "mean"),
        std=_three_floats(preprocessing, "std", positive=True),
        geometry=expected_geometry,
        hidden_size=_positive_int(config, "hidden_size"),
        horizon_steps=horizon,
    )


def _model_path(root: Path, raw: Mapping[str, Any], expected: str) -> Path:
    if raw.get("file") != expected:
        raise CompetitionArtifactError(f"model filename must be {expected}")
    path = root / expected
    if path.is_symlink() or not path.is_file():
        raise CompetitionArtifactError(f"model is missing or unsafe: {path}")
    return path


def _road_warp(preprocessing: Mapping[str, Any]) -> RoadWarpParameters:
    contract = _required_mapping(preprocessing, "road_warp", "preprocessing")
    parameters = _required_mapping(contract, "parameters", "road_warp")
    return RoadWarpParameters(
        top_y=_finite_number(parameters, "top_y"),
        bottom_y=_finite_number(parameters, "bottom_y"),
        top_left_x=_finite_number(parameters, "top_left_x"),
        top_right_x=_finite_number(parameters, "top_right_x"),
        bottom_left_x=_finite_number(parameters, "bottom_left_x"),
        bottom_right_x=_finite_number(parameters, "bottom_right_x"),
        bev_width=_positive_int(parameters, "bev_width"),
        bev_height=_positive_int(parameters, "bev_height"),
        dst_left_x=_finite_number(parameters, "dst_left_x"),
        dst_right_x=_finite_number(parameters, "dst_right_x"),
    )


def _verify_checksums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise CompetitionArtifactError("SHA256SUMS is missing or unsafe")
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise CompetitionArtifactError("invalid SHA256SUMS line")
        relative = parts[1].removeprefix("*")
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or not re.fullmatch(
            r"[A-Za-z0-9._/-]+", relative
        ):
            raise CompetitionArtifactError("unsafe SHA256SUMS path")
        if relative in expected:
            raise CompetitionArtifactError("duplicate SHA256SUMS path")
        expected[relative] = parts[0]
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(expected) != actual:
        raise CompetitionArtifactError("SHA256SUMS file list mismatch")
    for relative, digest in expected.items():
        path = root / relative
        if path.is_symlink() or _sha256(path) != digest:
            raise CompetitionArtifactError(f"bundle checksum mismatch: {relative}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CompetitionArtifactError(f"missing or unsafe YAML: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CompetitionArtifactError(f"invalid YAML: {path}") from exc
    if not isinstance(value, Mapping):
        raise CompetitionArtifactError(f"YAML root must be a mapping: {path}")
    return value


def _required_mapping(
    mapping: Mapping[str, Any],
    key: str,
    label: str,
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise CompetitionArtifactError(f"{label}.{key} must be a mapping")
    return value


def _required_string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise CompetitionArtifactError(f"{label}.{key} must be a string")
    return value


def _positive_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CompetitionArtifactError(f"{key} must be a positive integer")
    return value


def _finite_number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompetitionArtifactError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CompetitionArtifactError(f"{key} must be finite")
    return result


def _finite_positive(mapping: Mapping[str, Any], key: str) -> float:
    result = _finite_number(mapping, key)
    if result <= 0.0:
        raise CompetitionArtifactError(f"{key} must be positive")
    return result


def _probability(mapping: Mapping[str, Any], key: str) -> float:
    result = _finite_number(mapping, key)
    if not 0.0 < result <= 1.0:
        raise CompetitionArtifactError(f"{key} must be in (0,1]")
    return result


def _three_floats(
    mapping: Mapping[str, Any],
    key: str,
    *,
    positive: bool = False,
) -> tuple[float, float, float]:
    raw = mapping.get(key)
    if not isinstance(raw, list) or len(raw) != 3:
        raise CompetitionArtifactError(f"{key} must contain three values")
    values = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in values):
        raise CompetitionArtifactError(f"{key} must be finite")
    if positive and any(value <= 0.0 for value in values):
        raise CompetitionArtifactError(f"{key} must be positive")
    return values
