"""Strict loader for the traffic-light shortcut mission bundle."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from xycar_ai_drive.artifact import (
    ANGLE_REGRESSION_FIXED_SPEED_PREDICTION_MODE,
    COMPACT_CONTROL_ENCODING,
    CONTINUOUS_REGRESSION_PREDICTION_MODE,
    ArtifactContractError,
    PolicyArtifact,
    load_policy_artifact,
)
from xycar_ai_drive.steering_contract import NORMALIZED_STEERING_CONTRACT

BUNDLE_KIND = 'traffic_shortcut_bundle'
SUPPORTED_BUNDLE_SCHEMA_VERSIONS = {
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
}
BUNDLE_MANIFEST = 'manifest.yaml'
BUNDLE_CHECKSUMS = 'SHA256SUMS'
TRAFFIC_MODEL_SHA256 = (
    '24c1a38eacfb065c95e5577be29a2a542b985d6ea1954bf2fc94c52eb674aa41'
)
TRAFFIC_CLASSIFIER_SHA256 = (
    '3ddb126483e03211f59cd15eb1f96632b9c3d5ac0d42544ccd999601643b806b'
)
HUMAN_BBOX_TRAFFIC_MODEL_SHA256 = (
    '7d1bd24a025c6b7851c396e9e0cbc38dfad6f6e852c712e2cebdf8547a428e64'
)
HUMAN_BBOX_TRAFFIC_CLASSIFIER_SHA256 = (
    'e126f4f3036bcd4e44ab0ca4b5cc70a2e46f87bc859ae434719eb5f08de122b5'
)
EXPECTED_BASE_ARTIFACT_ID = (
    'front-cam-policy-vit-small-ar4-v2-nice-adaptive-joint-regression-'
    'sequence-init25-window5-20260821'
)
EXPECTED_SPEED35_BASE_ARTIFACT_ID = (
    'front-cam-policy-vit-small-ar4-v2-nice-ada-very-fast-joint-'
    'regression-sequence-init35-window5-speed35-20260823'
)
EXPECTED_SPEED35_FIX_BASE_ARTIFACT_ID = (
    'front-cam-policy-vit-small-ar4-v2-nice-ada-very-fast-fix-joint-'
    'regression-sequence-init35-window5-speed35-20260824'
)
EXPECTED_SHORTCUT_ARTIFACT_ID = (
    'nice-shortcut-resnet18-squarewarp-speed23-20260821'
)
EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID = (
    'nice-shortcut-resnet18-squarewarp-speed23-45sessions-20260821'
)
EXPECTED_SIGNAL_VOTE_BUNDLE_ID = (
    'traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-'
    'tl45-votes5-every3-20260821'
)
EXPECTED_EXPANDED_SIGNAL_VOTE_BUNDLE_ID = (
    'traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-'
    'tl45-votes5-every3-45sessions-20260821'
)
EXPECTED_CLASSIFIER_BUNDLE_ID = (
    'traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-'
    'yolo-cls-tl45-votes2-every3-45sessions-20260822'
)
EXPECTED_YOLO_MISSING_RELEASE_BUNDLE_ID = (
    'traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-'
    'yolo-cls-tl45-votes2-every3-yolo-miss30-release-45sessions-20260822'
)
EXPECTED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID = (
    'traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-'
    'yolo11s-humanbbox-cnn416-actions3-conf50-tl40to225-votes2-every3-'
    'yolo-miss30-release-45sessions-20260823'
)
EXPECTED_STABILIZED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID = (
    'traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-'
    'yolo11s-humanbbox-cnn416-actions3-conf50-tl40to225-stop3-go15-'
    'every3-yolo-miss30-release-45sessions-20260823'
)
EXPECTED_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID = (
    'traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-'
    'yolo11s-humanbbox-cnn416-actions3-conf50-tl40to225-stop3-go15-'
    'search3-classify1-yolo-miss30-release-45sessions-20260823'
)
EXPECTED_STOP10_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID = (
    'traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-'
    'yolo11s-humanbbox-cnn416-actions3-conf50-tl40to225-stop10-go15-'
    'search3-classify1-yolo-miss30-release-45sessions-20260823'
)
EXPECTED_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID = (
    'traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-'
    'yolo11s-humanbbox-cnn416-actions3-conf50-tl40to225-stop30-go30-'
    'search3-classify1-yolo-miss30-release-45sessions-20260823'
)
EXPECTED_SPEED35_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID = (
    'traffic-shortcut-nice-ada-very-fast-speed35-regression-resnet18-8s-'
    'shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-'
    'tl40to225-stop30-go30-search3-classify1-yolo-miss30-release-'
    '45sessions-20260823'
)
EXPECTED_SPEED35_STOP10_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID = (
    'traffic-shortcut-nice-ada-very-fast-speed35-regression-resnet18-8s-'
    'shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-'
    'tl40to225-stop10-go30-search3-classify1-yolo-miss30-release-'
    '45sessions-20260823'
)
EXPECTED_SPEED35_STOP15_GO15_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID = (
    'traffic-shortcut-nice-ada-very-fast-speed35-regression-resnet18-8s-'
    'shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-'
    'tl40to225-stop15-go15-search3-classify1-yolo-miss30-release-'
    '45sessions-20260823'
)
EXPECTED_SPEED35_INITIAL_STOP_ONCE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID = (
    'traffic-shortcut-nice-ada-very-fast-speed35-regression-resnet18-8s-'
    'shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-'
    'tl40to225-initial-stop15-nonstop3-left15-stop-once-search3-'
    'classify1-45sessions-20260823'
)
EXPECTED_SPEED35_INITIAL_WAIT_FRESH5_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID = (
    'traffic-shortcut-nice-ada-very-fast-speed35-regression-resnet18-8s-'
    'shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-'
    'tl40to225-initial-wait-all5-stop-once-left-direct-search3-'
    'classify1-vote-yolo3-45sessions-20260823'
)
EXPECTED_SPEED35_INITIAL_WAIT_FRESH3_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID = (
    'traffic-shortcut-nice-ada-very-fast-speed35-regression-resnet18-8s-'
    'shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-'
    'tl40to225-initial-wait-stop5-go3-stop-once-left-direct-search3-'
    'classify3-vote-yolo3-45sessions-20260823'
)
EXPECTED_SPEED35_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID = (
    'traffic-shortcut-nice-ada-very-fast-speed35-regression-resnet18-8s-'
    'shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-'
    'tl40to225-initial-wait-stop5-go1-stop-once-left-direct-search3-'
    'classify3-vote-yolo3-45sessions-20260823'
)
EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID = (
    'traffic-shortcut-nice-ada-very-fast-fix-speed35-regression-resnet18-'
    '8s-shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-'
    'tl40to225-initial-wait-stop5-go1-stop-once-left-direct-search3-'
    'classify3-vote-yolo3-45sessions-20260824'
)
EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_SESSION_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID = (
    'traffic-shortcut-nice-ada-very-fast-fix-speed35-regression-resnet18-'
    '8s-shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-'
    'tl40to225-initial-wait-stop5-go1-stoponce-leftsession1-search3-'
    'classify3-vote-yolo3-t500-45sessions-20260824'
)
EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_SESSION_4S_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID = (
    'traffic-shortcut-nice-ada-very-fast-fix-speed35-regression-resnet18-'
    '4s-shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-'
    'tl40to225-initial-wait-stop5-go1-stoponce-leftsession1-search3-'
    'classify3-vote-yolo3-t500-45sessions-20260824'
)
EXPECTED_SIGNAL_BUNDLE_SHORTCUT_IDS = {
    EXPECTED_SIGNAL_VOTE_BUNDLE_ID: EXPECTED_SHORTCUT_ARTIFACT_ID,
    EXPECTED_EXPANDED_SIGNAL_VOTE_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
    EXPECTED_CLASSIFIER_BUNDLE_ID: EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID,
    EXPECTED_YOLO_MISSING_RELEASE_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
    EXPECTED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
    EXPECTED_STABILIZED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
    EXPECTED_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
    EXPECTED_STOP10_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
    EXPECTED_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
    EXPECTED_SPEED35_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
    EXPECTED_SPEED35_STOP10_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
    EXPECTED_SPEED35_STOP15_GO15_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
    EXPECTED_SPEED35_INITIAL_STOP_ONCE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
    EXPECTED_SPEED35_INITIAL_WAIT_FRESH5_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
    EXPECTED_SPEED35_INITIAL_WAIT_FRESH3_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
    EXPECTED_SPEED35_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
    EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
    EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_SESSION_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
    EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_SESSION_4S_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID: (
        EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    ),
}
EXPECTED_ONNXRUNTIME_VERSION = '1.24.0'
EXPECTED_NUMPY_VERSION = '1.26.4'
EXPECTED_PROVIDERS = (
    'CUDAExecutionProvider',
    'CPUExecutionProvider',
)


@dataclass(frozen=True)
class TrafficDetectorContract:
    model_path: Path
    classifier_model_path: Path | None
    mode: str
    confidence_threshold: float
    bbox_width_min: int
    bbox_width_max: int
    inference_every_n_frames: int
    classification_every_n_frames_after_detection: int
    reuse_detected_bbox_between_yolo_frames: bool
    percentile: float
    red_index: int
    left_indices: tuple[int, int]
    straight_index: int
    red_consecutive_reads: int
    left_consecutive_reads: int
    straight_consecutive_reads: int
    classifier_crop_padding: float | None
    detector_preprocessing: str
    classifier_input_height: int | None
    classifier_input_width: int | None
    classifier_classes: tuple[str, ...]
    classifier_probability_threshold: float | None
    classifier_interpolation: str | None
    red_stop_yolo_missing_release_frames: int | None


@dataclass(frozen=True)
class TrafficShortcutBundle:
    root: Path
    artifact_id: str
    schema_version: int
    base: PolicyArtifact
    shortcut: PolicyArtifact
    detector: TrafficDetectorContract
    base_speed_cap: float
    shortcut_speed: float
    shortcut_duration_sec: float
    shortcut_entry_stop_control_cycles: int
    shortcut_exit_stop_control_cycles: int
    base_shadow_enabled: bool
    base_shadow_history_update: str | None
    base_shadow_max_age_sec: float | None
    successful_shortcut_once: bool
    successful_shortcut_once_scope: str
    action_priority: tuple[str, str, str]
    red_stop_yolo_missing_release_frames: int | None
    initial_stop_one_shot: bool
    initial_stop_clear_consecutive_reads: int | None
    headless_wait_for_first_signal: bool
    initial_left_direct_shortcut: bool
    control_vote_on_fresh_yolo_only: bool
    onnxruntime_version: str
    numpy_version: str
    providers: tuple[str, ...]


def load_traffic_shortcut_bundle(root: str | Path) -> TrafficShortcutBundle:
    root = _safe_root(root)
    _verify_bundle_checksums(root)
    manifest = _load_mapping(root / BUNDLE_MANIFEST)
    schema_version = manifest.get('schema_version')
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in SUPPORTED_BUNDLE_SCHEMA_VERSIONS
    ):
        raise ArtifactContractError(
            'traffic shortcut bundle schema must be 1..20'
        )
    if manifest.get('artifact_kind') != BUNDLE_KIND:
        raise ArtifactContractError(
            'artifact is not a traffic shortcut bundle'
        )
    artifact_id = _required_string(manifest, 'artifact_id', 'manifest')
    if artifact_id != root.name:
        raise ArtifactContractError('traffic shortcut bundle id mismatch')
    expected_shortcut_artifact_id = _expected_shortcut_artifact_id(
        schema_version=schema_version,
        artifact_id=artifact_id,
    )
    expected_base_artifact_id = _expected_base_artifact_id(
        schema_version=schema_version,
        artifact_id=artifact_id,
    )

    components = _required_mapping(manifest, 'components', 'manifest')
    base_contract = _required_mapping(components, 'base', 'components')
    shortcut_contract = _required_mapping(
        components,
        'shortcut',
        'components',
    )
    base = _load_component(root, base_contract, 'base')
    shortcut = _load_component(root, shortcut_contract, 'shortcut')
    _validate_policy_contracts(
        base,
        shortcut,
        base_contract,
        shortcut_contract,
        expected_base_artifact_id=expected_base_artifact_id,
        expected_shortcut_artifact_id=expected_shortcut_artifact_id,
    )

    signal = _required_mapping(components, 'traffic_light', 'components')
    signal_relative = _required_string(signal, 'file', 'traffic_light')
    _validate_relative_path(signal_relative)
    signal_path = root / signal_relative
    if signal_path.is_symlink() or not signal_path.is_file():
        raise ArtifactContractError('traffic light ONNX is missing or unsafe')
    if signal.get('format') != 'onnx':
        raise ArtifactContractError('traffic light model format must be onnx')
    expected_traffic_sha256 = (
        HUMAN_BBOX_TRAFFIC_MODEL_SHA256
        if schema_version
        in {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
        else TRAFFIC_MODEL_SHA256
    )
    if signal.get('sha256') != expected_traffic_sha256:
        raise ArtifactContractError(
            'traffic light model SHA-256 contract mismatch'
        )
    if _sha256_file(signal_path) != expected_traffic_sha256:
        raise ArtifactContractError('traffic light model checksum mismatch')
    if signal.get('input') != {
        'name': 'images',
        'dtype': 'float32',
        'shape': [1, 3, 640, 640],
    }:
        raise ArtifactContractError(
            'traffic light ONNX input contract mismatch'
        )
    if signal.get('output') != {
        'name': 'output0',
        'dtype': 'float32',
        'shape': [1, 5, 8400],
    }:
        raise ArtifactContractError(
            'traffic light ONNX output contract mismatch'
        )

    detector = _required_mapping(manifest, 'detector', 'manifest')
    bbox_width = detector.get('bbox_width_px')
    expected_bbox_width = (
        [40, 225]
        if schema_version
        in {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
        else [45, 200]
    )
    if bbox_width != expected_bbox_width:
        raise ArtifactContractError(
            'detector bbox width gate must be '
            f'[{expected_bbox_width[0]},{expected_bbox_width[1]}]'
        )
    confidence = _exact_number(detector, 'confidence_threshold', 0.25)
    inference_every = _exact_int(detector, 'inference_every_n_frames', 3)
    if schema_version in {8, 9, 10, 11, 12, 13, 14, 15}:
        classification_every = _exact_int(
            detector,
            'classification_every_n_frames_after_detection',
            1,
        )
        if detector.get('reuse_detected_bbox_between_yolo_frames') is not True:
            raise ArtifactContractError(
                'schema v8..v15 must reuse the detected bbox between YOLO frames'
            )
        reuse_detected_bbox = True
    elif schema_version in {16, 17, 18, 19, 20}:
        classification_every = _exact_int(
            detector,
            'classification_every_n_frames_after_detection',
            3,
        )
        if (
            detector.get('reuse_detected_bbox_between_yolo_frames')
            is not False
        ):
            raise ArtifactContractError(
                'schema v16..v20 must disable cached bbox classification'
            )
        reuse_detected_bbox = False
    else:
        if (
            'classification_every_n_frames_after_detection' in detector
            or 'reuse_detected_bbox_between_yolo_frames' in detector
        ):
            raise ArtifactContractError(
                'adaptive classifier cadence is only valid in schema v8..v20'
            )
        classification_every = inference_every
        reuse_detected_bbox = False
    detector_preprocessing = (
        'letterbox_640_center_pad114_bgr_to_rgb_float32_nchw_div255'
        if schema_version
        in {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
        else 'resize_640_bgr_to_rgb_float32_nchw_div255'
    )
    if detector.get('preprocessing') != detector_preprocessing:
        raise ArtifactContractError('traffic detector preprocessing mismatch')
    if detector.get('selection') != 'maximum_confidence_box':
        raise ArtifactContractError('traffic detector selection mismatch')
    if (
        schema_version in {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
        and detector.get('max_detections') != 1
    ):
        raise ArtifactContractError('traffic detector must return one box')
    classifier_path: Path | None = None
    classifier_crop_padding: float | None = None
    classifier_input_height: int | None = None
    classifier_input_width: int | None = None
    classifier_classes: tuple[str, ...] = ()
    classifier_probability_threshold: float | None = None
    classifier_interpolation: str | None = None
    if schema_version >= 4:
        (
            classifier_path,
            classifier_crop_padding,
            classifier_input_height,
            classifier_input_width,
            classifier_classes,
            classifier_probability_threshold,
            classifier_interpolation,
        ) = _load_classifier_contract(
            root,
            components,
            detector,
            schema_version=schema_version,
        )
        mode = 'yolo_cnn_classifier'
        percentile = 0.0
        red_index = 0
        left_indices = (2, 3)
        straight_index = 3
    else:
        lamp = _required_mapping(detector, 'lamp', 'detector')
        if lamp.get('count') != 4 or lamp.get('score') != 'hsv_v_percentile':
            raise ArtifactContractError('traffic lamp score contract mismatch')
        if lamp.get('red_index') != 0:
            raise ArtifactContractError('traffic lamp red index must be 0')
        if lamp.get('left_indices') != [2, 3]:
            raise ArtifactContractError(
                'traffic lamp left indices must be [2,3]'
            )
        if lamp.get('straight_index') != 3:
            raise ArtifactContractError(
                'traffic lamp straight index must be 3'
            )
        if lamp.get('straight_requires_red_off') is not True:
            raise ArtifactContractError('straight lamp must require red off')
        if lamp.get('relative_threshold') != '(min + max) / 2':
            raise ArtifactContractError(
                'traffic lamp relative threshold mismatch'
            )
        mode = 'hsv_lamp'
        percentile = _exact_number(lamp, 'percentile', 80.0)
        red_index = 0
        left_indices = (2, 3)
        straight_index = 3

    (
        red_consecutive_reads,
        left_consecutive_reads,
        straight_consecutive_reads,
    ) = _load_signal_vote_contract(
        manifest,
        schema_version=schema_version,
        artifact_id=artifact_id,
    )

    mission = _required_mapping(manifest, 'mission', 'manifest')
    expected_states = (
        [
            'OFF',
            'WAIT_FOR_SIGNAL',
            'BASE',
            'INITIAL_STOP',
            'SWITCH_TO_SHORTCUT',
            'SHORTCUT',
            'SWITCH_TO_BASE',
            'FAULT',
        ]
        if schema_version in {14, 15, 16, 17, 18, 19, 20}
        else [
            'OFF',
            'BASE',
            'RED_STOP',
            'SWITCH_TO_SHORTCUT',
            'SHORTCUT',
            'SWITCH_TO_BASE',
            'FAULT',
        ]
    )
    if mission.get('states') != expected_states:
        raise ArtifactContractError('mission state contract mismatch')
    expected_action_priority = (
        ['INITIAL_STOP', 'LEFT', 'STRAIGHT']
        if schema_version in {14, 15, 16, 17, 18, 19, 20}
        else ['STOP', 'LEFT', 'STRAIGHT']
    )
    if mission.get('action_priority') != expected_action_priority:
        raise ArtifactContractError('mission action priority mismatch')
    if schema_version == 1:
        if mission.get('transition_stop_control_cycles') != 1:
            raise ArtifactContractError(
                'mission transition stop must be one cycle'
            )
        shortcut_entry_stop_control_cycles = 1
        shortcut_exit_stop_control_cycles = 1
        base_shadow_enabled = False
        base_shadow_history_update = None
        base_shadow_max_age_sec = None
    else:
        transition = _required_mapping(mission, 'transition', 'mission')
        if (
            transition.get('shortcut_entry_stop_control_cycles') != 1
            or transition.get('shortcut_exit_stop_control_cycles') != 0
            or transition.get('shortcut_exit_command_source')
            != 'latest_base_shadow_prediction'
        ):
            raise ArtifactContractError(
                'schema v2/v3 shortcut transition contract mismatch'
            )
        shadow = _required_mapping(mission, 'base_shadow', 'mission')
        if (
            shadow.get('enabled_states') != ['SWITCH_TO_SHORTCUT', 'SHORTCUT']
            or shadow.get('history_seed')
            != 'active_base_history_before_entry_stop'
            or shadow.get('history_update') != 'capped_prediction_commands'
            or shadow.get('motor_publish_during_shortcut') is not False
            or shadow.get('failure_behavior') != 'fault_stop'
            or shadow.get('red_behavior')
            != (
                'ignore_after_initial_stop'
                if schema_version in {14, 15, 16, 17, 18, 19, 20}
                else 'discard'
            )
        ):
            raise ArtifactContractError(
                'schema v2/v3 Base shadow contract mismatch'
            )
        shortcut_entry_stop_control_cycles = 1
        shortcut_exit_stop_control_cycles = 0
        base_shadow_enabled = True
        base_shadow_history_update = 'capped_prediction_commands'
        base_shadow_max_age_sec = _exact_number(
            shadow,
            'stale_timeout_sec',
            0.50 if schema_version in {19, 20} else 0.25,
        )
    expected_red_cancels_shortcut = schema_version not in {
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    }
    if (
        mission.get('red_cancels_shortcut')
        is not expected_red_cancels_shortcut
    ):
        raise ArtifactContractError(
            'mission red cancellation contract mismatch'
        )
    if mission.get('red_cancel_consumes_success') is not False:
        raise ArtifactContractError(
            'red cancellation must not consume success'
        )
    expected_base_speed_cap = (
        35.0
        if schema_version in {11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
        else 25.0
    )
    base_speed_cap = _exact_number(
        mission,
        'base_speed_cap',
        expected_base_speed_cap,
    )
    if base_speed_cap > base.speed_output_max:
        raise ArtifactContractError(
            'Base speed cap exceeds the Base policy output maximum'
        )
    _exact_number(mission, 'a_hold_release_grace_sec', 0.12)
    shortcut_speed = _exact_number(mission, 'shortcut_speed', 23.0)
    shortcut_duration = _exact_number(
        mission,
        'shortcut_duration_sec',
        4.0 if schema_version == 20 else 8.0,
    )
    if mission.get('successful_shortcut_once') is not True:
        raise ArtifactContractError(
            'successful shortcut must be once per scope'
        )
    if schema_version in {19, 20}:
        if (
            mission.get('successful_shortcut_once_scope')
            != 'drive_gate_activation'
        ):
            raise ArtifactContractError(
                'schema v19/v20 shortcut scope must be drive_gate_activation'
            )
        successful_shortcut_once_scope = 'drive_gate_activation'
    else:
        if 'successful_shortcut_once_scope' in mission:
            raise ArtifactContractError(
                'legacy mission must not define shortcut once scope'
            )
        successful_shortcut_once_scope = 'process'
    if 5 <= schema_version <= 13:
        red_stop_yolo_missing_release_frames = _exact_int(
            mission,
            'red_stop_yolo_missing_release_frames',
            30,
        )
        if red_stop_yolo_missing_release_frames % inference_every != 0:
            raise ArtifactContractError(
                'YOLO missing release frames must align with detector cadence'
            )
    else:
        if 'red_stop_yolo_missing_release_frames' in mission:
            raise ArtifactContractError(
                'mission must not define YOLO missing-release'
            )
        red_stop_yolo_missing_release_frames = None
    if schema_version == 14:
        expected_initial_stop = {
            'gamepad_activation': 'lb_held_on_a_enable',
            'headless_activation': 'wait_for_first_valid_signal',
            'stop_consecutive_reads': 15,
            'clear_classes': ['STRAIGHT', 'LEFT'],
            'clear_consecutive_reads': 3,
            'clear_different_class_behavior': 'continue_candidate',
            'unknown_or_missing_behavior': (
                'reset_clear_candidate_retain_stop'
            ),
            'post_clear_action': 'BASE',
            'post_clear_stop_behavior': 'ignore',
            'ready_behavior': 'log_once_on_first_valid_class',
        }
        if mission.get('initial_stop') != expected_initial_stop:
            raise ArtifactContractError(
                'schema v14 initial-stop contract mismatch'
            )
        initial_stop_one_shot = True
        initial_stop_clear_consecutive_reads = 3
        headless_wait_for_first_signal = True
        initial_left_direct_shortcut = False
        control_vote_on_fresh_yolo_only = False
    elif schema_version in {15, 16, 17, 18, 19, 20}:
        clear_reads = {15: 5, 16: 3, 17: 1, 18: 1, 19: 1, 20: 1}[
            schema_version
        ]
        expected_initial_stop = {
            'gamepad_activation': 'lb_held_on_a_enable_wait_for_signal',
            'headless_activation': 'wait_for_first_valid_signal',
            'stop_consecutive_reads': 5,
            'clear_classes': ['STRAIGHT', 'LEFT'],
            'clear_consecutive_reads': clear_reads,
            'clear_different_class_behavior': 'restart_candidate_at_one',
            'unknown_or_missing_behavior': 'reset_candidate_retain_stop',
            'post_clear_action_by_class': {
                'STRAIGHT': 'BASE',
                'LEFT': 'SHORTCUT',
            },
            'post_clear_stop_behavior': 'ignore',
            'ready_behavior': 'log_once_on_first_valid_fresh_class',
        }
        if mission.get('initial_stop') != expected_initial_stop:
            raise ArtifactContractError(
                f'schema v{schema_version} initial-wait contract mismatch'
            )
        initial_stop_one_shot = True
        initial_stop_clear_consecutive_reads = clear_reads
        headless_wait_for_first_signal = True
        initial_left_direct_shortcut = True
        control_vote_on_fresh_yolo_only = True
    else:
        if 'initial_stop' in mission:
            raise ArtifactContractError(
                'legacy mission must not define initial_stop'
            )
        initial_stop_one_shot = False
        initial_stop_clear_consecutive_reads = None
        headless_wait_for_first_signal = False
        initial_left_direct_shortcut = False
        control_vote_on_fresh_yolo_only = False

    runtime = _required_mapping(manifest, 'host_runtime', 'manifest')
    if runtime.get('onnxruntime_version') != EXPECTED_ONNXRUNTIME_VERSION:
        raise ArtifactContractError('ONNX Runtime version contract mismatch')
    if runtime.get('numpy_version') != EXPECTED_NUMPY_VERSION:
        raise ArtifactContractError('NumPy version contract mismatch')
    if runtime.get('providers') != list(EXPECTED_PROVIDERS):
        raise ArtifactContractError('ONNX Runtime provider order mismatch')
    if runtime.get('policies_preloaded') is not True:
        raise ArtifactContractError('both policies must be preloaded')
    if runtime.get('shared_cuda_lock') is not True:
        raise ArtifactContractError('dual policies must share a CUDA lock')

    return TrafficShortcutBundle(
        root=root,
        artifact_id=artifact_id,
        schema_version=schema_version,
        base=base,
        shortcut=shortcut,
        detector=TrafficDetectorContract(
            model_path=signal_path,
            classifier_model_path=classifier_path,
            mode=mode,
            confidence_threshold=confidence,
            bbox_width_min=expected_bbox_width[0],
            bbox_width_max=expected_bbox_width[1],
            inference_every_n_frames=inference_every,
            classification_every_n_frames_after_detection=(
                classification_every
            ),
            reuse_detected_bbox_between_yolo_frames=(reuse_detected_bbox),
            percentile=percentile,
            red_index=red_index,
            left_indices=left_indices,
            straight_index=straight_index,
            red_consecutive_reads=red_consecutive_reads,
            left_consecutive_reads=left_consecutive_reads,
            straight_consecutive_reads=straight_consecutive_reads,
            classifier_crop_padding=classifier_crop_padding,
            detector_preprocessing=detector_preprocessing,
            classifier_input_height=classifier_input_height,
            classifier_input_width=classifier_input_width,
            classifier_classes=classifier_classes,
            classifier_probability_threshold=(
                classifier_probability_threshold
            ),
            classifier_interpolation=classifier_interpolation,
            red_stop_yolo_missing_release_frames=(
                red_stop_yolo_missing_release_frames
            ),
        ),
        base_speed_cap=base_speed_cap,
        shortcut_speed=shortcut_speed,
        shortcut_duration_sec=shortcut_duration,
        shortcut_entry_stop_control_cycles=(
            shortcut_entry_stop_control_cycles
        ),
        shortcut_exit_stop_control_cycles=(shortcut_exit_stop_control_cycles),
        base_shadow_enabled=base_shadow_enabled,
        base_shadow_history_update=base_shadow_history_update,
        base_shadow_max_age_sec=base_shadow_max_age_sec,
        successful_shortcut_once=True,
        successful_shortcut_once_scope=successful_shortcut_once_scope,
        action_priority=tuple(expected_action_priority),
        red_stop_yolo_missing_release_frames=(
            red_stop_yolo_missing_release_frames
        ),
        initial_stop_one_shot=initial_stop_one_shot,
        initial_stop_clear_consecutive_reads=(
            initial_stop_clear_consecutive_reads
        ),
        headless_wait_for_first_signal=headless_wait_for_first_signal,
        initial_left_direct_shortcut=initial_left_direct_shortcut,
        control_vote_on_fresh_yolo_only=(control_vote_on_fresh_yolo_only),
        onnxruntime_version=EXPECTED_ONNXRUNTIME_VERSION,
        numpy_version=EXPECTED_NUMPY_VERSION,
        providers=EXPECTED_PROVIDERS,
    )


def _load_signal_vote_contract(
    manifest: Mapping[str, object],
    *,
    schema_version: int,
    artifact_id: str,
) -> tuple[int, int, int]:
    if schema_version < 3:
        red_latch = _required_mapping(manifest, 'red_latch', 'manifest')
        if (
            red_latch.get('consecutive_red_reads') != 3
            or red_latch.get('unknown_behavior') != 'retain_latch'
            or red_latch.get('clear_actions') != ['LEFT', 'STRAIGHT']
            or 'signal_vote' in manifest
        ):
            raise ArtifactContractError('legacy red latch contract mismatch')
        return 3, 1, 1

    if schema_version == 3 and artifact_id not in {
        EXPECTED_SIGNAL_VOTE_BUNDLE_ID,
        EXPECTED_EXPANDED_SIGNAL_VOTE_BUNDLE_ID,
    }:
        raise ArtifactContractError('signal bundle id is not approved')
    if schema_version == 4 and artifact_id != EXPECTED_CLASSIFIER_BUNDLE_ID:
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 5
        and artifact_id != EXPECTED_YOLO_MISSING_RELEASE_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 6
        and artifact_id != EXPECTED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 7
        and artifact_id != EXPECTED_STABILIZED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 8
        and artifact_id != EXPECTED_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 9
        and artifact_id
        != EXPECTED_STOP10_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 10
        and artifact_id
        != EXPECTED_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 11
        and artifact_id
        != EXPECTED_SPEED35_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 12
        and artifact_id
        != EXPECTED_SPEED35_STOP10_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 13
        and artifact_id
        != EXPECTED_SPEED35_STOP15_GO15_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 14
        and artifact_id
        != EXPECTED_SPEED35_INITIAL_STOP_ONCE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 15
        and artifact_id
        != EXPECTED_SPEED35_INITIAL_WAIT_FRESH5_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 16
        and artifact_id
        != EXPECTED_SPEED35_INITIAL_WAIT_FRESH3_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 17
        and artifact_id
        != EXPECTED_SPEED35_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 18
        and artifact_id
        != EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 19
        and artifact_id
        != EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_SESSION_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 20
        and artifact_id
        != EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_SESSION_4S_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    signal_vote = _required_mapping(manifest, 'signal_vote', 'manifest')
    if schema_version in {17, 18, 19, 20}:
        expected = {
            'raw_classes': ['STOP', 'STRAIGHT', 'LEFT'],
            'consecutive_reads_by_raw_class': {
                'STOP': 5,
                'STRAIGHT': 1,
                'LEFT': 1,
            },
            'unknown_behavior': 'reset_candidate',
            'different_raw_class_behavior': 'restart_candidate_at_one',
            'stop_classes': ['STOP'],
            'stop_vote_behavior': 'only_while_initial_stop_armed',
            'post_initial_stop_behavior': 'ignore_stop',
            'navigation_actions': ['LEFT', 'STRAIGHT'],
            'control_vote_source': 'fresh_yolo_classifier_only',
            'cached_classifier_behavior': 'disabled',
        }
        if signal_vote != expected or 'red_latch' in manifest:
            raise ArtifactContractError(
                'schema v17..v20 classifier signal vote contract mismatch'
            )
        return 5, 1, 1
    if schema_version == 16:
        expected = {
            'raw_classes': ['STOP', 'STRAIGHT', 'LEFT'],
            'consecutive_reads_by_raw_class': {
                'STOP': 5,
                'STRAIGHT': 3,
                'LEFT': 3,
            },
            'unknown_behavior': 'reset_candidate',
            'different_raw_class_behavior': 'restart_candidate_at_one',
            'stop_classes': ['STOP'],
            'stop_vote_behavior': 'only_while_initial_stop_armed',
            'post_initial_stop_behavior': 'ignore_stop',
            'navigation_actions': ['LEFT', 'STRAIGHT'],
            'control_vote_source': 'fresh_yolo_classifier_only',
            'cached_classifier_behavior': 'disabled',
        }
        if signal_vote != expected or 'red_latch' in manifest:
            raise ArtifactContractError(
                'schema v16 classifier signal vote contract mismatch'
            )
        return 5, 3, 3
    if schema_version == 15:
        expected = {
            'raw_classes': ['STOP', 'STRAIGHT', 'LEFT'],
            'consecutive_reads_by_raw_class': {
                'STOP': 5,
                'STRAIGHT': 5,
                'LEFT': 5,
            },
            'unknown_behavior': 'reset_candidate',
            'different_raw_class_behavior': 'restart_candidate_at_one',
            'stop_classes': ['STOP'],
            'stop_vote_behavior': 'only_while_initial_stop_armed',
            'post_initial_stop_behavior': 'ignore_stop',
            'navigation_actions': ['LEFT', 'STRAIGHT'],
            'control_vote_source': 'fresh_yolo_classifier_only',
            'cached_classifier_behavior': 'diagnostics_only',
        }
        if signal_vote != expected or 'red_latch' in manifest:
            raise ArtifactContractError(
                'schema v15 classifier signal vote contract mismatch'
            )
        return 5, 5, 5
    if schema_version == 14:
        expected = {
            'raw_classes': ['STOP', 'STRAIGHT', 'LEFT'],
            'consecutive_reads_by_raw_class': {
                'STOP': 15,
                'STRAIGHT': 15,
                'LEFT': 15,
            },
            'unknown_behavior': 'reset_candidate',
            'different_raw_class_behavior': 'restart_candidate_at_one',
            'stop_classes': ['STOP'],
            'stop_vote_behavior': 'only_while_initial_stop_armed',
            'post_initial_stop_behavior': 'ignore_stop',
            'navigation_actions': ['LEFT', 'STRAIGHT'],
        }
        if signal_vote != expected or 'red_latch' in manifest:
            raise ArtifactContractError(
                'schema v14 classifier signal vote contract mismatch'
            )
        return 15, 15, 15
    if schema_version in {7, 8, 9, 10, 11, 12, 13}:
        if schema_version == 13:
            stop_reads, straight_reads, left_reads = 15, 15, 15
        elif schema_version == 12:
            stop_reads, straight_reads, left_reads = 10, 30, 30
        elif schema_version in {10, 11}:
            stop_reads, straight_reads, left_reads = 30, 30, 30
        elif schema_version == 9:
            stop_reads, straight_reads, left_reads = 10, 15, 15
        else:
            stop_reads, straight_reads, left_reads = 3, 15, 15
        expected = {
            'raw_classes': ['STOP', 'STRAIGHT', 'LEFT'],
            'consecutive_reads_by_raw_class': {
                'STOP': stop_reads,
                'STRAIGHT': straight_reads,
                'LEFT': left_reads,
            },
            'unknown_behavior': 'reset_candidate',
            'different_raw_class_behavior': 'restart_candidate_at_one',
            'stop_classes': ['STOP'],
            'stop_latch_behavior': 'retain_until_confirmed_go_action',
            'stop_clear_classes': ['LEFT', 'STRAIGHT'],
        }
        if signal_vote != expected or 'red_latch' in manifest:
            raise ArtifactContractError(
                'schema v7..v13 classifier signal vote contract mismatch'
            )
        return stop_reads, left_reads, straight_reads
    if schema_version == 6:
        expected = {
            'raw_classes': ['STOP', 'STRAIGHT', 'LEFT'],
            'consecutive_reads': 2,
            'unknown_behavior': 'reset_candidate',
            'different_raw_class_behavior': 'restart_candidate_at_one',
            'stop_classes': ['STOP'],
            'stop_latch_behavior': 'retain_until_confirmed_go_action',
            'stop_clear_classes': ['LEFT', 'STRAIGHT'],
        }
        if signal_vote != expected or 'red_latch' in manifest:
            raise ArtifactContractError(
                'schema v6 classifier signal vote contract mismatch'
            )
        return 2, 2, 2
    if schema_version >= 4:
        expected = {
            'raw_classes': [
                'red',
                'yellow',
                'left_green',
                'straight_green',
            ],
            'consecutive_reads': 2,
            'unknown_behavior': 'reset_candidate',
            'different_raw_class_behavior': 'restart_candidate_at_one',
            'stop_classes': ['red', 'yellow'],
            'stop_latch_behavior': 'retain_until_confirmed_green_class',
            'stop_clear_classes': ['left_green', 'straight_green'],
        }
        if signal_vote != expected or 'red_latch' in manifest:
            raise ArtifactContractError(
                'schema v4/v5 classifier signal vote contract mismatch'
            )
        return 2, 2, 2
    expected = {
        'actions': ['RED', 'LEFT', 'STRAIGHT'],
        'consecutive_reads': 5,
        'unknown_behavior': 'reset_candidate',
        'different_action_behavior': 'restart_candidate_at_one',
        'red_latch_behavior': 'retain_until_confirmed_clear_action',
        'red_clear_actions': ['LEFT', 'STRAIGHT'],
    }
    if signal_vote != expected or 'red_latch' in manifest:
        raise ArtifactContractError('schema v3 signal vote contract mismatch')
    return 5, 5, 5


def _load_classifier_contract(
    root: Path,
    components: Mapping[str, object],
    detector: Mapping[str, object],
    *,
    schema_version: int,
) -> tuple[
    Path,
    float,
    int,
    int,
    tuple[str, ...],
    float | None,
    str,
]:
    classifier = _required_mapping(
        components,
        'traffic_classifier',
        'components',
    )
    relative = _required_string(classifier, 'file', 'traffic_classifier')
    _validate_relative_path(relative)
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ArtifactContractError(
            'traffic classifier ONNX is missing or unsafe'
        )
    expected_sha256 = (
        HUMAN_BBOX_TRAFFIC_CLASSIFIER_SHA256
        if schema_version
        in {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
        else TRAFFIC_CLASSIFIER_SHA256
    )
    if (
        classifier.get('format') != 'onnx'
        or classifier.get('sha256') != expected_sha256
        or _sha256_file(path) != expected_sha256
    ):
        raise ArtifactContractError(
            'traffic classifier ONNX checksum mismatch'
        )
    if schema_version in {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}:
        input_height = 128
        input_width = 416
        classes = ('STOP', 'STRAIGHT', 'LEFT')
        interpolation = 'pillow_bilinear_antialias'
        decision = 'softmax_argmax_min_probability'
        probability_threshold = 0.5
    else:
        input_height = 48
        input_width = 96
        classes = ('red', 'yellow', 'left_green', 'straight_green')
        interpolation = 'area'
        decision = 'softmax_argmax_without_threshold'
        probability_threshold = None
    if (
        classifier.get('input')
        != {
            'name': 'image',
            'dtype': 'float32',
            'shape': [1, 3, input_height, input_width],
        }
        or classifier.get('output')
        != {
            'name': 'logits',
            'dtype': 'float32',
            'shape': [1, len(classes)],
        }
        or classifier.get('classes') != list(classes)
    ):
        raise ArtifactContractError(
            'traffic classifier ONNX contract mismatch'
        )
    classifier_detector = _required_mapping(detector, 'classifier', 'detector')
    if (
        classifier_detector.get('resize_width') != input_width
        or classifier_detector.get('resize_height') != input_height
        or classifier_detector.get('interpolation') != interpolation
        or classifier_detector.get('color_space') != 'RGB'
        or classifier_detector.get('decision') != decision
        or classifier_detector.get('normalization')
        != {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        }
    ):
        raise ArtifactContractError(
            'traffic classifier preprocessing mismatch'
        )
    padding = _exact_number(
        classifier_detector,
        'crop_padding_fraction',
        0.15,
    )
    if probability_threshold is not None:
        _exact_number(
            classifier_detector,
            'minimum_probability',
            probability_threshold,
        )
    elif 'minimum_probability' in classifier_detector:
        raise ArtifactContractError(
            'legacy traffic classifier must not set minimum probability'
        )
    return (
        path,
        padding,
        input_height,
        input_width,
        classes,
        probability_threshold,
        interpolation,
    )


def _load_component(
    root: Path,
    contract: Mapping[str, object],
    name: str,
) -> PolicyArtifact:
    relative = _required_string(contract, 'directory', name)
    _validate_relative_path(relative)
    component_root = root / relative
    expected_relative = f'policies/{contract.get("artifact_id")}'
    if relative != expected_relative:
        raise ArtifactContractError(
            f'{name} component directory must be {expected_relative}'
        )
    artifact = load_policy_artifact(component_root)
    if contract.get('artifact_id') != artifact.artifact_id:
        raise ArtifactContractError(f'{name} component artifact id mismatch')
    digest = contract.get('source_sha256s_sha256')
    if (
        not isinstance(digest, str)
        or not re.fullmatch(r'[0-9a-f]{64}', digest)
        or _sha256_file(component_root / BUNDLE_CHECKSUMS) != digest
    ):
        raise ArtifactContractError(
            f'{name} source checksum identity mismatch'
        )
    return artifact


def _expected_shortcut_artifact_id(
    *,
    schema_version: int,
    artifact_id: str,
) -> str:
    if schema_version < 3:
        return EXPECTED_SHORTCUT_ARTIFACT_ID
    if schema_version == 3 and artifact_id not in {
        EXPECTED_SIGNAL_VOTE_BUNDLE_ID,
        EXPECTED_EXPANDED_SIGNAL_VOTE_BUNDLE_ID,
    }:
        raise ArtifactContractError('signal bundle id is not approved')
    if schema_version == 4 and artifact_id != EXPECTED_CLASSIFIER_BUNDLE_ID:
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 5
        and artifact_id != EXPECTED_YOLO_MISSING_RELEASE_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 6
        and artifact_id != EXPECTED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 7
        and artifact_id != EXPECTED_STABILIZED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 8
        and artifact_id != EXPECTED_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 9
        and artifact_id
        != EXPECTED_STOP10_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 10
        and artifact_id
        != EXPECTED_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 11
        and artifact_id
        != EXPECTED_SPEED35_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 12
        and artifact_id
        != EXPECTED_SPEED35_STOP10_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 13
        and artifact_id
        != EXPECTED_SPEED35_STOP15_GO15_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 14
        and artifact_id
        != EXPECTED_SPEED35_INITIAL_STOP_ONCE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    if (
        schema_version == 15
        and artifact_id
        != EXPECTED_SPEED35_INITIAL_WAIT_FRESH5_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
    ):
        raise ArtifactContractError('signal bundle id is not approved')
    try:
        return EXPECTED_SIGNAL_BUNDLE_SHORTCUT_IDS[artifact_id]
    except KeyError as exc:
        raise ArtifactContractError(
            'signal bundle id is not approved'
        ) from exc


def _expected_base_artifact_id(
    *,
    schema_version: int,
    artifact_id: str,
) -> str:
    if schema_version in {11, 12, 13, 14, 15, 16, 17, 18, 19, 20}:
        expected_artifact_id = {
            11: EXPECTED_SPEED35_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
            12: EXPECTED_SPEED35_STOP10_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
            13: EXPECTED_SPEED35_STOP15_GO15_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
            14: EXPECTED_SPEED35_INITIAL_STOP_ONCE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
            15: EXPECTED_SPEED35_INITIAL_WAIT_FRESH5_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
            16: EXPECTED_SPEED35_INITIAL_WAIT_FRESH3_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
            17: EXPECTED_SPEED35_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
            18: EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
            19: EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_SESSION_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
            20: EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_SESSION_4S_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
        }[schema_version]
        if artifact_id != expected_artifact_id:
            raise ArtifactContractError('signal bundle id is not approved')
        return (
            EXPECTED_SPEED35_FIX_BASE_ARTIFACT_ID
            if schema_version in {18, 19, 20}
            else EXPECTED_SPEED35_BASE_ARTIFACT_ID
        )
    return EXPECTED_BASE_ARTIFACT_ID


def _validate_policy_contracts(
    base: PolicyArtifact,
    shortcut: PolicyArtifact,
    base_contract: Mapping[str, object],
    shortcut_contract: Mapping[str, object],
    *,
    expected_base_artifact_id: str,
    expected_shortcut_artifact_id: str,
) -> None:
    if base.artifact_id != expected_base_artifact_id:
        raise ArtifactContractError('base artifact id is not approved')
    if shortcut.artifact_id != expected_shortcut_artifact_id:
        raise ArtifactContractError('shortcut artifact id is not approved')
    if base_contract.get('schema_version') != 6:
        raise ArtifactContractError('base policy schema must be 6')
    if shortcut_contract.get('schema_version') != 7:
        raise ArtifactContractError('shortcut policy schema must be 7')
    expected_speed_output_max = (
        35.0
        if expected_base_artifact_id
        in {
            EXPECTED_SPEED35_BASE_ARTIFACT_ID,
            EXPECTED_SPEED35_FIX_BASE_ARTIFACT_ID,
        }
        else 30.0
    )
    expected_initial_speed = 35 if expected_speed_output_max == 35.0 else 25
    if (
        base.prediction_mode != CONTINUOUS_REGRESSION_PREDICTION_MODE
        or base.control_encoding != COMPACT_CONTROL_ENCODING
        or base.history is None
        or base.history.update != 'externally_executed_commands'
        or base.speed_output_max != expected_speed_output_max
        or base.history.speed_output_max != int(expected_speed_output_max)
        or base.history.initial_class_ids != (50, 50 + expected_initial_speed)
    ):
        raise ArtifactContractError('base policy AR contract mismatch')
    if (
        shortcut.prediction_mode
        != ANGLE_REGRESSION_FIXED_SPEED_PREDICTION_MODE
        or shortcut.history is not None
        or shortcut.fixed_speed != 23.0
        or shortcut.speed_normalization_divisor != 25.0
    ):
        raise ArtifactContractError('shortcut fixed-speed contract mismatch')
    if (
        base.steering_contract != NORMALIZED_STEERING_CONTRACT
        or shortcut.steering_contract != NORMALIZED_STEERING_CONTRACT
    ):
        raise ArtifactContractError('policy steering contract mismatch')
    if base.road_warp is None or shortcut.road_warp is None:
        raise ArtifactContractError('both policies require road warp')
    if base.road_warp != shortcut.road_warp:
        raise ArtifactContractError('Base and shortcut road warp must match')
    if base.image_size != 224 or shortcut.image_size != 224:
        raise ArtifactContractError('both policy inputs must be 224 square')


def _safe_root(value: str | Path) -> Path:
    root = Path(value).expanduser()
    if root.is_symlink():
        raise ArtifactContractError('bundle root must not be a symlink')
    root = root.resolve()
    if not root.is_dir():
        raise ArtifactContractError(f'bundle directory is missing: {root}')
    return root


def _verify_bundle_checksums(root: Path) -> None:
    checksum_path = root / BUNDLE_CHECKSUMS
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise ArtifactContractError('bundle SHA256SUMS is missing or unsafe')
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding='utf-8').splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ArtifactContractError('invalid bundle SHA256SUMS line')
        digest, relative = parts
        relative = relative.removeprefix('*')
        _validate_relative_path(relative)
        if not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ArtifactContractError('invalid bundle SHA-256 digest')
        if relative in expected:
            raise ArtifactContractError(
                f'duplicate bundle checksum: {relative}'
            )
        expected[relative] = digest
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ArtifactContractError(f'bundle contains a symlink: {path}')
        if not path.is_dir() and not path.is_file():
            raise ArtifactContractError(
                f'bundle contains a special file: {path}'
            )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob('*')
        if path.is_file() and path != checksum_path
    }
    if actual != set(expected):
        raise ArtifactContractError('bundle SHA256SUMS file list mismatch')
    for relative, digest in expected.items():
        if _sha256_file(root / relative) != digest:
            raise ArtifactContractError(
                f'bundle checksum mismatch: {relative}'
            )


def _load_mapping(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactContractError(
            f'missing or unsafe bundle manifest: {path}'
        )
    try:
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    except yaml.YAMLError as exc:
        raise ArtifactContractError('bundle manifest YAML is invalid') from exc
    if not isinstance(payload, Mapping):
        raise ArtifactContractError('bundle manifest must contain a mapping')
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
        raise ArtifactContractError(f'{context}.{key} must be a string')
    return value


def _exact_number(
    payload: Mapping[str, object],
    key: str,
    expected: float,
) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactContractError(f'{key} must be numeric')
    result = float(value)
    if not math.isfinite(result) or result != expected:
        raise ArtifactContractError(f'{key} must be {expected:g}')
    return result


def _exact_int(
    payload: Mapping[str, object],
    key: str,
    expected: int,
) -> int:
    value = payload.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value != expected
    ):
        raise ArtifactContractError(f'{key} must be {expected}')
    return value


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or '..' in path.parts
        or not re.fullmatch(r'[A-Za-z0-9._/-]+', value)
    ):
        raise ArtifactContractError(f'unsafe bundle path: {value}')


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
