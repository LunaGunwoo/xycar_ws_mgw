"""Atomically assemble the approved Base/ResNet18/traffic ONNX bundle."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import yaml

BUNDLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
BUNDLE_KIND = "traffic_shortcut_bundle"
LEGACY_BUNDLE_ID = "traffic-shortcut-nice-regression-resnet18-8s-20260821"
SHADOW_BUNDLE_ID = (
    "traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-"
    "20260821"
)
SIGNAL_VOTE_BUNDLE_ID = (
    "traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-"
    "tl45-votes5-every3-20260821"
)
BASE_ID = (
    "front-cam-policy-vit-small-ar4-v2-nice-adaptive-joint-regression-"
    "sequence-init25-window5-20260821"
)
SHORTCUT_ID = "nice-shortcut-resnet18-squarewarp-speed23-20260821"
EXPANDED_SHORTCUT_ID = (
    "nice-shortcut-resnet18-squarewarp-speed23-45sessions-20260821"
)
BUNDLE_ID = (
    "traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-"
    "tl45-votes5-every3-45sessions-20260821"
)
CLASSIFIER_BUNDLE_ID = (
    "traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-"
    "yolo-cls-tl45-votes2-every3-45sessions-20260822"
)
YOLO_MISSING_RELEASE_BUNDLE_ID = (
    "traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-"
    "yolo-cls-tl45-votes2-every3-yolo-miss30-release-45sessions-20260822"
)
HUMAN_BBOX_CLASSIFIER_BUNDLE_ID = (
    "traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-"
    "yolo11s-humanbbox-cnn416-actions3-conf50-tl40to225-votes2-every3-"
    "yolo-miss30-release-45sessions-20260823"
)
BUNDLE_CONTRACTS = {
    LEGACY_BUNDLE_ID: (1, 3, SHORTCUT_ID),
    SHADOW_BUNDLE_ID: (2, 3, SHORTCUT_ID),
    SIGNAL_VOTE_BUNDLE_ID: (3, 5, SHORTCUT_ID),
    BUNDLE_ID: (3, 5, EXPANDED_SHORTCUT_ID),
    CLASSIFIER_BUNDLE_ID: (4, 2, EXPANDED_SHORTCUT_ID),
    YOLO_MISSING_RELEASE_BUNDLE_ID: (5, 2, EXPANDED_SHORTCUT_ID),
    HUMAN_BBOX_CLASSIFIER_BUNDLE_ID: (6, 2, EXPANDED_SHORTCUT_ID),
}
TRAFFIC_SHA256 = (
    "24c1a38eacfb065c95e5577be29a2a542b985d6ea1954bf2fc94c52eb674aa41"
)
CLASSIFIER_SHA256 = (
    "3ddb126483e03211f59cd15eb1f96632b9c3d5ac0d42544ccd999601643b806b"
)
HUMAN_BBOX_TRAFFIC_SHA256 = (
    "7d1bd24a025c6b7851c396e9e0cbc38dfad6f6e852c712e2cebdf8547a428e64"
)
HUMAN_BBOX_CLASSIFIER_SHA256 = (
    "e126f4f3036bcd4e44ab0ca4b5cc70a2e46f87bc859ae434719eb5f08de122b5"
)
HUMAN_BBOX_TRAFFIC_CHECKPOINT_SHA256 = (
    "4a7f326e574fe6332d062345bf44d6f04e85b77989b6eac00016f2a8341e1dc8"
)
HUMAN_BBOX_CLASSIFIER_CHECKPOINT_SHA256 = (
    "b0f60032913456ba1551260c12e349194a00bc0c8279d770c996195ef8827000"
)
HUMAN_BBOX_DATASET_MANIFEST_SHA256 = (
    "cdff1b246aecd0fa0fa8b0cd51f13a4cd0a538f06aa312a54925a002a3486746"
)
YOLO_REFERENCE_SHA256 = (
    "50b2071a765d74add0c51bd91c3392d3ce8d08e24302dbe4f9389809b3ee7262"
)
SHORTCUT_REFERENCE_SHA256 = (
    "389ad9074a97ecb8ab65fa3ebb1de0a37d6a8cf1729b2a9364c8db76c3ee9d06"
)


class TrafficBundleBuildError(ValueError):
    """Raised when a component cannot safely enter the mission bundle."""


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the versioned traffic shortcut mission bundle"
    )
    parser.add_argument("--base-artifact", required=True)
    parser.add_argument("--shortcut-artifact", required=True)
    parser.add_argument("--traffic-model", required=True)
    parser.add_argument("--traffic-classifier")
    parser.add_argument("--artifact-id", default=BUNDLE_ID)
    parser.add_argument("--output-root", default="artifacts/models")
    arguments = parser.parse_args(argv)
    result = build_traffic_shortcut_bundle(
        base_artifact=Path(arguments.base_artifact),
        shortcut_artifact=Path(arguments.shortcut_artifact),
        traffic_model=Path(arguments.traffic_model),
        traffic_classifier=(
            Path(arguments.traffic_classifier)
            if arguments.traffic_classifier is not None
            else None
        ),
        artifact_id=arguments.artifact_id,
        output_root=Path(arguments.output_root),
    )
    print(result)


def build_traffic_shortcut_bundle(
    *,
    base_artifact: Path,
    shortcut_artifact: Path,
    traffic_model: Path,
    traffic_classifier: Path | None,
    artifact_id: str,
    output_root: Path,
) -> Path:
    if not BUNDLE_ID_PATTERN.fullmatch(artifact_id):
        raise TrafficBundleBuildError(f"invalid bundle id: {artifact_id}")
    bundle_contract = BUNDLE_CONTRACTS.get(artifact_id)
    if bundle_contract is None:
        raise TrafficBundleBuildError(
            "approved bundle id must be one of "
            + ", ".join(BUNDLE_CONTRACTS)
        )
    (
        schema_version,
        consecutive_signal_reads,
        shortcut_artifact_id,
    ) = bundle_contract
    base_artifact = _verify_policy_artifact(
        base_artifact,
        expected_id=BASE_ID,
        expected_schema=6,
    )
    shortcut_artifact = _verify_policy_artifact(
        shortcut_artifact,
        expected_id=shortcut_artifact_id,
        expected_schema=7,
    )
    traffic_model = traffic_model.expanduser()
    if traffic_model.is_symlink():
        raise TrafficBundleBuildError("traffic ONNX must not be a symlink")
    traffic_model = traffic_model.resolve()
    if not traffic_model.is_file():
        raise TrafficBundleBuildError(f"traffic ONNX is missing: {traffic_model}")
    expected_traffic_sha256 = (
        HUMAN_BBOX_TRAFFIC_SHA256 if schema_version == 6 else TRAFFIC_SHA256
    )
    expected_classifier_sha256 = (
        HUMAN_BBOX_CLASSIFIER_SHA256
        if schema_version == 6
        else CLASSIFIER_SHA256
    )
    if _sha256_file(traffic_model) != expected_traffic_sha256:
        raise TrafficBundleBuildError("traffic ONNX SHA-256 mismatch")
    if schema_version >= 4:
        if traffic_classifier is None:
            raise TrafficBundleBuildError("traffic classifier is required")
        traffic_classifier = _verify_onnx_file(
            traffic_classifier,
            expected_sha256=expected_classifier_sha256,
            label="traffic classifier ONNX",
        )
    elif traffic_classifier is not None:
        raise TrafficBundleBuildError(
            "traffic classifier is only valid for schema v4/v5/v6"
        )

    base_manifest = _load_mapping(base_artifact / "manifest.yaml")
    shortcut_manifest = _load_mapping(shortcut_artifact / "manifest.yaml")
    _validate_policy_manifests(
        base_manifest,
        shortcut_manifest,
        expected_shortcut_id=shortcut_artifact_id,
    )

    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / artifact_id
    if final.exists() or final.is_symlink():
        raise TrafficBundleBuildError(f"bundle already exists: {final}")
    temporary = output_root / f".incoming-{artifact_id}-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise TrafficBundleBuildError(f"temporary bundle exists: {temporary}")
    temporary.mkdir(mode=0o755)
    try:
        policies_dir = temporary / "policies"
        policies_dir.mkdir()
        _copy_artifact(base_artifact, policies_dir / BASE_ID)
        _copy_artifact(
            shortcut_artifact,
            policies_dir / shortcut_artifact_id,
        )
        signal_dir = temporary / "signal"
        signal_dir.mkdir()
        shutil.copy2(traffic_model, signal_dir / "traffic_light.onnx")
        if traffic_classifier is not None:
            shutil.copy2(traffic_classifier, signal_dir / "tl_cls.onnx")
        provenance_dir = temporary / "provenance"
        provenance_dir.mkdir()
        provenance = {
            "schema_version": 1,
            "source_references": {
                "team_shortcut_py": {
                    "read_only_path": "/home/xytron/xycar_ws_minju/src/track_drive/track_drive/shortcut.py",
                    "sha256": SHORTCUT_REFERENCE_SHA256,
                    "runtime_dependency": False,
                },
                "team_yolo_tl_py": {
                    "read_only_path": "/home/xytron/yolo_tl.py",
                    "sha256": YOLO_REFERENCE_SHA256,
                    "runtime_dependency": False,
                },
                "traffic_light_onnx": {
                    "read_only_path": (
                        str(traffic_model)
                        if schema_version == 6
                        else "/home/xytron/traffic_light.onnx"
                    ),
                    "sha256": expected_traffic_sha256,
                },
                "tl_cls_onnx": {
                    "read_only_path": (
                        str(traffic_classifier)
                        if schema_version == 6
                        and traffic_classifier is not None
                        else "/home/xytron/tl_cls.onnx"
                    ),
                    "sha256": expected_classifier_sha256,
                },
            },
        }
        if schema_version == 6:
            provenance["human_corrected_two_stage"] = {
                "detector_checkpoint_sha256": (
                    HUMAN_BBOX_TRAFFIC_CHECKPOINT_SHA256
                ),
                "classifier_checkpoint_sha256": (
                    HUMAN_BBOX_CLASSIFIER_CHECKPOINT_SHA256
                ),
                "dataset_manifest_sha256": (
                    HUMAN_BBOX_DATASET_MANIFEST_SHA256
                ),
                "detector_export": {
                    "format": "onnx",
                    "opset": 17,
                    "static_shape": True,
                    "simplified": False,
                },
                "classifier_export": {
                    "format": "onnx",
                    "opset": 17,
                    "static_shape": True,
                },
                "development_only": True,
                "known_scene_leakage": True,
            }
        (provenance_dir / "references.yaml").write_text(
            yaml.safe_dump(provenance, sort_keys=False),
            encoding="utf-8",
        )
        manifest = _bundle_manifest(
            artifact_id=artifact_id,
            schema_version=schema_version,
            consecutive_signal_reads=consecutive_signal_reads,
            base_artifact=base_artifact,
            shortcut_artifact=shortcut_artifact,
            shortcut_artifact_id=shortcut_artifact_id,
            traffic_classifier=traffic_classifier,
        )
        (temporary / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )
        _write_checksums(temporary)
        _verify_built_bundle(
            temporary,
            expected_id=artifact_id,
            expected_schema=schema_version,
            expected_signal_reads=consecutive_signal_reads,
            expected_shortcut_id=shortcut_artifact_id,
            expected_classifier=traffic_classifier is not None,
        )
        temporary.rename(final)
    except BaseException:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise
    return final


def _bundle_manifest(
    *,
    artifact_id: str,
    schema_version: int,
    consecutive_signal_reads: int,
    base_artifact: Path,
    shortcut_artifact: Path,
    shortcut_artifact_id: str,
    traffic_classifier: Path | None,
) -> dict[str, object]:
    mission = {
        "states": [
            "OFF",
            "BASE",
            "RED_STOP",
            "SWITCH_TO_SHORTCUT",
            "SHORTCUT",
            "SWITCH_TO_BASE",
            "FAULT",
        ],
        "action_priority": ["STOP", "LEFT", "STRAIGHT"],
        "a_hold_release_grace_sec": 0.12,
        "shortcut_duration_sec": 8.0,
        "successful_shortcut_once": True,
        "red_cancels_shortcut": True,
        "red_cancel_consumes_success": False,
        "base_speed_cap": 25.0,
        "shortcut_speed": 23.0,
    }
    if schema_version >= 5:
        mission["red_stop_yolo_missing_release_frames"] = 30
    if schema_version == 1:
        mission["transition_stop_control_cycles"] = 1
    else:
        mission["transition"] = {
            "shortcut_entry_stop_control_cycles": 1,
            "shortcut_exit_stop_control_cycles": 0,
            "shortcut_exit_command_source": (
                "latest_base_shadow_prediction"
            ),
        }
        mission["base_shadow"] = {
            "enabled_states": ["SWITCH_TO_SHORTCUT", "SHORTCUT"],
            "history_seed": "active_base_history_before_entry_stop",
            "history_update": "capped_prediction_commands",
            "motor_publish_during_shortcut": False,
            "stale_timeout_sec": 0.25,
            "failure_behavior": "fault_stop",
            "red_behavior": "discard",
        }
    manifest = {
        "schema_version": schema_version,
        "artifact_kind": BUNDLE_KIND,
        "artifact_id": artifact_id,
        "created_at": datetime.now(UTC).isoformat(),
        "components": {
            "base": {
                "directory": f"policies/{BASE_ID}",
                "artifact_id": BASE_ID,
                "schema_version": 6,
                "source_sha256s_sha256": _sha256_file(
                    base_artifact / "SHA256SUMS"
                ),
            },
            "shortcut": {
                "directory": f"policies/{shortcut_artifact_id}",
                "artifact_id": shortcut_artifact_id,
                "schema_version": 7,
                "source_sha256s_sha256": _sha256_file(
                    shortcut_artifact / "SHA256SUMS"
                ),
            },
            "traffic_light": {
                "format": "onnx",
                "file": "signal/traffic_light.onnx",
                "sha256": (
                    HUMAN_BBOX_TRAFFIC_SHA256
                    if schema_version == 6
                    else TRAFFIC_SHA256
                ),
                "input": {
                    "name": "images",
                    "dtype": "float32",
                    "shape": [1, 3, 640, 640],
                },
                "output": {
                    "name": "output0",
                    "dtype": "float32",
                    "shape": [1, 5, 8400],
                },
            },
        },
        "detector": {
            "confidence_threshold": 0.25,
            "bbox_width_px": [45, 200],
            "inference_every_n_frames": 3,
            "preprocessing": "resize_640_bgr_to_rgb_float32_nchw_div255",
            "selection": "maximum_confidence_box",
            "lamp": {
                "count": 4,
                "score": "hsv_v_percentile",
                "percentile": 80.0,
                "relative_threshold": "(min + max) / 2",
                "red_index": 0,
                "left_indices": [2, 3],
                "straight_index": 3,
                "straight_requires_red_off": True,
            },
        },
        "mission": mission,
        "host_runtime": {
            "onnxruntime_version": "1.24.0",
            "numpy_version": "1.26.4",
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "policies_preloaded": True,
            "shared_cuda_lock": True,
        },
    }
    if schema_version >= 4:
        assert traffic_classifier is not None
        if schema_version == 6:
            classifier_sha256 = HUMAN_BBOX_CLASSIFIER_SHA256
            classifier_height = 128
            classifier_width = 416
            classifier_classes = ["STOP", "STRAIGHT", "LEFT"]
            classifier_interpolation = "pillow_bilinear_antialias"
            classifier_decision = "softmax_argmax_min_probability"
        else:
            classifier_sha256 = CLASSIFIER_SHA256
            classifier_height = 48
            classifier_width = 96
            classifier_classes = [
                "red",
                "yellow",
                "left_green",
                "straight_green",
            ]
            classifier_interpolation = "area"
            classifier_decision = "softmax_argmax_without_threshold"
        manifest["components"]["traffic_classifier"] = {
            "format": "onnx",
            "file": "signal/tl_cls.onnx",
            "sha256": classifier_sha256,
            "input": {
                "name": "image",
                "dtype": "float32",
                "shape": [1, 3, classifier_height, classifier_width],
            },
            "output": {
                "name": "logits",
                "dtype": "float32",
                "shape": [1, len(classifier_classes)],
            },
            "classes": classifier_classes,
        }
        manifest["detector"] = {
            "confidence_threshold": 0.25,
            "bbox_width_px": (
                [40, 225] if schema_version == 6 else [45, 200]
            ),
            "inference_every_n_frames": 3,
            "preprocessing": (
                "letterbox_640_center_pad114_bgr_to_rgb_float32_nchw_div255"
                if schema_version == 6
                else "resize_640_bgr_to_rgb_float32_nchw_div255"
            ),
            "selection": "maximum_confidence_box",
            "classifier": {
                "crop_padding_fraction": 0.15,
                "resize_width": classifier_width,
                "resize_height": classifier_height,
                "interpolation": classifier_interpolation,
                "color_space": "RGB",
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "decision": classifier_decision,
            },
        }
        if schema_version == 6:
            manifest["detector"]["max_detections"] = 1
            manifest["detector"]["classifier"]["minimum_probability"] = 0.5
    if schema_version < 3:
        manifest["red_latch"] = _legacy_red_latch_contract()
    elif schema_version == 3:
        manifest["signal_vote"] = _signal_vote_contract(
            consecutive_signal_reads
        )
    else:
        manifest["signal_vote"] = _classifier_signal_vote_contract(
            consecutive_signal_reads,
            schema_version=schema_version,
        )
    return manifest


def _legacy_red_latch_contract() -> dict[str, object]:
    return {
        "consecutive_red_reads": 3,
        "unknown_behavior": "retain_latch",
        "clear_actions": ["LEFT", "STRAIGHT"],
    }


def _signal_vote_contract(consecutive_reads: int) -> dict[str, object]:
    return {
        "actions": ["RED", "LEFT", "STRAIGHT"],
        "consecutive_reads": consecutive_reads,
        "unknown_behavior": "reset_candidate",
        "different_action_behavior": "restart_candidate_at_one",
        "red_latch_behavior": "retain_until_confirmed_clear_action",
        "red_clear_actions": ["LEFT", "STRAIGHT"],
    }


def _classifier_signal_vote_contract(
    consecutive_reads: int,
    *,
    schema_version: int,
) -> dict[str, object]:
    if schema_version == 6:
        return {
            "raw_classes": ["STOP", "STRAIGHT", "LEFT"],
            "consecutive_reads": consecutive_reads,
            "unknown_behavior": "reset_candidate",
            "different_raw_class_behavior": "restart_candidate_at_one",
            "stop_classes": ["STOP"],
            "stop_latch_behavior": "retain_until_confirmed_go_action",
            "stop_clear_classes": ["LEFT", "STRAIGHT"],
        }
    return {
        "raw_classes": [
            "red",
            "yellow",
            "left_green",
            "straight_green",
        ],
        "consecutive_reads": consecutive_reads,
        "unknown_behavior": "reset_candidate",
        "different_raw_class_behavior": "restart_candidate_at_one",
        "stop_classes": ["red", "yellow"],
        "stop_latch_behavior": "retain_until_confirmed_green_class",
        "stop_clear_classes": ["left_green", "straight_green"],
    }


def _validate_policy_manifests(
    base: Mapping[str, object],
    shortcut: Mapping[str, object],
    *,
    expected_shortcut_id: str,
) -> None:
    if base.get("schema_version") != 6 or base.get("artifact_id") != BASE_ID:
        raise TrafficBundleBuildError("base manifest contract mismatch")
    if (
        shortcut.get("schema_version") != 7
        or shortcut.get("artifact_id") != expected_shortcut_id
    ):
        raise TrafficBundleBuildError("shortcut manifest contract mismatch")
    base_model = _required_mapping(base, "model", "base")
    shortcut_model = _required_mapping(shortcut, "model", "shortcut")
    if (
        base_model.get("prediction_mode") != "continuous_regression"
        or base_model.get("control_encoding") != "driver_compact_v2"
    ):
        raise TrafficBundleBuildError("base regression contract mismatch")
    history = _required_mapping(base, "history", "base")
    if history.get("update") != "externally_executed_commands":
        raise TrafficBundleBuildError("base must use executed AR history")
    runtime = _required_mapping(shortcut, "runtime", "shortcut")
    if (
        shortcut_model.get("prediction_mode") != "angle_regression_fixed_speed"
        or float(runtime.get("fixed_speed", -1.0)) != 23.0
        or float(runtime.get("speed_normalization_divisor", -1.0)) != 25.0
    ):
        raise TrafficBundleBuildError("shortcut fixed-speed contract mismatch")
    preprocessing_contracts = []
    for name, manifest in (("base", base), ("shortcut", shortcut)):
        steering = _required_mapping(manifest, "steering_contract", name)
        if steering.get("name") != "normalized_percent_v2":
            raise TrafficBundleBuildError(f"{name} steering contract mismatch")
        preprocessing = _required_mapping(manifest, "preprocessing", name)
        if (
            preprocessing.get("geometry")
            != "perspective_road_warp_then_bicubic_resize"
            or preprocessing.get("image_size") != 224
        ):
            raise TrafficBundleBuildError(f"{name} square warp contract mismatch")
        preprocessing_contracts.append(
            _required_mapping(preprocessing, "road_warp", name)
        )
    if preprocessing_contracts[0] != preprocessing_contracts[1]:
        raise TrafficBundleBuildError("Base and shortcut road warp must match")


def _verify_policy_artifact(
    path: Path,
    *,
    expected_id: str,
    expected_schema: int,
    require_directory_id: bool = True,
) -> Path:
    root = path.expanduser()
    if root.is_symlink():
        raise TrafficBundleBuildError("policy artifact must not be a symlink")
    root = root.resolve()
    if not root.is_dir() or (require_directory_id and root.name != expected_id):
        raise TrafficBundleBuildError(f"unexpected policy artifact: {root}")
    _verify_checksums(root, include_nested_checksum_files=False)
    manifest = _load_mapping(root / "manifest.yaml")
    if (
        manifest.get("artifact_id") != expected_id
        or manifest.get("schema_version") != expected_schema
    ):
        raise TrafficBundleBuildError("policy artifact manifest mismatch")
    return root


def _verify_onnx_file(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> Path:
    root = path.expanduser()
    if root.is_symlink():
        raise TrafficBundleBuildError(f"{label} must not be a symlink")
    root = root.resolve()
    if not root.is_file():
        raise TrafficBundleBuildError(f"{label} is missing: {root}")
    if _sha256_file(root) != expected_sha256:
        raise TrafficBundleBuildError(f"{label} SHA-256 mismatch")
    return root


def _copy_artifact(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_symlink():
            raise TrafficBundleBuildError(f"policy artifact contains symlink: {path}")
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def _verify_built_bundle(
    root: Path,
    *,
    expected_id: str,
    expected_schema: int,
    expected_signal_reads: int,
    expected_shortcut_id: str,
    expected_classifier: bool,
) -> None:
    _verify_checksums(root, include_nested_checksum_files=True)
    manifest = _load_mapping(root / "manifest.yaml")
    if (
        manifest.get("schema_version") != expected_schema
        or manifest.get("artifact_kind") != BUNDLE_KIND
        or manifest.get("artifact_id") != expected_id
    ):
        raise TrafficBundleBuildError("built bundle identity mismatch")
    expected_traffic_sha256 = (
        HUMAN_BBOX_TRAFFIC_SHA256
        if expected_schema == 6
        else TRAFFIC_SHA256
    )
    expected_classifier_sha256 = (
        HUMAN_BBOX_CLASSIFIER_SHA256
        if expected_schema == 6
        else CLASSIFIER_SHA256
    )
    if (
        _sha256_file(root / "signal" / "traffic_light.onnx")
        != expected_traffic_sha256
    ):
        raise TrafficBundleBuildError("built bundle traffic ONNX mismatch")
    classifier_path = root / "signal" / "tl_cls.onnx"
    if expected_classifier:
        if _sha256_file(classifier_path) != expected_classifier_sha256:
            raise TrafficBundleBuildError(
                "built bundle traffic classifier ONNX mismatch"
            )
    elif classifier_path.exists():
        raise TrafficBundleBuildError("legacy bundle must not contain classifier")
    if expected_schema < 3:
        if (
            manifest.get("red_latch") != _legacy_red_latch_contract()
            or "signal_vote" in manifest
        ):
            raise TrafficBundleBuildError(
                "built bundle legacy red latch mismatch"
            )
    elif expected_schema == 3 and (
        manifest.get("signal_vote")
        != _signal_vote_contract(expected_signal_reads)
        or "red_latch" in manifest
    ):
        raise TrafficBundleBuildError("built bundle signal vote mismatch")
    elif expected_schema >= 4 and (
        manifest.get("signal_vote")
        != _classifier_signal_vote_contract(
            expected_signal_reads,
            schema_version=expected_schema,
        )
        or "red_latch" in manifest
    ):
        raise TrafficBundleBuildError(
            "built bundle classifier signal vote mismatch"
        )
    if (
        expected_schema >= 5
        and _required_mapping(manifest, "mission", "bundle").get(
            "red_stop_yolo_missing_release_frames"
        )
        != 30
    ):
        raise TrafficBundleBuildError(
            "built bundle YOLO missing-release contract mismatch"
        )
    _verify_policy_artifact(
        root / "policies" / BASE_ID,
        expected_id=BASE_ID,
        expected_schema=6,
    )
    _verify_policy_artifact(
        root / "policies" / expected_shortcut_id,
        expected_id=expected_shortcut_id,
        expected_schema=7,
    )


def _write_checksums(root: Path) -> None:
    files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "SHA256SUMS"
    )
    (root / "SHA256SUMS").write_text(
        "".join(f"{_sha256_file(root / path)}  {path}\n" for path in files),
        encoding="utf-8",
    )


def _verify_checksums(root: Path, *, include_nested_checksum_files: bool) -> None:
    checksum_path = root / "SHA256SUMS"
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise TrafficBundleBuildError("SHA256SUMS is missing or unsafe")
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise TrafficBundleBuildError("invalid SHA256SUMS line")
        digest, relative = parts
        relative = relative.removeprefix("*")
        _validate_relative_path(relative)
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or relative in expected:
            raise TrafficBundleBuildError("invalid SHA256SUMS entry")
        expected[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != checksum_path
        and (include_nested_checksum_files or path.name != "SHA256SUMS")
    }
    if set(expected) != actual:
        raise TrafficBundleBuildError("SHA256SUMS file list mismatch")
    for relative, digest in expected.items():
        path = root / relative
        if path.is_symlink() or _sha256_file(path) != digest:
            raise TrafficBundleBuildError(f"checksum mismatch: {relative}")


def _load_mapping(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise TrafficBundleBuildError(f"missing or unsafe manifest: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TrafficBundleBuildError(f"invalid YAML: {path}") from exc
    if not isinstance(payload, Mapping):
        raise TrafficBundleBuildError(f"YAML root must be a mapping: {path}")
    return payload


def _required_mapping(
    payload: Mapping[str, object],
    key: str,
    context: str,
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TrafficBundleBuildError(f"{context}.{key} must be a mapping")
    return value


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or not re.fullmatch(r"[A-Za-z0-9._/-]+", value)
    ):
        raise TrafficBundleBuildError(f"unsafe artifact path: {value}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
