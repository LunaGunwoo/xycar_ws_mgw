#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

xycar_ai_require_command mktemp
xycar_ai_require_command mkfifo
xycar_ai_require_command rsync
[[ "${XYCAR_AI_DEFAULT_VEHICLE_SSH}" == "xytron@xycar" ]] ||
  xycar_ai_die "unexpected default vehicle Tailscale SSH target"
xycar_ai_validate_ssh_target \
  "${XYCAR_AI_DEFAULT_VEHICLE_SSH}" "default vehicle SSH target"

XYCAR_TEST_ROOT="$(mktemp -d)"
xycar_ai_cleanup_test() {
  if [[ -d "${XYCAR_TEST_ROOT}" ]]; then
    find "${XYCAR_TEST_ROOT}" -depth -mindepth 1 -delete
    rmdir "${XYCAR_TEST_ROOT}"
  fi
}
trap xycar_ai_cleanup_test EXIT

mkdir -p \
  "${XYCAR_TEST_ROOT}/dataset-source/20260724_010101_001_session" \
  "${XYCAR_TEST_ROOT}/dataset-source/20260724_010102_001_session_2" \
  "${XYCAR_TEST_ROOT}/dataset-source/20260724_010103_001_incomplete" \
  "${XYCAR_TEST_ROOT}/dataset-source/_recording_20260724_010104_001"
for directory in "${XYCAR_TEST_ROOT}"/dataset-source/*; do
  printf 'sample\n' >"${directory}/samples.csv"
done
printf 'partial\n' \
  >"${XYCAR_TEST_ROOT}/dataset-source/20260724_010101_001_session/frame.part"

XYCAR_TEST_ENV=(
  env
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1
  XYCAR_AI_ALLOW_ROOT_FILESYSTEM_SHARED=1
  XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_TEST_ROOT}/dataset-source"
  XYCAR_AI_SHARED_DATASET_ROOT="${XYCAR_TEST_ROOT}/shared-dataset"
)

if "${XYCAR_TEST_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/publish_dataset_ssd.sh" --apply \
  >/dev/null 2>&1; then
  xycar_ai_die "publish accepted a missing SSD marker without --init"
fi
"${XYCAR_TEST_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/publish_dataset_ssd.sh" --init >/dev/null
[[ ! -e "${XYCAR_TEST_ROOT}/shared-dataset" ]]
"${XYCAR_TEST_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/publish_dataset_ssd.sh" --init --apply >/dev/null

XYCAR_SHARED_DESTINATION="${XYCAR_TEST_ROOT}/shared-dataset/teleop"
[[ -f "${XYCAR_TEST_ROOT}/shared-dataset/.xycar-ai-dataset-share" ]]
[[ -d "${XYCAR_SHARED_DESTINATION}/20260724_010101_001_session" ]]
[[ -d "${XYCAR_SHARED_DESTINATION}/20260724_010102_001_session_2" ]]
[[ -d "${XYCAR_SHARED_DESTINATION}/20260724_010103_001_incomplete" ]]
[[ ! -e "${XYCAR_SHARED_DESTINATION}/_recording_20260724_010104_001" ]]
[[ ! -e \
  "${XYCAR_SHARED_DESTINATION}/20260724_010101_001_session/frame.part" ]]

printf 'destination-only\n' \
  >"${XYCAR_SHARED_DESTINATION}/preserved-ssd-file"
printf 'new\n' >"${XYCAR_TEST_ROOT}/dataset-source/new-file"
"${XYCAR_TEST_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/publish_dataset_ssd.sh" >/dev/null
[[ ! -e "${XYCAR_SHARED_DESTINATION}/new-file" ]]
"${XYCAR_TEST_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/publish_dataset_ssd.sh" --apply >/dev/null
[[ -f "${XYCAR_SHARED_DESTINATION}/new-file" ]]
[[ -f "${XYCAR_SHARED_DESTINATION}/preserved-ssd-file" ]]

mkdir -p "${XYCAR_TEST_ROOT}/dataset-pulled"
XYCAR_PULL_ENV=(
  env
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1
  XYCAR_AI_ALLOW_ROOT_FILESYSTEM_SHARED=1
  XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_TEST_ROOT}/dataset-pulled"
  XYCAR_AI_SHARED_DATASET_ROOT="${XYCAR_TEST_ROOT}/shared-dataset"
)
"${XYCAR_PULL_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" >/dev/null
[[ ! -e "${XYCAR_TEST_ROOT}/dataset-pulled/new-file" ]]
"${XYCAR_PULL_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --apply >/dev/null
[[ -f "${XYCAR_TEST_ROOT}/dataset-pulled/new-file" ]]
printf 'local-only\n' >"${XYCAR_TEST_ROOT}/dataset-pulled/preserved-local-file"
"${XYCAR_PULL_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --apply >/dev/null
[[ -f "${XYCAR_TEST_ROOT}/dataset-pulled/preserved-local-file" ]]

mkdir -p "${XYCAR_TEST_ROOT}/missing-marker"
if env \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_ALLOW_ROOT_FILESYSTEM_SHARED=1 \
  XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_TEST_ROOT}/dataset-pulled" \
  XYCAR_AI_SHARED_DATASET_ROOT="${XYCAR_TEST_ROOT}/missing-marker" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --apply \
  >/dev/null 2>&1; then
  xycar_ai_die "pull accepted a missing SSD marker"
fi

mkdir -p "${XYCAR_TEST_ROOT}/artifact"
printf 'schema_version: 1\nartifact_id: fixture\n' \
  >"${XYCAR_TEST_ROOT}/artifact/manifest.yaml"
printf 'model\n' >"${XYCAR_TEST_ROOT}/artifact/model.onnx"
(
  cd "${XYCAR_TEST_ROOT}/artifact"
  sha256sum manifest.yaml model.onnx >SHA256SUMS
)
xycar_ai_verify_sha_manifest "${XYCAR_TEST_ROOT}/artifact"
if (xycar_ai_validate_artifact_id '../unsafe') >/dev/null 2>&1; then
  xycar_ai_die "unsafe artifact id was accepted"
fi
if (xycar_ai_validate_absolute_path '/tmp/unsafe;path' 'test path') \
  >/dev/null 2>&1; then
  xycar_ai_die "unsafe absolute path was accepted"
fi
if (xycar_ai_validate_ssh_target '-oProxyCommand=unsafe' 'test SSH target') \
  >/dev/null 2>&1; then
  xycar_ai_die "unsafe SSH target was accepted"
fi
mkdir -p "${XYCAR_TEST_ROOT}/unsafe-artifact"
printf 'schema_version: 1\n' \
  >"${XYCAR_TEST_ROOT}/unsafe-artifact/manifest.yaml"
printf '%064d  ../outside.onnx\n' 0 \
  >"${XYCAR_TEST_ROOT}/unsafe-artifact/SHA256SUMS"
if (xycar_ai_verify_sha_manifest \
  "${XYCAR_TEST_ROOT}/unsafe-artifact") >/dev/null 2>&1; then
  xycar_ai_die "unsafe checksum path was accepted"
fi
printf 'unlisted\n' >"${XYCAR_TEST_ROOT}/artifact/unlisted.bin"
if (xycar_ai_verify_sha_manifest \
  "${XYCAR_TEST_ROOT}/artifact") >/dev/null 2>&1; then
  xycar_ai_die "unlisted artifact file was accepted"
fi
find "${XYCAR_TEST_ROOT}/artifact" -type f \
  -name unlisted.bin -delete
mkfifo "${XYCAR_TEST_ROOT}/artifact/unlisted.fifo"
if (xycar_ai_verify_sha_manifest \
  "${XYCAR_TEST_ROOT}/artifact") >/dev/null 2>&1; then
  xycar_ai_die "non-regular artifact entry was accepted"
fi

printf 'AI workflow fixture tests passed\n'
