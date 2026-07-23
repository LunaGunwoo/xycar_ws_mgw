#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

xycar_ai_require_command mktemp
xycar_ai_require_command mkfifo
xycar_ai_require_command rsync

XYCAR_TEST_ROOT="$(mktemp -d)"
xycar_ai_cleanup_test() {
  if [[ -d "${XYCAR_TEST_ROOT}" ]]; then
    find "${XYCAR_TEST_ROOT}" -depth -mindepth 1 -delete
    rmdir "${XYCAR_TEST_ROOT}"
  fi
}
trap xycar_ai_cleanup_test EXIT

mkdir -p \
  "${XYCAR_TEST_ROOT}/code-source/.venv" \
  "${XYCAR_TEST_ROOT}/code-source/datasets" \
  "${XYCAR_TEST_ROOT}/code-source/artifacts" \
  "${XYCAR_TEST_ROOT}/code-destination/.venv" \
  "${XYCAR_TEST_ROOT}/code-destination/datasets" \
  "${XYCAR_TEST_ROOT}/code-destination/artifacts"
printf 'new\n' >"${XYCAR_TEST_ROOT}/code-source/current.txt"
printf 'source-venv\n' >"${XYCAR_TEST_ROOT}/code-source/.venv/ignore"
printf 'source-dataset\n' >"${XYCAR_TEST_ROOT}/code-source/datasets/ignore"
printf 'source-artifact\n' >"${XYCAR_TEST_ROOT}/code-source/artifacts/ignore"
printf 'old\n' >"${XYCAR_TEST_ROOT}/code-destination/stale.txt"
printf 'venv\n' >"${XYCAR_TEST_ROOT}/code-destination/.venv/keep"
printf 'dataset\n' >"${XYCAR_TEST_ROOT}/code-destination/datasets/keep"
printf 'artifact\n' >"${XYCAR_TEST_ROOT}/code-destination/artifacts/keep"

rsync -a --delete \
  --filter="merge ${XYCAR_AI_SCRIPT_DIR}/code-rsync-filter.rules" \
  "${XYCAR_TEST_ROOT}/code-source/" \
  "${XYCAR_TEST_ROOT}/code-destination/"

[[ -f "${XYCAR_TEST_ROOT}/code-destination/current.txt" ]]
[[ ! -e "${XYCAR_TEST_ROOT}/code-destination/stale.txt" ]]
[[ -f "${XYCAR_TEST_ROOT}/code-destination/.venv/keep" ]]
[[ -f "${XYCAR_TEST_ROOT}/code-destination/datasets/keep" ]]
[[ -f "${XYCAR_TEST_ROOT}/code-destination/artifacts/keep" ]]
[[ ! -e "${XYCAR_TEST_ROOT}/code-destination/.venv/ignore" ]]
[[ ! -e "${XYCAR_TEST_ROOT}/code-destination/datasets/ignore" ]]
[[ ! -e "${XYCAR_TEST_ROOT}/code-destination/artifacts/ignore" ]]

mkdir -p \
  "${XYCAR_TEST_ROOT}/dataset-source/20260724_010101_001_session" \
  "${XYCAR_TEST_ROOT}/dataset-source/20260724_010102_001_session_2" \
  "${XYCAR_TEST_ROOT}/dataset-source/20260724_010103_001_incomplete" \
  "${XYCAR_TEST_ROOT}/dataset-source/_recording_20260724_010104_001" \
  "${XYCAR_TEST_ROOT}/dataset-destination"
for directory in "${XYCAR_TEST_ROOT}"/dataset-source/*; do
  printf 'sample\n' >"${directory}/samples.csv"
done

rsync -a \
  --filter="merge ${XYCAR_AI_SCRIPT_DIR}/dataset-rsync-filter.rules" \
  "${XYCAR_TEST_ROOT}/dataset-source/" \
  "${XYCAR_TEST_ROOT}/dataset-destination/"

[[ -d "${XYCAR_TEST_ROOT}/dataset-destination/20260724_010101_001_session" ]]
[[ -d "${XYCAR_TEST_ROOT}/dataset-destination/20260724_010102_001_session_2" ]]
[[ -d "${XYCAR_TEST_ROOT}/dataset-destination/20260724_010103_001_incomplete" ]]
[[ ! -e "${XYCAR_TEST_ROOT}/dataset-destination/_recording_20260724_010104_001" ]]

printf 'changed\n' \
  >"${XYCAR_TEST_ROOT}/dataset-source/20260724_010101_001_session/samples.csv"
printf 'destination-only\n' \
  >"${XYCAR_TEST_ROOT}/dataset-destination/preserved-local-file"
rsync -a \
  --filter="merge ${XYCAR_AI_SCRIPT_DIR}/dataset-rsync-filter.rules" \
  "${XYCAR_TEST_ROOT}/dataset-source/" \
  "${XYCAR_TEST_ROOT}/dataset-destination/"
grep -qx 'changed' \
  "${XYCAR_TEST_ROOT}/dataset-destination/20260724_010101_001_session/samples.csv"
[[ -f "${XYCAR_TEST_ROOT}/dataset-destination/preserved-local-file" ]]

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
