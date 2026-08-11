#!/usr/bin/env bash

set -euo pipefail

XYCAR_AI_SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd -P
)"
XYCAR_MGW_ROOT="$(
  cd -- "${XYCAR_AI_SCRIPT_DIR}/../.." >/dev/null 2>&1
  pwd -P
)"

XYCAR_AI_AUTHORING_ROOT="${XYCAR_AI_AUTHORING_ROOT:-/home/xytron/xycar_ws/apps/xycar_ws_mgw}"
XYCAR_AI_BUNDLE_ROOT="${XYCAR_AI_BUNDLE_ROOT:-${XYCAR_MGW_ROOT}/ai}"
XYCAR_AI_DEFAULT_VEHICLE_SSH="xytron@xycar"
XYCAR_AI_VEHICLE_SSH="${XYCAR_AI_VEHICLE_SSH:-${XYCAR_AI_DEFAULT_VEHICLE_SSH}}"
XYCAR_AI_VEHICLE_DATASET_ROOT="${XYCAR_AI_VEHICLE_DATASET_ROOT:-/home/xytron/xycar_data/teleop}"
XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_AI_LOCAL_DATASET_ROOT:-${XYCAR_AI_BUNDLE_ROOT}/datasets/teleop}"
XYCAR_AI_LOCAL_ARTIFACT_ROOT="${XYCAR_AI_LOCAL_ARTIFACT_ROOT:-${XYCAR_AI_BUNDLE_ROOT}/artifacts/models}"
XYCAR_AI_VEHICLE_ARTIFACT_ROOT="${XYCAR_AI_VEHICLE_ARTIFACT_ROOT:-/home/xytron/xycar_ws_mgw/artifacts/models}"
XYCAR_AI_UV_VERSION="${XYCAR_AI_UV_VERSION:-0.11.24}"
XYCAR_AI_SHARED_MARKER_NAME=".xycar-ai-dataset-share"
XYCAR_AI_SHARED_MARKER_CONTENT="xycar-ai-dataset-share-v1"

xycar_ai_die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

xycar_ai_require_command() {
  command -v "$1" >/dev/null 2>&1 ||
    xycar_ai_die "required command is unavailable: $1"
}

xycar_ai_require_authoring_checkout() {
  if [[ "${XYCAR_AI_ALLOW_ANY_CHECKOUT:-0}" == "1" ]]; then
    return
  fi
  local actual_root
  actual_root="$(git -C "${XYCAR_MGW_ROOT}" rev-parse --show-toplevel 2>/dev/null)" ||
    xycar_ai_die "MGW root is not a Git checkout: ${XYCAR_MGW_ROOT}"
  [[ "${actual_root}" == "${XYCAR_MGW_ROOT}" ]] ||
    xycar_ai_die "unexpected Git root: ${actual_root}"
  [[ "${XYCAR_MGW_ROOT}" == "${XYCAR_AI_AUTHORING_ROOT}" ]] ||
    xycar_ai_die \
      "run this command only from the authoring checkout: ${XYCAR_AI_AUTHORING_ROOT}"
  local origin_url
  origin_url="$(git -C "${XYCAR_MGW_ROOT}" remote get-url origin 2>/dev/null)" ||
    xycar_ai_die "MGW origin is unavailable"
  [[ "${origin_url}" == "https://github.com/LunaGunwoo/xycar_ws_mgw.git" ]] ||
    xycar_ai_die "unexpected MGW origin: ${origin_url}"
}

xycar_ai_validate_absolute_path() {
  local value="$1"
  local label="$2"
  [[ "${value}" == /* ]] || xycar_ai_die "${label} must be absolute: ${value}"
  [[ "${value}" =~ ^/[A-Za-z0-9._/-]+$ ]] ||
    xycar_ai_die "${label} contains unsupported characters: ${value}"
  [[ "/${value#/}/" != *"/../"* ]] ||
    xycar_ai_die "${label} must not contain '..': ${value}"
  [[ "${value}" != "/" ]] || xycar_ai_die "${label} must not be filesystem root"
}

xycar_ai_require_shared_dataset_root() {
  [[ -n "${XYCAR_AI_SHARED_DATASET_ROOT:-}" ]] ||
    xycar_ai_die \
      "XYCAR_AI_SHARED_DATASET_ROOT is required (for example /mnt/e/xycar-ai-dataset)"
  xycar_ai_validate_absolute_path \
    "${XYCAR_AI_SHARED_DATASET_ROOT}" "shared dataset root"
}

xycar_ai_require_shared_mount() {
  if [[ "${XYCAR_AI_ALLOW_ROOT_FILESYSTEM_SHARED:-0}" == "1" ]]; then
    return
  fi
  xycar_ai_require_command findmnt
  local probe_path="${XYCAR_AI_SHARED_DATASET_ROOT}"
  while [[ ! -e "${probe_path}" ]]; do
    probe_path="$(dirname -- "${probe_path}")"
  done
  local mount_target
  mount_target="$(findmnt -n -o TARGET -T "${probe_path}")" ||
    xycar_ai_die "shared dataset path is not on a mounted filesystem"
  [[ "${mount_target}" != "/" ]] ||
    xycar_ai_die \
      "shared dataset path resolves to the system filesystem; mount the SSD first"
}

xycar_ai_verify_shared_marker() {
  local marker="${XYCAR_AI_SHARED_DATASET_ROOT}/${XYCAR_AI_SHARED_MARKER_NAME}"
  [[ -f "${marker}" ]] ||
    xycar_ai_die "shared dataset marker is missing: ${marker}"
  [[ "$(<"${marker}")" == "${XYCAR_AI_SHARED_MARKER_CONTENT}" ]] ||
    xycar_ai_die "shared dataset marker has unexpected content: ${marker}"
}

xycar_ai_validate_ssh_target() {
  local value="$1"
  local label="$2"
  [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9.-]*$ ]] ||
    xycar_ai_die "${label} must use a safe user@host form: ${value}"
}

xycar_ai_validate_artifact_id() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
    xycar_ai_die "invalid artifact id: $1"
}

xycar_ai_verify_sha_manifest() {
  local artifact_dir="$1"
  [[ -f "${artifact_dir}/manifest.yaml" ]] ||
    xycar_ai_die "artifact manifest is missing: ${artifact_dir}/manifest.yaml"
  [[ -f "${artifact_dir}/SHA256SUMS" ]] ||
    xycar_ai_die "artifact checksum file is missing: ${artifact_dir}/SHA256SUMS"
  if find "${artifact_dir}" -mindepth 1 \
    ! -type d ! -type f -print -quit | grep -q .; then
    xycar_ai_die \
      "artifact must contain only directories and regular files: ${artifact_dir}"
  fi
  awk '
    NF < 2 { exit 1 }
    {
      path = $2
      sub(/^\*/, "", path)
      if (path !~ /^[A-Za-z0-9._\/-]+$/ ||
          path ~ /^\// ||
          path ~ /(^|\/)\.\.(\/|$)/) {
        exit 1
      }
    }
  ' "${artifact_dir}/SHA256SUMS" ||
    xycar_ai_die "SHA256SUMS contains an unsafe path"
  while IFS= read -r artifact_file; do
    awk -v expected="${artifact_file}" '
      {
        path = $2
        sub(/^\*/, "", path)
        if (path == expected) {
          found = 1
        }
      }
      END { exit(found ? 0 : 1) }
    ' "${artifact_dir}/SHA256SUMS" ||
      xycar_ai_die "artifact file is missing from SHA256SUMS: ${artifact_file}"
  done < <(
    find "${artifact_dir}" -type f \
      ! -path "${artifact_dir}/SHA256SUMS" \
      -printf '%P\n'
  )
  (
    cd -- "${artifact_dir}"
    sha256sum -c SHA256SUMS
  )
}
