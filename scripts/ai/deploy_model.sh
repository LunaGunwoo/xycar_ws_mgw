#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

XYCAR_DRY_RUN=0
XYCAR_ARTIFACT_ID=""
while (($#)); do
  case "$1" in
    --dry-run)
      XYCAR_DRY_RUN=1
      ;;
    -h|--help)
      printf 'usage: %s <artifact-id> [--dry-run]\n' "$0"
      exit 0
      ;;
    -*)
      xycar_ai_die "unknown argument: $1"
      ;;
    *)
      [[ -z "${XYCAR_ARTIFACT_ID}" ]] ||
        xycar_ai_die "only one artifact id is allowed"
      XYCAR_ARTIFACT_ID="$1"
      ;;
  esac
  shift
done

[[ -n "${XYCAR_ARTIFACT_ID}" ]] ||
  xycar_ai_die "artifact id is required"
xycar_ai_validate_artifact_id "${XYCAR_ARTIFACT_ID}"
xycar_ai_require_authoring_checkout
xycar_ai_require_command rsync
xycar_ai_require_command sha256sum
xycar_ai_require_command ssh
xycar_ai_validate_ssh_target "${XYCAR_AI_VEHICLE_SSH}" "vehicle SSH target"
xycar_ai_validate_absolute_path \
  "${XYCAR_AI_LOCAL_ARTIFACT_ROOT}" "local artifact root"
xycar_ai_validate_absolute_path \
  "${XYCAR_AI_VEHICLE_ARTIFACT_ROOT}" "vehicle artifact root"

XYCAR_LOCAL_ARTIFACT="${XYCAR_AI_LOCAL_ARTIFACT_ROOT}/${XYCAR_ARTIFACT_ID}"
[[ -d "${XYCAR_LOCAL_ARTIFACT}" ]] ||
  xycar_ai_die "local artifact is missing: ${XYCAR_LOCAL_ARTIFACT}"
xycar_ai_verify_sha_manifest "${XYCAR_LOCAL_ARTIFACT}"

XYCAR_REMOTE_FINAL="${XYCAR_AI_VEHICLE_ARTIFACT_ROOT}/${XYCAR_ARTIFACT_ID}"
XYCAR_REMOTE_INCOMING="${XYCAR_AI_VEHICLE_ARTIFACT_ROOT}/.incoming-${XYCAR_ARTIFACT_ID}-$$"
ssh "${XYCAR_AI_VEHICLE_SSH}" bash -s -- \
  "${XYCAR_AI_VEHICLE_ARTIFACT_ROOT}" \
  "${XYCAR_REMOTE_FINAL}" \
  "${XYCAR_REMOTE_INCOMING}" \
  "${XYCAR_DRY_RUN}" <<'REMOTE_PREPARE'
set -euo pipefail
artifact_root="$1"
final_path="$2"
incoming_path="$3"
dry_run="$4"
[[ -d "/home/xytron/xycar_ws_mgw/.git" ]] ||
  { printf 'vehicle MGW checkout is missing\n' >&2; exit 1; }
[[ ! -e "${final_path}" ]] ||
  { printf 'vehicle artifact already exists: %s\n' "${final_path}" >&2; exit 1; }
[[ ! -e "${incoming_path}" ]] ||
  { printf 'vehicle temporary path exists: %s\n' "${incoming_path}" >&2; exit 1; }
if [[ "${dry_run}" != "1" ]]; then
  mkdir -p "${artifact_root}"
elif [[ ! -d "${artifact_root}" ]]; then
  printf 'dry-run: vehicle artifact root would be created: %s\n' \
    "${artifact_root}"
fi
REMOTE_PREPARE

if ((XYCAR_DRY_RUN)); then
  printf 'dry-run: would deploy %s to %s:%s\n' \
    "${XYCAR_LOCAL_ARTIFACT}" \
    "${XYCAR_AI_VEHICLE_SSH}" \
    "${XYCAR_REMOTE_FINAL}"
  exit 0
fi

XYCAR_REMOTE_CLEANUP_NEEDED=1
xycar_ai_cleanup_remote_incoming() {
  if ((XYCAR_REMOTE_CLEANUP_NEEDED)); then
    ssh "${XYCAR_AI_VEHICLE_SSH}" bash -s -- \
      "${XYCAR_REMOTE_INCOMING}" <<'REMOTE_CLEANUP' >/dev/null 2>&1 || true
set -euo pipefail
incoming_path="$1"
if [[ -d "${incoming_path}" ]]; then
  find "${incoming_path}" -depth -delete
fi
REMOTE_CLEANUP
  fi
}
trap xycar_ai_cleanup_remote_incoming EXIT

XYCAR_RSYNC_ARGS=(
  -a
  --no-links
  --no-devices
  --no-specials
  --protect-args
  --human-readable
  --itemize-changes
)
rsync "${XYCAR_RSYNC_ARGS[@]}" \
  "${XYCAR_LOCAL_ARTIFACT}/" \
  "${XYCAR_AI_VEHICLE_SSH}:${XYCAR_REMOTE_INCOMING}/"

ssh "${XYCAR_AI_VEHICLE_SSH}" bash -s -- \
  "${XYCAR_AI_VEHICLE_ARTIFACT_ROOT}" \
  "${XYCAR_REMOTE_INCOMING}" \
  "${XYCAR_REMOTE_FINAL}" <<'REMOTE_VERIFY'
set -euo pipefail
artifact_root="$1"
incoming_path="$2"
final_path="$3"
cleanup_incoming() {
  if [[ -d "${incoming_path}" ]]; then
    find "${incoming_path}" -depth -delete
  fi
}
trap cleanup_incoming EXIT
[[ -d "/home/xytron/xycar_ws_mgw/.git" ]] ||
  { printf 'vehicle MGW checkout is missing\n' >&2; exit 1; }
[[ -f "${incoming_path}/manifest.yaml" ]] ||
  { printf 'manifest.yaml is missing\n' >&2; exit 1; }
[[ -f "${incoming_path}/SHA256SUMS" ]] ||
  { printf 'SHA256SUMS is missing\n' >&2; exit 1; }
if find "${incoming_path}" -mindepth 1 \
  ! -type d ! -type f -print -quit | grep -q .; then
  printf 'artifact must contain only directories and regular files\n' >&2
  exit 1
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
' "${incoming_path}/SHA256SUMS"
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
  ' "${incoming_path}/SHA256SUMS" || {
    printf 'artifact file is missing from SHA256SUMS: %s\n' \
      "${artifact_file}" >&2
    exit 1
  }
done < <(
  find "${incoming_path}" -type f \
    ! -path "${incoming_path}/SHA256SUMS" \
    -printf '%P\n'
)
(
  cd -- "${incoming_path}"
  sha256sum -c SHA256SUMS
)
mkdir -p "${artifact_root}"
mv "${incoming_path}" "${final_path}"
trap - EXIT
REMOTE_VERIFY
XYCAR_REMOTE_CLEANUP_NEEDED=0
trap - EXIT

printf 'Model artifact deployed and verified: %s:%s\n' \
  "${XYCAR_AI_VEHICLE_SSH}" "${XYCAR_REMOTE_FINAL}"
