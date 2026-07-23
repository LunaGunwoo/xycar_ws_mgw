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
xycar_ai_validate_ssh_target "${XYCAR_AI_TRAIN_SSH}" "training SSH target"
xycar_ai_validate_absolute_path \
  "${XYCAR_AI_TRAIN_ARTIFACT_ROOT}" "training artifact root"
xycar_ai_validate_absolute_path \
  "${XYCAR_AI_LOCAL_ARTIFACT_ROOT}" "local artifact root"

XYCAR_REMOTE_ARTIFACT="${XYCAR_AI_TRAIN_ARTIFACT_ROOT}/${XYCAR_ARTIFACT_ID}"
XYCAR_LOCAL_ARTIFACT="${XYCAR_AI_LOCAL_ARTIFACT_ROOT}/${XYCAR_ARTIFACT_ID}"
[[ ! -e "${XYCAR_LOCAL_ARTIFACT}" ]] ||
  xycar_ai_die "local artifact already exists: ${XYCAR_LOCAL_ARTIFACT}"

ssh "${XYCAR_AI_TRAIN_SSH}" bash -s -- \
  "${XYCAR_REMOTE_ARTIFACT}" <<'REMOTE_PREFLIGHT'
set -euo pipefail
artifact_dir="$1"
[[ -f "${artifact_dir}/manifest.yaml" ]] ||
  { printf 'artifact manifest is missing\n' >&2; exit 1; }
[[ -f "${artifact_dir}/SHA256SUMS" ]] ||
  { printf 'artifact checksum file is missing\n' >&2; exit 1; }
if find "${artifact_dir}" -mindepth 1 \
  ! -type d ! -type f -print -quit | grep -q .; then
  printf 'artifact must contain only directories and regular files\n' >&2
  exit 1
fi
REMOTE_PREFLIGHT

if ((XYCAR_DRY_RUN)); then
  XYCAR_DRY_DEST="$(mktemp -d)"
  xycar_ai_cleanup_dry_dest() {
    if [[ -d "${XYCAR_DRY_DEST}" ]]; then
      find "${XYCAR_DRY_DEST}" -depth -mindepth 1 -delete
      rmdir "${XYCAR_DRY_DEST}"
    fi
  }
  trap xycar_ai_cleanup_dry_dest EXIT
  rsync -a --no-links --no-devices --no-specials --protect-args \
    --dry-run --human-readable --itemize-changes \
    "${XYCAR_AI_TRAIN_SSH}:${XYCAR_REMOTE_ARTIFACT}/" \
    "${XYCAR_DRY_DEST}/"
  exit 0
fi

mkdir -p "${XYCAR_AI_LOCAL_ARTIFACT_ROOT}"
XYCAR_INCOMING="${XYCAR_AI_LOCAL_ARTIFACT_ROOT}/.incoming-${XYCAR_ARTIFACT_ID}-$$"
[[ ! -e "${XYCAR_INCOMING}" ]] ||
  xycar_ai_die "temporary artifact path already exists: ${XYCAR_INCOMING}"
xycar_ai_cleanup_incoming() {
  if [[ -d "${XYCAR_INCOMING}" ]]; then
    find "${XYCAR_INCOMING}" -depth -delete
  fi
}
trap xycar_ai_cleanup_incoming EXIT

mkdir "${XYCAR_INCOMING}"
rsync -a --no-links --no-devices --no-specials --protect-args \
  --human-readable --itemize-changes \
  "${XYCAR_AI_TRAIN_SSH}:${XYCAR_REMOTE_ARTIFACT}/" \
  "${XYCAR_INCOMING}/"
xycar_ai_verify_sha_manifest "${XYCAR_INCOMING}"
mv "${XYCAR_INCOMING}" "${XYCAR_LOCAL_ARTIFACT}"
trap - EXIT

printf 'Model artifact fetched and verified: %s\n' "${XYCAR_LOCAL_ARTIFACT}"
