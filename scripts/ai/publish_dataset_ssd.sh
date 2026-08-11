#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

XYCAR_APPLY=0
XYCAR_INIT=0
XYCAR_CHECKSUM=0
while (($#)); do
  case "$1" in
    --apply)
      XYCAR_APPLY=1
      ;;
    --init)
      XYCAR_INIT=1
      ;;
    --checksum)
      XYCAR_CHECKSUM=1
      ;;
    -h|--help)
      printf 'usage: %s [--apply] [--init] [--checksum]\n' "$0"
      printf 'default: dry-run; initialize a new SSD only with --init --apply\n'
      exit 0
      ;;
    *)
      xycar_ai_die "unknown argument: $1"
      ;;
  esac
  shift
done

xycar_ai_require_authoring_checkout
xycar_ai_require_command rsync
xycar_ai_require_shared_dataset_root
xycar_ai_require_shared_mount
xycar_ai_validate_absolute_path \
  "${XYCAR_AI_LOCAL_DATASET_ROOT}" "local dataset root"
[[ -d "${XYCAR_AI_LOCAL_DATASET_ROOT}" ]] ||
  xycar_ai_die "local dataset root is missing: ${XYCAR_AI_LOCAL_DATASET_ROOT}"

XYCAR_MARKER="${XYCAR_AI_SHARED_DATASET_ROOT}/${XYCAR_AI_SHARED_MARKER_NAME}"
XYCAR_SHARED_TELEOP="${XYCAR_AI_SHARED_DATASET_ROOT}/teleop"
if [[ ! -f "${XYCAR_MARKER}" ]]; then
  ((XYCAR_INIT)) || xycar_ai_die "shared dataset marker is missing: ${XYCAR_MARKER}"
  if ((!XYCAR_APPLY)); then
    printf 'dry-run: would initialize %s and publish the local dataset\n' \
      "${XYCAR_AI_SHARED_DATASET_ROOT}"
    exit 0
  fi
  [[ ! -e "${XYCAR_AI_SHARED_DATASET_ROOT}" ]] ||
    [[ -d "${XYCAR_AI_SHARED_DATASET_ROOT}" ]] ||
    xycar_ai_die "shared dataset root is not a directory"
  if [[ -d "${XYCAR_AI_SHARED_DATASET_ROOT}" ]] &&
     find "${XYCAR_AI_SHARED_DATASET_ROOT}" -mindepth 1 -print -quit | grep -q .; then
    xycar_ai_die \
      "refusing to initialize a non-empty unmarked directory: ${XYCAR_AI_SHARED_DATASET_ROOT}"
  fi
  mkdir -p "${XYCAR_SHARED_TELEOP}"
  printf '%s\n' "${XYCAR_AI_SHARED_MARKER_CONTENT}" >"${XYCAR_MARKER}"
fi
xycar_ai_verify_shared_marker
[[ -d "${XYCAR_SHARED_TELEOP}" ]] ||
  xycar_ai_die "shared teleop directory is missing: ${XYCAR_SHARED_TELEOP}"

XYCAR_RSYNC_ARGS=(
  -rt
  --modify-window=1
  --omit-dir-times
  --human-readable
  --itemize-changes
  --info=progress2
  --partial
  --partial-dir=.rsync-partial
  --filter="merge ${XYCAR_AI_SCRIPT_DIR}/dataset-rsync-filter.rules"
)
if ((!XYCAR_APPLY)); then
  XYCAR_RSYNC_ARGS+=(--dry-run)
fi
if ((XYCAR_CHECKSUM)); then
  XYCAR_RSYNC_ARGS+=(--checksum)
fi

rsync "${XYCAR_RSYNC_ARGS[@]}" \
  "${XYCAR_AI_LOCAL_DATASET_ROOT}/" \
  "${XYCAR_SHARED_TELEOP}/"

if ((XYCAR_APPLY)); then
  printf 'Dataset publish applied: %s -> %s\n' \
    "${XYCAR_AI_LOCAL_DATASET_ROOT}" "${XYCAR_SHARED_TELEOP}"
else
  printf 'Dataset publish dry-run complete; rerun with --apply to copy changes.\n'
fi
