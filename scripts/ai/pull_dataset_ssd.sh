#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

XYCAR_APPLY=0
XYCAR_CHECKSUM=0
while (($#)); do
  case "$1" in
    --apply)
      XYCAR_APPLY=1
      ;;
    --checksum)
      XYCAR_CHECKSUM=1
      ;;
    -h|--help)
      printf 'usage: %s [--apply] [--checksum]\n' "$0"
      printf 'default: dry-run; --apply pulls SSD data into the local AI dataset\n'
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
xycar_ai_verify_shared_marker
xycar_ai_validate_absolute_path \
  "${XYCAR_AI_LOCAL_DATASET_ROOT}" "local dataset root"

XYCAR_SHARED_TELEOP="${XYCAR_AI_SHARED_DATASET_ROOT}/teleop"
[[ -d "${XYCAR_SHARED_TELEOP}" ]] ||
  xycar_ai_die "shared teleop directory is missing: ${XYCAR_SHARED_TELEOP}"
if ((XYCAR_APPLY)); then
  mkdir -p "${XYCAR_AI_LOCAL_DATASET_ROOT}"
elif [[ ! -d "${XYCAR_AI_LOCAL_DATASET_ROOT}" ]]; then
  xycar_ai_die \
    "local dataset root is missing; run bootstrap_env.sh before this dry-run: ${XYCAR_AI_LOCAL_DATASET_ROOT}"
fi

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
  "${XYCAR_SHARED_TELEOP}/" \
  "${XYCAR_AI_LOCAL_DATASET_ROOT}/"

if ((XYCAR_APPLY)); then
  printf 'Dataset pull applied: %s -> %s\n' \
    "${XYCAR_SHARED_TELEOP}" "${XYCAR_AI_LOCAL_DATASET_ROOT}"
else
  printf 'Dataset pull dry-run complete; rerun with --apply to copy changes.\n'
fi
