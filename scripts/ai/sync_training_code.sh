#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

XYCAR_DRY_RUN=0
XYCAR_INIT=0
while (($#)); do
  case "$1" in
    --dry-run)
      XYCAR_DRY_RUN=1
      ;;
    --init)
      XYCAR_INIT=1
      ;;
    -h|--help)
      printf 'usage: %s [--dry-run] [--init]\n' "$0"
      exit 0
      ;;
    *)
      xycar_ai_die "unknown argument: $1"
      ;;
  esac
  shift
done

xycar_ai_require_authoring_checkout
xycar_ai_require_command git
xycar_ai_require_command mktemp
xycar_ai_require_command rsync
xycar_ai_require_command ssh
xycar_ai_validate_ssh_target "${XYCAR_AI_TRAIN_SSH}" "training SSH target"
xycar_ai_validate_absolute_path "${XYCAR_AI_TRAIN_ROOT}" "training root"
xycar_ai_validate_absolute_path "${XYCAR_AI_TRAIN_UV}" "training uv path"
[[ -d "${XYCAR_AI_BUNDLE_ROOT}" ]] ||
  xycar_ai_die "AI bundle is missing: ${XYCAR_AI_BUNDLE_ROOT}"

XYCAR_STAGE_DIR="$(mktemp -d)"
xycar_ai_cleanup_stage() {
  if [[ -d "${XYCAR_STAGE_DIR}" ]]; then
    find "${XYCAR_STAGE_DIR}" -depth -mindepth 1 -delete
    rmdir "${XYCAR_STAGE_DIR}"
  fi
}
trap xycar_ai_cleanup_stage EXIT

XYCAR_FILE_LIST="${XYCAR_STAGE_DIR}/files.zlist"
git -C "${XYCAR_MGW_ROOT}" \
  ls-files -z --cached --others --exclude-standard -- ai/ \
  >"${XYCAR_FILE_LIST}"

if [[ ! -s "${XYCAR_FILE_LIST}" ]]; then
  xycar_ai_die "AI bundle has no source-controlled files"
fi

sed -z 's#^ai/##' "${XYCAR_FILE_LIST}" >"${XYCAR_STAGE_DIR}/relative.zlist"
mkdir "${XYCAR_STAGE_DIR}/bundle"
rsync -a --from0 \
  --files-from="${XYCAR_STAGE_DIR}/relative.zlist" \
  "${XYCAR_AI_BUNDLE_ROOT}/" \
  "${XYCAR_STAGE_DIR}/bundle/"

XYCAR_REMOTE_MODE="existing"
if ((XYCAR_INIT)); then
  XYCAR_REMOTE_MODE="init"
fi
ssh "${XYCAR_AI_TRAIN_SSH}" bash -s -- \
  "${XYCAR_AI_TRAIN_ROOT}" \
  "${XYCAR_REMOTE_MODE}" \
  "${XYCAR_DRY_RUN}" <<'REMOTE_CHECK'
set -euo pipefail
training_root="$1"
mode="$2"
dry_run="$3"
marker="${training_root}/.xycar-ai-workspace"
if [[ -f "${marker}" ]]; then
  exit 0
fi
if [[ "${mode}" != "init" ]]; then
  printf 'training workspace marker is missing: %s\n' "${marker}" >&2
  exit 1
fi
if [[ -d "${training_root}" ]] &&
   find "${training_root}" -mindepth 1 -print -quit | grep -q .; then
  printf 'refusing to initialize non-empty unmarked directory: %s\n' \
    "${training_root}" >&2
  exit 1
fi
if [[ "${dry_run}" != "1" ]]; then
  mkdir -p "${training_root}"
fi
REMOTE_CHECK

XYCAR_RSYNC_ARGS=(
  -a
  --delete-delay
  --human-readable
  --itemize-changes
  --omit-dir-times
  --protect-args
  --filter="merge ${XYCAR_AI_SCRIPT_DIR}/code-rsync-filter.rules"
)
if ((XYCAR_DRY_RUN)); then
  XYCAR_RSYNC_ARGS+=(--dry-run)
fi

rsync "${XYCAR_RSYNC_ARGS[@]}" \
  "${XYCAR_STAGE_DIR}/bundle/" \
  "${XYCAR_AI_TRAIN_SSH}:${XYCAR_AI_TRAIN_ROOT}/"

if ((XYCAR_DRY_RUN)); then
  printf 'Dry run complete; remote environment was not changed.\n'
  exit 0
fi

XYCAR_SOURCE_COMMIT="$(git -C "${XYCAR_MGW_ROOT}" rev-parse HEAD)"
if [[ -n "$(git -C "${XYCAR_MGW_ROOT}" status --porcelain -- ai/)" ]]; then
  XYCAR_SOURCE_DIRTY=true
else
  XYCAR_SOURCE_DIRTY=false
fi
XYCAR_SOURCE_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

ssh "${XYCAR_AI_TRAIN_SSH}" bash -s -- \
  "${XYCAR_AI_TRAIN_ROOT}" \
  "${XYCAR_AI_TRAIN_UV}" \
  "${XYCAR_AI_UV_VERSION}" \
  "${XYCAR_SOURCE_COMMIT}" \
  "${XYCAR_SOURCE_DIRTY}" \
  "${XYCAR_SOURCE_TIME}" <<'REMOTE_SYNC'
set -euo pipefail
training_root="$1"
uv_path="$2"
expected_uv="$3"
source_commit="$4"
source_dirty="$5"
source_time="$6"

[[ -f "${training_root}/.xycar-ai-workspace" ]] ||
  { printf 'workspace marker disappeared\n' >&2; exit 1; }
[[ -x "${uv_path}" ]] ||
  { printf 'uv is unavailable: %s\n' "${uv_path}" >&2; exit 1; }
actual_uv="$("${uv_path}" --version | awk '{print $2}')"
[[ "${actual_uv}" == "${expected_uv}" ]] ||
  {
    printf 'uv version mismatch: %s != %s\n' \
      "${actual_uv}" "${expected_uv}" >&2
    exit 1
  }

mkdir -p \
  "${training_root}/datasets/teleop" \
  "${training_root}/artifacts/models"
cd -- "${training_root}"
"${uv_path}" python install 3.12
"${uv_path}" lock --check
"${uv_path}" sync --locked --managed-python
printf '{\n  "mgw_commit": "%s",\n  "dirty": %s,\n  "synced_at": "%s"\n}\n' \
  "${source_commit}" "${source_dirty}" "${source_time}" \
  >.source-state.json
REMOTE_SYNC

printf 'Training code and uv environment synchronized to %s:%s\n' \
  "${XYCAR_AI_TRAIN_SSH}" "${XYCAR_AI_TRAIN_ROOT}"
