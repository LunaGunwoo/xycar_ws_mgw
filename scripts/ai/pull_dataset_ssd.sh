#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

XYCAR_APPLY=0
XYCAR_CHECKSUM=0
XYCAR_DIRECT=0
XYCAR_ALL=0
XYCAR_MIRROR=0
XYCAR_DIRECT_MIN_FORWARD_SPEED="20.0"
while (($#)); do
  case "$1" in
    --apply)
      XYCAR_APPLY=1
      ;;
    --checksum)
      XYCAR_CHECKSUM=1
      ;;
    --direct)
      XYCAR_DIRECT=1
      ;;
    --all)
      XYCAR_ALL=1
      ;;
    --mirror)
      XYCAR_MIRROR=1
      ;;
    -h|--help)
      printf 'usage: %s [--direct] [--all] [--mirror] [--apply] [--checksum]\n' "$0"
      printf 'default: dry-run; --direct reads a mounted D:\\teleop-style SSD source\n'
      printf '%s\n' \
        '--direct --all copies every non-active source entry without metadata filtering'
      printf 'direct mode preserves local-only files unless --mirror is specified\n'
      exit 0
      ;;
    *)
      xycar_ai_die "unknown argument: $1"
      ;;
  esac
  shift
done

((!XYCAR_MIRROR || XYCAR_DIRECT)) ||
  xycar_ai_die "--mirror is available only with --direct"
((!XYCAR_ALL || XYCAR_DIRECT)) ||
  xycar_ai_die "--all is available only with --direct"
((!XYCAR_ALL || !XYCAR_MIRROR)) ||
  xycar_ai_die "--all cannot be combined with --mirror"

xycar_ai_require_authoring_checkout
xycar_ai_require_command rsync
xycar_ai_validate_absolute_path \
  "${XYCAR_AI_LOCAL_DATASET_ROOT}" "local dataset root"

xycar_ai_build_rsync_args() {
  XYCAR_RSYNC_ARGS=(
    -rt
    --modify-window=1
    --omit-dir-times
    --human-readable
    --itemize-changes
    --info=progress2
    --partial
    --partial-dir=.rsync-partial
    --protect-args
    --filter="merge ${XYCAR_AI_SCRIPT_DIR}/dataset-rsync-filter.rules"
  )
  if ((!XYCAR_APPLY)); then
    XYCAR_RSYNC_ARGS+=(--dry-run)
  fi
  if ((XYCAR_CHECKSUM)); then
    XYCAR_RSYNC_ARGS+=(--checksum)
  fi
}

xycar_ai_pull_shared_dataset() {
  xycar_ai_require_shared_dataset_root
  xycar_ai_require_shared_mount
  xycar_ai_verify_shared_marker

  local shared_teleop="${XYCAR_AI_SHARED_DATASET_ROOT}/teleop"
  [[ -d "${shared_teleop}" ]] ||
    xycar_ai_die "shared teleop directory is missing: ${shared_teleop}"
  if ((XYCAR_APPLY)); then
    mkdir -p "${XYCAR_AI_LOCAL_DATASET_ROOT}"
  elif [[ ! -d "${XYCAR_AI_LOCAL_DATASET_ROOT}" ]]; then
    xycar_ai_die \
      "local dataset root is missing; run bootstrap_env.sh before this dry-run: ${XYCAR_AI_LOCAL_DATASET_ROOT}"
  fi

  xycar_ai_build_rsync_args
  rsync "${XYCAR_RSYNC_ARGS[@]}" \
    "${shared_teleop}/" \
    "${XYCAR_AI_LOCAL_DATASET_ROOT}/"

  if ((XYCAR_APPLY)); then
    printf 'Dataset pull applied: %s -> %s\n' \
      "${shared_teleop}" "${XYCAR_AI_LOCAL_DATASET_ROOT}"
  else
    printf 'Dataset pull dry-run complete; rerun with --apply to copy changes.\n'
  fi
}

xycar_ai_select_direct_sessions() {
  local root="$1"
  local allow_empty="$2"
  local -a selector_args=(
    --root "${root}"
    --min-forward-speed "${XYCAR_DIRECT_MIN_FORWARD_SPEED}"
  )
  if ((allow_empty)); then
    selector_args+=(--allow-empty)
  fi
  local selection_output
  selection_output="$(
    cd -- "${XYCAR_AI_BUNDLE_ROOT}"
    uv run --locked python -m xycar_ai.select_policy_sessions \
      "${selector_args[@]}"
  )" || xycar_ai_die "failed to select eligible direct SSD sessions under ${root}"
  XYCAR_SELECTED_SESSIONS=()
  if [[ -n "${selection_output}" ]]; then
    mapfile -t XYCAR_SELECTED_SESSIONS <<<"${selection_output}"
  fi
}

xycar_ai_delete_managed_session() {
  local session="$1"
  local target="${XYCAR_AI_LOCAL_DATASET_ROOT}/${session}"
  [[ "${session}" =~ ^[0-9]{8}_[0-9]{6}_[0-9]{3}_session(_[0-9]+)?$ ]] ||
    xycar_ai_die "refusing to delete an invalid session name: ${session}"
  [[ -d "${target}" && ! -L "${target}" ]] ||
    xycar_ai_die "managed session deletion target is unsafe: ${target}"
  find "${target}" -depth -mindepth 1 -delete
  rmdir -- "${target}"
}

xycar_ai_pull_direct_dataset() {
  xycar_ai_require_command find
  xycar_ai_require_command findmnt
  xycar_ai_require_command realpath
  xycar_ai_require_command uv
  xycar_ai_validate_absolute_path \
    "${XYCAR_AI_SSD_TELEOP_ROOT}" "direct SSD teleop root"
  [[ -d "${XYCAR_AI_SSD_TELEOP_ROOT}" ]] ||
    xycar_ai_die \
      "direct SSD source is missing; mount the Windows drive first: ${XYCAR_AI_SSD_TELEOP_ROOT}"
  local source_real
  source_real="$(realpath -e -- "${XYCAR_AI_SSD_TELEOP_ROOT}")"

  if [[ "${XYCAR_AI_ALLOW_NON_WINDOWS_SOURCE:-0}" != "1" ]]; then
    local source_filesystem source_mount
    source_filesystem="$(findmnt -n -o FSTYPE -T "${source_real}")" ||
      xycar_ai_die "cannot resolve direct SSD source filesystem"
    source_mount="$(findmnt -n -o TARGET -T "${source_real}")" ||
      xycar_ai_die "cannot resolve direct SSD source mount"
    xycar_ai_validate_external_windows_mount_values \
      "${source_filesystem}" "${source_mount}"
  fi

  [[ ! -L "${XYCAR_AI_LOCAL_DATASET_ROOT}" ]] ||
    xycar_ai_die "local dataset destination must not be a symlink"
  [[ -d "${XYCAR_AI_LOCAL_DATASET_ROOT}" ]] ||
    xycar_ai_die "local dataset destination is missing"
  local destination_real destination_filesystem
  destination_real="$(realpath -e -- "${XYCAR_AI_LOCAL_DATASET_ROOT}")"
  [[ "${source_real}" != "${destination_real}" ]] ||
    xycar_ai_die "direct SSD source and local destination must differ"
  destination_filesystem="$(findmnt -n -o FSTYPE -T "${destination_real}")" ||
    xycar_ai_die "cannot resolve local dataset filesystem"
  [[ "${destination_filesystem}" == "ext4" ]] ||
    xycar_ai_die \
      "local dataset destination must be on ext4: ${destination_filesystem}"
  xycar_ai_verify_local_dataset_marker "${destination_real}"

  if ((XYCAR_ALL)); then
    printf 'Selecting all non-active content without metadata filtering\n'
    xycar_ai_build_rsync_args
    XYCAR_RSYNC_ARGS+=(
      --exclude="/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}"
    )
    rsync "${XYCAR_RSYNC_ARGS[@]}" \
      "${source_real}/" \
      "${destination_real}/"
    if ((XYCAR_APPLY)); then
      printf 'Direct SSD all-content pull applied: %s -> %s\n' \
        "${source_real}" "${destination_real}"
    else
      printf 'Direct SSD all-content dry-run complete; rerun with --apply to copy changes.\n'
    fi
    return
  fi

  xycar_ai_select_direct_sessions "${source_real}" 0
  local -a source_sessions=("${XYCAR_SELECTED_SESSIONS[@]}")
  ((${#source_sessions[@]} > 0)) ||
    xycar_ai_die "no eligible direct SSD sessions were selected"

  local -A source_session_set=()
  local session destination_session
  for session in "${source_sessions[@]}"; do
    source_session_set["${session}"]=1
  done

  local -a removed_sessions=()
  if ((XYCAR_MIRROR)); then
    xycar_ai_select_direct_sessions "${destination_real}" 1
    for session in "${XYCAR_SELECTED_SESSIONS[@]}"; do
      if [[ -z "${source_session_set[${session}]:-}" ]]; then
        removed_sessions+=("${session}")
      fi
    done
  fi

  printf 'Selected %d completed gamepad session(s) with max_forward_speed >= %s\n' \
    "${#source_sessions[@]}" "${XYCAR_DIRECT_MIN_FORWARD_SPEED}"
  xycar_ai_build_rsync_args
  if ((XYCAR_MIRROR)); then
    XYCAR_RSYNC_ARGS+=(--delete-delay --delete-excluded)
  fi
  for session in "${source_sessions[@]}"; do
    destination_session="${destination_real}/${session}"
    [[ ! -L "${destination_session}" ]] ||
      xycar_ai_die "local dataset session must not be a symlink: ${destination_session}"
    rsync "${XYCAR_RSYNC_ARGS[@]}" \
      "${source_real}/${session}/" \
      "${destination_session}/"
  done

  for session in "${removed_sessions[@]}"; do
    printf '*deleting managed session %s\n' "${session}"
    if ((XYCAR_APPLY)); then
      xycar_ai_delete_managed_session "${session}"
    fi
  done

  local mode="safe pull"
  if ((XYCAR_MIRROR)); then
    mode="managed mirror"
  fi
  if ((XYCAR_APPLY)); then
    printf 'Direct SSD %s applied: %s -> %s\n' \
      "${mode}" "${source_real}" "${destination_real}"
  else
    printf 'Direct SSD %s dry-run complete; rerun with --apply to apply changes.\n' \
      "${mode}"
  fi
}

if ((XYCAR_DIRECT)); then
  xycar_ai_pull_direct_dataset
else
  xycar_ai_pull_shared_dataset
fi
