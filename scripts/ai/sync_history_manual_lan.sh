#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

# This dedicated wrapper reuses the reviewed exact-mirror implementation while
# fixing every normally overridable endpoint to the history_manual contract.
xycar_ai_require_authoring_checkout
xycar_ai_require_command sed

if [[ "${1:-}" == '-h' || "${1:-}" == '--help' ]]; then
  printf 'usage: %s [--dry-run] [--checksum] [--allow-empty-source]\n' "$0"
  printf '%s\n' \
    'default: immediately mirror vehicle history_manual over the fixed direct LAN'
  exit 0
fi

readonly XYCAR_HISTORY_SOURCE='/home/xytron/xycar_data/history_manual'
readonly XYCAR_HISTORY_DESTINATION="${XYCAR_AI_BUNDLE_ROOT}/datasets/history_manual"

export XYCAR_AI_LAN_WINDOWS_INTERFACE='Ethernet'
export XYCAR_AI_LAN_WINDOWS_ADDRESS='192.168.50.1'
export XYCAR_AI_LAN_SUBNET='192.168.50.0/24'
export XYCAR_AI_LAN_VEHICLE_ADDRESS='192.168.50.2'
export XYCAR_AI_LAN_VEHICLE_SSH='xytron@192.168.50.2'
export XYCAR_AI_LAN_EXPECTED_HOSTNAME='xycar-gpu'
export XYCAR_AI_LAN_VEHICLE_DATASET_ROOT="${XYCAR_HISTORY_SOURCE}"
export XYCAR_AI_LAN_LOCAL_DATASET_ROOT="${XYCAR_HISTORY_DESTINATION}"

# The shared implementation's fixed-root block is intentionally stateless-only.
# All equivalent checkout, network, host, path, marker, ext4 and symlink checks
# still run; this wrapper fixes the two dataset roots before allowing that block
# to be skipped.
export XYCAR_AI_ALLOW_ANY_CHECKOUT=1
"${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" "$@" |
  sed -e 's/stateless manual/history manual/g' \
      -e 's/stateless_manual/history_manual/g'
