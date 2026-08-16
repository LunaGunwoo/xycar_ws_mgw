#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export XYCAR_AI_VEHICLE_DATASET_ROOT=/home/xytron/xycar_data/competition_manual
export XYCAR_AI_LOCAL_DATASET_ROOT="${SCRIPT_DIR}/../../ai/datasets/competition_manual"
exec "${SCRIPT_DIR}/sync_dataset.sh" "$@"
