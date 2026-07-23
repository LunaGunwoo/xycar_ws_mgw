#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

xycar_ai_require_authoring_checkout
xycar_ai_require_command curl
xycar_ai_require_command git

XYCAR_USER_HOME="$(getent passwd "$(id -u)" | cut -d: -f6)"
[[ -n "${XYCAR_USER_HOME}" ]] ||
  xycar_ai_die "could not determine the current user's home directory"
XYCAR_LOCAL_BIN="${XYCAR_USER_HOME}/.local/bin"
XYCAR_LOCAL_UV="${XYCAR_LOCAL_BIN}/uv"

if [[ ! -x "${XYCAR_LOCAL_UV}" ]]; then
  printf 'Installing uv %s in %s\n' \
    "${XYCAR_AI_UV_VERSION}" "${XYCAR_LOCAL_BIN}"
  curl -LsSf \
    "https://astral.sh/uv/${XYCAR_AI_UV_VERSION}/install.sh" |
    env \
      UV_INSTALL_DIR="${XYCAR_LOCAL_BIN}" \
      UV_NO_MODIFY_PATH=1 \
      sh
fi

XYCAR_INSTALLED_UV_VERSION="$("${XYCAR_LOCAL_UV}" --version | awk '{print $2}')"
[[ "${XYCAR_INSTALLED_UV_VERSION}" == "${XYCAR_AI_UV_VERSION}" ]] ||
  xycar_ai_die \
    "uv version mismatch: ${XYCAR_INSTALLED_UV_VERSION} != ${XYCAR_AI_UV_VERSION}"

cd -- "${XYCAR_AI_BUNDLE_ROOT}"
"${XYCAR_LOCAL_UV}" python install 3.12
"${XYCAR_LOCAL_UV}" lock --check
"${XYCAR_LOCAL_UV}" sync --locked --managed-python
"${XYCAR_LOCAL_UV}" run --locked python -c \
  'import sys; assert sys.version_info[:2] == (3, 12)'

printf 'uv environment ready: %s/.venv\n' "${XYCAR_AI_BUNDLE_ROOT}"
