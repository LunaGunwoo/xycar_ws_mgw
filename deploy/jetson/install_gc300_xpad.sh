#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly EXPECTED_ARCH=aarch64
readonly EXPECTED_KERNEL=5.15.148-tegra
readonly XPAD_URL='https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/plain/drivers/input/joystick/xpad.c?h=v5.15.148'
readonly XPAD_SHA256=34a522dc1a3bdb8434334a1b1a67bde678adcaa0eafa14b38ee1f9c02d2070e7

if [ "$(uname -m)" != "${EXPECTED_ARCH}" ]; then
    echo '[ERROR] GC300 xpad module은 Jetson ARM64 host 전용입니다.' >&2
    exit 1
fi

kernel_release=$(uname -r)
if [ "${kernel_release}" != "${EXPECTED_KERNEL}" ]; then
    printf '[ERROR] 검증되지 않은 kernel입니다: expected=%s actual=%s\n' \
        "${EXPECTED_KERNEL}" "${kernel_release}" >&2
    exit 1
fi

header_root="/usr/src/linux-headers-${kernel_release}-ubuntu22.04_aarch64/3rdparty/canonical/linux-jammy/kernel-source"
required_header_files=(
    Makefile
    Module.symvers
    include/generated/autoconf.h
)
for relative_path in "${required_header_files[@]}"; do
    if [ ! -f "${header_root}/${relative_path}" ]; then
        printf '[ERROR] NVIDIA kernel header가 불완전합니다: %s\n' \
            "${header_root}/${relative_path}" >&2
        exit 1
    fi
done

build_dir=$(mktemp -d /tmp/xycar-gc300-xpad.XXXXXX)
trap 'rm -rf "${build_dir}"' EXIT

curl -fsSL --retry 3 --proto '=https' "${XPAD_URL}" \
    -o "${build_dir}/xpad.c"
printf '%s  %s\n' "${XPAD_SHA256}" "${build_dir}/xpad.c" \
    | sha256sum --check --status
patch --batch --forward --ignore-whitespace -d "${build_dir}" -p1 \
    < "${SCRIPT_DIR}/xpad-msi-gc300.patch"
install -m 0644 "${SCRIPT_DIR}/xpad-module.Makefile" \
    "${build_dir}/Makefile"

make -C "${header_root}" M="${build_dir}" modules

module_path="${build_dir}/xpad.ko"
vermagic=$(modinfo -F vermagic "${module_path}")
case "${vermagic}" in
    "${kernel_release} "*) ;;
    *)
        printf '[ERROR] xpad vermagic mismatch: %s\n' "${vermagic}" >&2
        exit 1
        ;;
esac
if ! modinfo "${module_path}" \
    | grep -F 'alias:          usb:v0DB0p*d*dc*dsc*dp*icFFisc5Dip01in*' \
        >/dev/null; then
    echo '[ERROR] built xpad module에 MSI GC300 USB alias가 없습니다.' >&2
    exit 1
fi

installed_module="/lib/modules/${kernel_release}/updates/xycar/xpad.ko"
if [ -f "${installed_module}" ] && cmp -s "${module_path}" "${installed_module}"; then
    echo "xpad module은 이미 최신 상태입니다: ${installed_module}"
else
    if grep -Eq '^xpad ' /proc/modules; then
        if pgrep -af '(^|/)(joy_node|game_controller_node)( |$)' >/dev/null; then
            echo '[ERROR] gamepad node가 xpad를 사용 중입니다. node 종료 후 다시 실행하세요.' >&2
            exit 1
        fi
        sudo modprobe -r xpad
    fi
    sudo install -D -m 0644 "${module_path}" "${installed_module}"
fi

sudo install -m 0644 "${SCRIPT_DIR}/xycar-xpad.conf" \
    /etc/modules-load.d/xycar-xpad.conf
sudo depmod -a "${kernel_release}"
sudo modprobe joydev
sudo modprobe xpad

if lsusb -d 0db0:c0f0 >/dev/null 2>&1; then
    if [ ! -e /dev/input/js0 ]; then
        echo '[ERROR] GC300은 연결됐지만 /dev/input/js0가 생성되지 않았습니다.' >&2
        exit 1
    fi
    printf 'GC300 xpad 준비 완료: %s (%s)\n' \
        /dev/input/js0 "$(cat /sys/class/input/js0/device/name)"
else
    echo 'xpad 설치 완료. GC300 연결 후 /dev/input/js0를 확인하세요.'
fi
