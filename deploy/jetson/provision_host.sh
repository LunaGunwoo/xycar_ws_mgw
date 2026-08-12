#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [ "$(uname -m)" != "aarch64" ]; then
    echo "[ERROR] Jetson ARM64 host에서만 실행할 수 있습니다." >&2
    exit 1
fi
if [ "$(. /etc/os-release && echo "${VERSION_CODENAME}")" != "jammy" ]; then
    echo "[ERROR] ROS 2 Humble host는 Ubuntu 22.04 jammy여야 합니다." >&2
    exit 1
fi

sudo apt-get update
sudo apt-get install -y --no-remove \
    curl gnupg2 locales lsb-release software-properties-common
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
sudo add-apt-repository -y universe

sudo install -d -m 0755 /usr/share/keyrings
curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | sudo tee /usr/share/keyrings/ros-archive-keyring.gpg >/dev/null
architecture=$(dpkg --print-architecture)
codename=$(. /etc/os-release && echo "${UBUNTU_CODENAME}")
echo "deb [arch=${architecture} signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${codename} main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null

sudo apt-get update
packages=(
    build-essential
    cmake
    docker.io
    git
    libserial-dev
    patch
    python3-colcon-common-extensions
    python3-opencv
    python3-rosdep
    python3-serial
    python3-torch
    python3-vcstool
    python3-wxgtk4.0
    python3-yaml
    rsync
    ros-dev-tools
    ros-humble-ackermann-msgs
    ros-humble-asio-cmake-module
    ros-humble-camera-info-manager
    ros-humble-compressed-depth-image-transport
    ros-humble-compressed-image-transport
    ros-humble-cv-bridge
    ros-humble-desktop
    ros-humble-diagnostic-aggregator
    ros-humble-image-transport-plugins
    ros-humble-image-view
    ros-humble-joy
    ros-humble-rmw-fastrtps-cpp
    ros-humble-rosbridge-server
    ros-humble-rqt-runtime-monitor
    ros-humble-rviz-imu-plugin
    ros-humble-serial-driver
    ros-humble-tf-transformations
    ros-humble-usb-cam
    ros-humble-v4l2-camera
    ros-humble-web-video-server
    usbutils
    v4l-utils
)
missing=()
for package in "${packages[@]}"; do
    if ! apt-cache show "${package}" >/dev/null 2>&1; then
        missing+=("${package}")
    fi
done
if [ "${#missing[@]}" -ne 0 ]; then
    printf '[ERROR] arm64 apt package를 찾을 수 없습니다: %s\n' "${missing[*]}" >&2
    exit 1
fi
apt_simulation=$(mktemp)
if ! apt-get --simulate install "${packages[@]}" >"${apt_simulation}"; then
    cat "${apt_simulation}" >&2
    echo '[ERROR] apt dependency simulation failed.' >&2
    exit 1
fi
if grep -Eq '^Remv (nvidia-|libnvidia-|cuda-|jetpack|nvidia-l4t)' \
    "${apt_simulation}"; then
    cat "${apt_simulation}" >&2
    echo '[ERROR] NVIDIA/L4T package removal was proposed; aborting.' >&2
    exit 1
fi
sudo apt-get install -y --no-remove "${packages[@]}"
rm -f "${apt_simulation}"

"${SCRIPT_DIR}/install_gc300_xpad.sh"

if [ ! -e /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update

sudo systemctl enable --now docker
sudo groupadd -f docker
sudo usermod -aG dialout,docker "${USER}"
if command -v nvidia-ctk >/dev/null 2>&1; then
    sudo nvidia-ctk runtime configure --runtime=docker
else
    echo "[ERROR] JetPack의 nvidia-container-toolkit이 없습니다." >&2
    exit 1
fi

# JetPack 6.2.1's Tegra kernel does not provide CONFIG_IP_NF_RAW. Moby ships
# this explicit compatibility switch for such kernels. Runtime wrappers do
# not publish bridge ports: motor/bridge use host networking and GPU uses none.
sudo install -d -m 0755 /etc/systemd/system/docker.service.d
printf '%s\n' \
    '[Service]' \
    'Environment="DOCKER_INSECURE_NO_IPTABLES_RAW=1"' \
    | sudo tee /etc/systemd/system/docker.service.d/xycar-jetson.conf \
        >/dev/null
sudo dockerd --validate --config-file=/etc/docker/daemon.json
sudo systemctl daemon-reload
sudo systemctl restart docker

sudo install -m 0644 "${SCRIPT_DIR}/10-xycar.rules" \
    /etc/udev/rules.d/10-xycar.rules
sudo udevadm control --reload-rules

uv_version=0.11.24
uv_archive=uv-aarch64-unknown-linux-gnu.tar.gz
uv_url="https://github.com/astral-sh/uv/releases/download/${uv_version}/${uv_archive}"
uv_tmp=$(mktemp -d)
trap 'rm -rf "${uv_tmp}"' EXIT
curl -fL "${uv_url}" -o "${uv_tmp}/${uv_archive}"
curl -fL "${uv_url}.sha256" -o "${uv_tmp}/${uv_archive}.sha256"
(
    cd "${uv_tmp}"
    sha256sum --check "${uv_archive}.sha256"
    tar -xzf "${uv_archive}"
)
install -d -m 0755 "${HOME}/.local/bin"
install -m 0755 \
    "${uv_tmp}/uv-aarch64-unknown-linux-gnu/uv" \
    "${HOME}/.local/bin/uv"
install -m 0755 \
    "${uv_tmp}/uv-aarch64-unknown-linux-gnu/uvx" \
    "${HOME}/.local/bin/uvx"

managed_start='# >>> xycar jetson environment >>>'
managed_end='# <<< xycar jetson environment <<<'
if grep -Fq "${managed_start}" "${HOME}/.bashrc"; then
    sed -i \
        "\|^${managed_start}$|,\|^${managed_end}$|d" \
        "${HOME}/.bashrc"
fi
{
    echo "${managed_start}"
    echo 'source /opt/ros/humble/setup.bash'
    echo 'export ROS_DOMAIN_ID=7'
    echo 'export ROS_LOCALHOST_ONLY=1'
    echo 'export ROS_NAMESPACE=xycar'
    echo 'export RMW_IMPLEMENTATION=rmw_fastrtps_cpp'
    echo 'export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH'
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}'
    echo "${managed_end}"
} >> "${HOME}/.bashrc"

echo "Host provisioning 완료. docker/dialout group 반영을 위해 다시 로그인하세요."
