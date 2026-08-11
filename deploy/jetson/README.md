# Jetson deployment runtime

JetPack 6.2.1 ARM64 차량 PC에서 ROS 2 Humble host, ROS 1 Noetic motor,
`ros1_bridge`, CUDA policy server를 재현하는 배포 자산이다. image base와 bridge
source는 `images.lock.env`의 digest·commit으로 고정한다.

운영 기준상 기존 x86 mini PC `xycar`는 CPU inference 비교·rollback용으로
보존하고, 앞으로의 model 배포와 실차 inference는 Jetson `xycar-gpu`의 CUDA GPU
runtime을 기본으로 한다. GPU 오류 시 `xycar`로 자동 fallback하지 않고 motion
OFF와 `[0,0]`으로 fail-closed한다.
Bridge image는 전체 Humble desktop을 중복 빌드하지 않고 `ros1_bridge`의
build·exec dependency closure만 Focal 안에서 source build한다.
Focal GCC에서 필요한 `rmw/time.h`의 `<stdbool.h>` 호환 patch는 적용 전
`images.lock.env`의 RMW commit을 검증한다.

## 정적 설치 순서

아래 명령은 package 설치와 container build만 수행하며 hardware node를 시작하지
않는다.

```bash
cd /home/xytron/xycar_ws_mgw
./deploy/jetson/provision_host.sh
# group 변경 반영을 위해 logout/login
./deploy/jetson/build_images.sh
./deploy/jetson/build_noetic_ws.sh
./deploy/jetson/install_runtime.sh
```

`install_runtime.sh`는 기존 `~/.local/bin/motor`와
`~/xycar_ws/etc/gui-shell/x27.sh`를 timestamped migration backup에 보존한 뒤,
Desktop `x27.desktop`이 Jetson motor wrapper를 절대 경로로 실행하도록 설치한다.

학습용 `ai/uv.lock`은 4090 Laptop CUDA 환경이므로 Jetson에서 `uv sync`하지 않는다.
GPU image build context는 runtime package인 `src/xycar_ai_drive`로 제한하며 dataset,
artifact, 학습 output과 다른 workspace source를 Docker daemon에 보내지 않는다.
JetPack 6.2.1의 Tegra kernel에는 `CONFIG_IP_NF_RAW`가 없으므로 host provision은
Moby의 명시적 호환 환경변수 `DOCKER_INSECURE_NO_IPTABLES_RAW=1`을 systemd에
설정한다. 이 설정은 published bridge port의 LAN 격리를 낮추므로 Jetson runtime은
bridge port를 publish하지 않는다. motor/bridge는 host network, GPU는 network
none만 사용한다.
Motor bridge는 ROS 1 subscriber만 있을 때 type을 추론하지 못하는 dynamic bridge를
사용하지 않는다. `/ros1_bridge/topics`에 `/xycar_motor`의 양쪽 type을 고정한
parameter bridge를 사용하고, host와 container 사이 Fast DDS data path는
`FASTDDS_BUILTIN_TRANSPORTS=UDPv4`로 고정한다.

## 실차 실행

`motor`와 `xycar-ai-gpu`는 각각 motor/serial 또는 camera·gamepad·motor publisher를
시작한다. 매 실행 직전 사용자 승인을 받고 바퀴 지지, 전원 차단 수단, Ctrl+C
정지 경로와 경쟁 `/xycar_motor` publisher 부재를 확인한다. 둘 다 boot service로
등록하지 않는다.

```bash
motor
xycar-ai-gpu
```

GPU server는 network와 hardware device 없이 실행되고, host Humble node와 권한
`0600` Unix socket으로만 통신한다. server 단절·timeout·artifact/device mismatch는
CPU fallback 없이 motion OFF와 `[0,0]`으로 처리한다.
