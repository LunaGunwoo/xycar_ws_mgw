# Jetson deployment runtime

JetPack 6.2.1 ARM64 차량 PC에서 ROS 2 Humble host, ROS 1 Noetic motor,
`ros1_bridge`, CUDA policy server를 재현하는 배포 자산이다. image base와 bridge
source는 `images.lock.env`의 digest·commit으로 고정한다.

운영 기준상 기존 x86 mini PC `xycar`는 CPU inference 비교·rollback용으로
보존하고, 앞으로의 model 배포와 실차 inference는 Jetson `xycar-gpu`의 CUDA GPU
runtime을 기본으로 한다. GPU 오류 시 `xycar`로 자동 fallback하지 않고 motion
OFF와 `[0,0]`으로 fail-closed한다.
두 PC는 fallback을 위해 같은 domain ID 7 설정을 보존하지만, Jetson runtime은
`ROS_LOCALHOST_ONLY=1`로 제한한다. 따라서 켜져 있는 기존 CPU PC의 camera, Joy,
bridge publisher가 Jetson의 절대 토픽에 섞이지 않는다. Jetson motor bridge
container는 host network를 사용하므로 같은 loopback graph에서 통신한다.
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

`provision_host.sh`는 JetPack 6.2.1의 `5.15.148-tegra` kernel header로 MSI
FORCE GC300 WIRELESS용 `xpad` module을 빌드한다. upstream Linux 5.15.148
source checksum을 검증하고 MSI vendor alias만 백포트하며, kernel이나 L4T
package를 교체하지 않는다. module은 `/lib/modules/.../updates/xycar/`에 설치되고
재부팅 때 자동 load된다. 검증된 kernel release와 다르면 설치를 중단한다.
같은 script가 설치하는 camera udev 규칙은 USB 장치의 V4L2 index 0 capture node만
`/dev/videoCAM`에 연결하며 index 1 보조 node는 제외한다.

GC300이 연결된 상태에서 `/dev/input/js0`와 ROS Joy 메시지는 다음처럼 확인한다.
실제 USB gamepad 접근이므로 하네스의 매 실행 전 승인 규칙을 적용한다.

```bash
ls -l /dev/input/js0
ros2 run joy game_controller_node --ros-args \
  -p device_id:=0 -p autorepeat_rate:=20.0
# 별도 terminal
ROS_DOMAIN_ID=7 ros2 topic echo /joy
```

`install_runtime.sh`는 기존 `~/.local/bin/motor`와
`~/.local/bin/xycar-ai-gpu`, `~/xycar_ws/etc/gui-shell/x27.sh`를 timestamped
migration backup에 보존한 뒤, Desktop `x27.desktop`이 Jetson motor wrapper를
절대 경로로 실행하도록 설치한다. GPU wrapper와 image lock은
`~/.local/lib/xycar-ai-gpu/`에 함께 복사하므로 source checkout 위치나 이후의
부분 빌드에 의존하지 않는다. motor wrapper와 lock도 같은 이유로
`~/.local/lib/xycar-motor/`에 복사한다.
stateless 수집 profile 두 개는 `~/.config/xycar/`에 파일이 없을 때만 설치한다.
차량에서 튜닝한 기존 profile은 이후 재설치에서도 덮어쓰지 않는다.

```text
~/.config/xycar/gamepad_stateless_manual.yaml
~/.config/xycar/guided_stateless_collection.yaml
```

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
`FASTDDS_BUILTIN_TRANSPORTS=UDPv4`로 고정한다. Fast DDS가 host ROS 2 process와
bridge container를 같은 machine의 shared-memory participant로 판단하므로 bridge는
`--ipc host`와 wrapper를 실행한 host 사용자의 UID/GID를 함께 사용한다. IPC가
분리되거나 bridge를 root로 실행해 shared-memory port가 `root:root 0644`가 되면
discovery endpoint는 보여도 host 사용자의 ROS 2 payload가 ROS 1으로 전달되지
않는다.

## 실차 실행

`motor`와 `xycar-ai-gpu`는 각각 motor/serial 또는 camera·gamepad·motor publisher를
시작한다. 매 실행 직전 사용자 승인을 받고 바퀴 지지, 전원 차단 수단, Ctrl+C
정지 경로와 경쟁 `/xycar_motor` publisher 부재를 확인한다. 둘 다 boot service로
등록하지 않는다.

```bash
motor
xycar-ai-gpu
```

기본 `xycar-ai-gpu`는 `front_cam_policy.launch.py`를 실행한다. 사람 보정 데이터
수집은 source checkout의 Jetson 전용 launch를 사용한다. 이 launch는 wrapper에
`HOST_POLICY_LAUNCH=guided_policy_collection.launch.py`를 전달하고 generation과
speed cap을 host collector까지 전달한다.

```bash
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch xycar_ai_drive jetson_guided_collection.launch.py \
  params_file:=/home/xytron/.config/xycar/guided_stateless_collection.yaml \
  artifact_id:=<schema-v1-stateless-artifact-id> \
  curriculum_generation:=1 speed_cap:=9.0 \
  use_camera:=true use_gamepad:=true allow_motion:=true
```

이 명령도 camera·gamepad·motor publisher를 시작하므로 실행마다 별도 실차 승인이
필요하다. `run_gpu_policy.sh`의 `HOST_POLICY_LAUNCH`는 허용된 host launch 선택용이며
GPU server의 network-none, versioned artifact와 Unix socket 안전 경계는 동일하다.

GPU server는 network와 hardware device 없이 실행되고, host Humble node와 권한
`0600` Unix socket으로만 통신한다. server 단절·timeout·artifact/device mismatch는
CPU fallback 없이 motion OFF와 `[0,0]`으로 처리한다.
