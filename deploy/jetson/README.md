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

Motor image `xycar/noetic-motor:jp6.2.1-noetic-steering-v3`부터 외부
`/xycar_motor` angle은 normalized percent `-100..100`이고 ROS 1 watchdog이
유효값만 `×0.5`로 변환해 내부 `/xycar_motor_safe` driver command
`-50..50`으로 발행한다. 범위 밖 angle, malformed, NaN/Inf, stale와 시간 역행은
clamp하지 않고 `[0,0]`으로 닫힌다. speed는 변환하지 않는다.
rollback은 motor image, 그 source와 일치하는 installed wrapper 및 steering artifact
계약을 한 세트로 되돌릴 때만 허용한다. legacy artifact와 normalized adapter 또는
normalized artifact와 legacy adapter를 혼합하지 않는다.

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
실시간 warp tuner의 기존 GUI layout을 위해 `python3-tk`, `python3-pil`과
`python3-pil.imagetk`도 host dependency로 설치한다.

GC300이 연결된 상태에서 `/dev/input/js0`와 ROS Joy 메시지는 다음처럼 확인한다.
실제 USB gamepad 접근이므로 하네스의 매 실행 전 승인 규칙을 적용한다.

```bash
ls -l /dev/input/js0
ros2 run joy game_controller_node --ros-args \
  -p device_id:=0 -p autorepeat_rate:=20.0
# 별도 terminal
ROS_DOMAIN_ID=7 ros2 topic echo /joy
```

`install_runtime.sh`는 기존 `~/.local/bin/motor`,
`~/.local/bin/xycar-ai-gpu`, `~/.local/bin/xycar-ai-competition`과
`~/xycar_ws/etc/gui-shell/x27.sh`를 timestamped
migration backup에 보존한 뒤, Desktop `x27.desktop`이 Jetson motor wrapper를
절대 경로로 실행하도록 설치한다. 설치되는 `x27.sh`는 오래된 workspace setup을
source하지 않고 `/home/xytron/.local/bin/motor`만 실행한다. GPU wrapper와 image lock은
`~/.local/lib/xycar-ai-gpu/`에 함께 복사하므로 source checkout 위치나 이후의
부분 빌드에 의존하지 않는다. motor wrapper와 lock도 같은 이유로
`~/.local/lib/xycar-motor/`에 복사한다.
기존 legacy profile은 그대로 보존하며 installer가 생성·수정·삭제하지 않는다.
normalized v2 profile 세 개는 `~/.config/xycar/`에 같은 이름의 파일이 없을 때만
설치한다. 차량에서 튜닝한 versioned profile도 이후 재설치에서 덮어쓰지 않는다.

Guided collector의 Controller 조향 takeover 계약에서는 새 versioned 외부
profile에 `max_steering_angle: 100.0`과
`steering_contract: normalized_percent_v2`가 반드시 있어야 한다. 기존 Guided
profile은 수정하지 않고 rollback/reference로 보존한다.

```text
~/.config/xycar/gamepad_stateless_manual_normalized_v2.yaml
~/.config/xycar/guided_stateless_collection_normalized_v2.yaml
~/.config/xycar/guided_policy_collection_normalized_v2.yaml
~/.config/xycar/competition_mission_collection_normalized_v2.yaml
```

기존 이름의 profile은 rollback 자료다. 새 수집 node는 versioned profile의
`steering_contract: normalized_percent_v2`가 없으면 시작을 거부한다.

학습용 `ai/uv.lock`은 4090 Laptop CUDA 환경이므로 Jetson에서 `uv sync`하지 않는다.
GPU image build context는 runtime package인 `src/xycar_ai_drive`로 제한하며 dataset,
artifact, 학습 output과 다른 workspace source를 Docker daemon에 보내지 않는다.
JetPack 6.2.1의 Tegra kernel에는 `CONFIG_IP_NF_RAW`가 없으므로 host provision은
Moby의 명시적 호환 환경변수 `DOCKER_INSECURE_NO_IPTABLES_RAW=1`을 systemd에
설정한다. 이 설정은 published bridge port의 LAN 격리를 낮추므로 Jetson runtime은
bridge port를 publish하지 않는다. motor/bridge는 host network, GPU는 network
none만 사용한다.
Motor bridge는 ROS 1 subscriber만 있을 때 type을 추론하지 못하는 dynamic bridge를
사용하지 않는다. `/ros1_bridge/topics`에 `/xycar_motor`와
`/xycar/parking/vesc_odom`의 양쪽 type을 고정한 parameter bridge를 사용한다.
`vesc_to_odom_node`는 VESC speed와 servo position command를 wheelbase `0.33 m`로
적분하며 `odom/base_link` frame의 `nav_msgs/Odometry`를 발행한다. TF는 주차
localizer/EKF 한 곳만 발행하도록 ROS 1 node에서는 끈다. `motor` wrapper readiness는
watchdog, motor, VESC driver, odometry node와 ROS 1/ROS 2 odometry 첫 sample을 모두 요구한다.
ROS 2 sample은 Fast DDS 검색 순서 차이를 허용하도록 2초 단위로 최대 15회만
재시도한다. 이 제한 안에 sample이 없으면 기존과 같이 zero cleanup 후 종료한다.
host와 container 사이 Fast DDS data path는
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
  params_file:=/home/xytron/.config/xycar/guided_policy_collection_normalized_v2.yaml \
  artifact_id:=<normalized-policy-artifact-id> \
  curriculum_generation:=1 speed_cap:=30.0 \
  use_camera:=true use_gamepad:=true allow_motion:=true
```

이 명령도 camera·gamepad·motor publisher를 시작하므로 실행마다 별도 실차 승인이
필요하다. `run_gpu_policy.sh`의 `HOST_POLICY_LAUNCH`는 허용된 host launch 선택용이며
GPU server의 network-none, versioned artifact와 Unix socket 안전 경계는 동일하다.

GPU server는 network와 hardware device 없이 실행되고, host Humble node와 권한
`0600` Unix socket으로만 통신한다. server 단절·timeout·artifact/device mismatch는
CPU fallback 없이 motion OFF와 `[0,0]`으로 처리한다.

기본 GPU runtime은 Base schema v6, angle-only fixed-speed schema v7, traffic bundle
schema v22와 두 policy 공유 CUDA IPC를 지원하는
`xycar/ai-drive:jp6.2.1-pytorch25.06-schema22-traffic-red-first-4s-tl35-v23` tag다. 기본 nice_adaptive artifact는
`front-cam-policy-vit-small-ar4-v2-nice-adaptive-joint-regression-sequence-init25-window5-20260821`
이며 실차 명령에서 `speed_cap:=25.0`을 명시한다. 일반 policy launch의 하위 호환
기본 cap `30`은 바꾸지 않는다. 기존 `xycar/ai-drive:jp6.2.1-pytorch25.06` image와
schema v5 분류 artifact는 rollback용으로 삭제하지 않는다.

GPU image 또는 `images.lock.env`를 변경한 배포는 image build만으로 끝내지 않는다.
`install_runtime.sh`로 wrapper와 lock을 함께 설치하고, camera나 motor를 시작하기
전에 source와 설치본이 같은지 확인한다. 설치본 lock이 이전 image를 가리키면
schema v22 artifact가 container 시작 직후 종료될 수 있다.

```bash
cd /home/xytron/xycar_ws_mgw
./deploy/jetson/install_runtime.sh
cmp deploy/jetson/images.lock.env \
  /home/xytron/.local/lib/xycar-ai-gpu/images.lock.env
grep -Fx 'GPU_IMAGE=xycar/ai-drive:jp6.2.1-pytorch25.06-schema22-traffic-red-first-4s-tl35-v23' \
  /home/xytron/.local/lib/xycar-ai-gpu/images.lock.env
docker image inspect xycar/ai-drive:jp6.2.1-pytorch25.06-schema22-traffic-red-first-4s-tl35-v23 >/dev/null
```

좌회전 전용 정사각형-warp artifact는 `speed_cap:=23.0`을 명시한다. 이 model-only
시험에는 좌회전 감지나 8초 timer가 없고 A hold 동안만 연속 추론한다.

```bash
ros2 launch xycar_ai_drive jetson_gpu_policy.launch.py artifact_id:=nice-shortcut-resnet18-squarewarp-speed23-45sessions-20260821 speed_cap:=23.0 use_camera:=true use_gamepad:=true allow_motion:=true
```

## Traffic shortcut bundle wrapper

`jetson_traffic_shortcut.launch.py`는 host의 NumPy `1.26.4`, ONNX Runtime `1.24.0`,
CUDA→CPU provider와 bundle checksum을 camera 시작 전에 검사한다. 실제 traffic
ONNX synthetic inference가 끝나면 network-none CUDA container 하나에 Base와
ResNet18을 모두 preload하고 두 socket을 연다. 두 server는 한 CUDA lock을 공유하고
host 통합 node만 선택된 policy를 호출한다. `install_runtime.sh`가 설치하는
`run_gpu_traffic_shortcut.sh`의 절대 경로를 사용하며 motor bridge는 시작하지 않는다.
managed camera 또는 gamepad가 종료되면 mission launch 전체를 종료한다. wrapper는
inner launch leader가 먼저 사라진 경우에도 남은 process group과 CUDA container를
정리하며, SIGINT/SIGTERM 제한 시간을 넘긴 고아 group은 마지막에 SIGKILL로 제거한다.
이전 `traffic_shortcut_policy` process가 남아 있으면 새 실행을 거부한다.
schema v22는 2026-08-24 fix speed-35 Base와 사람 보정 YOLO11s의 640 letterbox 결과에서 confidence `0.25` 이상
최고 box 하나만 선택한다. bundle 원본 폭 gate `35..225`는 checksum 검증과 함께
유지하고 host의 `signal_bbox_width_min_px` 기본 `40`을 적용해 실효 gate를
`40..225`로 만든다. 각 축 `15%` padded crop은
Pillow bilinear 416×128/ImageNet 전처리 후 `STOP/STRAIGHT/LEFT` CNN으로 분류한다.
softmax `0.50` 미만은 `UNKNOWN`이다. YOLO와 CNN은 같은 fresh frame에서 camera
sequence 간격 3 이상마다 함께 실행하고, 중간 frame에서는 cached CNN을 실행하지 않는다.
Base 구간과 shadow는 cap `35`, initial history `(0,35)×4`와 token `[50,85]×4`를
사용하고, shortcut 구간의 고정 speed `23`은 변경하지 않는다.
SDL `game_controller_node`의 LB `buttons[9]`와 A를 함께 누르면 Base 접근과 반복 신호
처리를 시작하며 A만 누르면 조우별 정지를 건너뛰되 LEFT shortcut 판정은 유지한다.
headless는 준비 완료 후 Base로 자동 출발한다. LEFT 성공 전에는 유효 bbox 폭에
도달한 매 조우에서 `[0,0]`을 발행하고, 첫 실제 정지 발행부터
`signal_stop_wait_sec` 기본 `1.0초`를 기다린다. dwell 중 class는 무시한다. 이후
첫 fresh STRAIGHT는 Base, LEFT는 한 제어 주기 정지 뒤 shortcut, STOP/UNKNOWN/no-box는
계속 정지다. STRAIGHT 출발 뒤 accepted bbox가 없는 camera frame 30개가 누적되면
다음 조우를 재무장한다. 별도 lap counter는 없다. 터미널은 raw class 변경 즉시 및
2 Hz heartbeat 외에 정지 trigger, dwell과 재무장 event를 출력한다.
LEFT 확정 뒤 `TRANSITION_STOP cycle=1/1`에 해당하는 `[0,0]`을 정확히 한 번
발행하고 다음 Shortcut prediction을 기다린다. shortcut이 실제 motor를 제어하는
동안 Base self-AR shadow를 계속 갱신하되 발행하지 않는다. 첫 Shortcut motor
command부터 `shortcut_duration_sec` 기본 `5.0초` 뒤 최신 0.50초 이내 shadow
command를 즉시 발행하며,
policy/post-reset age 0.50초 초과나 IPC 0.40초 timeout은 fallback 없이 정지한다.
dual-policy server의 history reset timeout도 0.50초로 맞춘다.
성공은 Base shadow 승격까지 끝난 뒤에만 기록하며 이후 같은 process에서는 STOP과
LEFT를 모두 무시하고 Base로 계속 주행한다. 시작 로그에는 bundle 원본 `35px/4초`와
실효 `40px/1초/5초`를 함께 표시한다. 기존 schema v21 35px gate/RED 비필수 초기 대기,
schema v20 4초/40px gate와 schema v19 8초 session-rearm,
schema v18 process당 1회 fix-Base GO1,
schema v17 기존 Base GO1, schema v16
stop5-go3/search3-classify3, schema v15
initial-wait-all5/search3-classify1, schema v14
initial-stop15/nonstop3, schema v13 `stop15-go15`, schema v12 `stop10-go30`, schema v11 `stop30-go30`, schema v9 `stop10-go15`, schema v8
`stop3-go15`과 schema v6 `votes2`
bundle은 rollback용으로 보존한다.

```bash
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch xycar_ai_drive jetson_traffic_shortcut.launch.py \
  bundle_id:=traffic-shortcut-nice-ada-very-fast-fix-speed35-regression-resnet18-4s-shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-tl35to225-red5first-go1-stoponce-leftsession1-search3-classify3-vote-yolo3-t500-45sessions-20260824 \
  signal_bbox_width_min_px:=40 signal_stop_wait_sec:=1.0 \
  shortcut_duration_sec:=5.0 \
  use_camera:=true use_gamepad:=false allow_motion:=true \
  use_monitor_gui:=false
```

camera, gamepad와 motor publisher를 시작하므로 실차 실행마다 별도 직전 승인을
받고 바퀴 지지/안전 공간, 전원 차단, A release·`Ctrl+C`, 경쟁 publisher 부재를
확인한다. Gamepad mode에서 반복 조우 정지를 사용하려면 `buttons[9]`인 LB를 누른 채
A를 누르고, 조우별 정지를 건너뛰려면 A만 누른다. headless는 위 명령으로 Base 접근을
자동 시작한다.

`use_monitor_gui:=true`에서는 ROS callback을 background executor가 계속 비우고
Tk는 기본 15 Hz로 최신 snapshot만 렌더한다. `LIVE CAMERA`에는 가장 최신 camera
frame만 bbox 없이 표시하고, `MODEL INPUT (EXACT)`에는 signal timestamp와 정확히
매칭된 inference frame에만 bbox와 확대 crop을 표시한다. 늦게 도착한 이전
timestamp/sequence는 최신 표시를 되돌리지 않는다. exact frame이 1초 이상
오래되면 `HISTORICAL/STALE`로 표시하며, 매칭 실패 시 다른 frame에 overlay하지
않고 placeholder를 표시한다.

## Competition bundle wrapper

`xycar-ai-competition`은 Base, signal과 shortcut model을 하나의 CUDA container에
모두 preload/warm-up한 뒤 competition host launch를 시작한다. 기본 mode는 motor
publisher가 없는 `signal_shadow`다.

```bash
COMPETITION_BUNDLE_ID=<competition-bundle-id> \
COMPETITION_RUN_MODE=signal_shadow \
xycar-ai-competition
```

camera device를 여는 shadow 실행도 매번 승인이 필요하다. 아래 moving mode는
camera, gamepad와 motor publisher를 시작하므로 별도 `motor` 실행을 포함해 각각
실행 직전 승인을 다시 받고 바퀴 지지/안전 공간, 전원 차단, A와 `Ctrl+C` 정지,
경쟁 publisher 부재를 확인한다.

```bash
COMPETITION_BUNDLE_ID=<competition-bundle-id> \
COMPETITION_RUN_MODE=shortcut_only ALLOW_MOTION=true \
xycar-ai-competition

COMPETITION_BUNDLE_ID=<competition-bundle-id> \
COMPETITION_RUN_MODE=combined ALLOW_MOTION=true \
xycar-ai-competition
```

moving mode는 DRIVE OFF로 시작하고 A release 뒤 rising edge로만 활성화된다.
`shortcut_only`는 persistent handoff 확인 뒤 자동 정지한다. wrapper는 GPU
container에 network나 hardware device를 주지 않고 artifact root를 read-only로
mount한다. competition socket은 기존 stateless policy socket과 분리된
`/run/user/<uid>/xycar-ai/competition.sock`을 사용한다.
