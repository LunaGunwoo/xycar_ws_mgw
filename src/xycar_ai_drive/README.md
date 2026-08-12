# xycar_ai_drive

고정 224x224 TorchScript ViT 정책으로 `/image_raw`를 계속 추론하고,
게임패드로 일반 AI 주행 또는 사람 보정 기반 고속 데이터 수집을 수행하는
ROS 2 패키지다.
모터 메시지는 `std_msgs/Float32MultiArray([angle, speed])`이며 ON/OFF와 관계없이
20 Hz로 발행한다.

## 동작 계약

- 시작 상태는 항상 OFF다. 시작 후 A를 한 번 놓은 다음 누르면 ON, 다시 누르면
  OFF가 되는 상승 edge toggle이다. A를 계속 누르고 있는 것은 추가 toggle이 아니다.
- 추론은 OFF 상태에서도 최신 camera frame으로 계속 수행한다. ON 상태에서만 최신
  유효 예측을 motor command로 사용하고, OFF 상태에서는 `[0, 0]`을 발행한다.
- AR artifact는 `(angle=0, speed=25)` class 네 쌍으로 history를 시작하고 성공한
  schema v2에서는 argmax 예측, schema v3에서는 실제 motor에 발행된 명령만
  queue에 추가한다. 추론됐지만 발행되지 않은 명령은 v3 history에 넣지 않는다.
  추론 실패 또는 성공한 추론 사이가 0.25초 이상 벌어지면 초기 history로 reset한다.
- A 버튼으로 ON이 되는 순간에도 AR history와 저장된 예측을 reset한다. reset 뒤
  새 camera frame의 첫 예측이 완료될 때까지 motor output은 `[0, 0]`을 유지한다.
- angle은 `angle_class_id - 100`이다. speed는
  `max(0, speed_class_id - 100)`이므로 reverse는 금지하지만 양수 예측에는 별도
  cap을 적용하지 않아 label 계약의 최대 `100`까지 전달한다.
- Joy 또는 camera prediction이 0.25초 이상 stale, 추론/변환 오류, motor
  subscriber 소실, 다른 motor publisher 출현 시 즉시 OFF와 `[0, 0]`으로
  전환한다. 조건이 복구돼도 A release 후 새 press가 있어야 다시 ON된다.
- Jetson의 양방향 parameter bridge `/ros_bridge`는
  `allowed_motor_relay_nodes`에 명시된 필수 relay라 경쟁 publisher 판정에서
  제외한다. Fast DDS가 bridge node 이름을 `UNKNOWN`으로 보고할 때는 같은 DDS
  participant GID를 가진 unnamed publisher/subscriber 쌍만 bridge로 인정한다.
  이름 없는 publisher 하나만 있거나 participant가 다른 경우와 `gamepad_teleop`을
  포함한 그 밖의 publisher는 계속 fail-closed로 주행을 차단한다.
- 기본 launch의 `allow_motion:=true`는 A toggle 후 실제로 움직일 수 있다.
  `allow_motion:=false`는 nonzero command를 차단하는 점검용 gate지만 node 자체가
  motor publisher이므로 실행 전 승인 규칙은 그대로 적용된다.

상태 확인용 `/front_cam_policy/prediction`은 `[angle, speed, inference_ms]`,
`/front_cam_policy/enabled`는 현재 toggle 상태를 발행한다. 이 debug topic을
실제 motor command의 대체 계약으로 사용하지 않는다.

## Model artifact

기본 artifact 경로는 다음과 같다.

```text
/home/xytron/xycar_ws_mgw/artifacts/models/
  front-cam-policy-baseline-e6-20260810/
    model.ts
    manifest.yaml
    SHA256SUMS
```

node는 시작 전에 checksum, input, angle/speed `[1,201]` tuple output,
normalization과 label decode 계약을 검증한다. schema v1 stateless artifact는 고정
RGB `[1,3,224,224]` 하나를 받고 계속 지원한다. schema v2 AR artifact는 RGB image와
int64 history `[1,4,2]` tuple을 받으며 history pair 순서는
`[angle_class_id, speed_class_id]`, 시간 순서는 오래된 값부터 최신 값까지다. 이후
schema v3는 같은 tensor shape를 사용하지만 history source를 실제 실행 명령으로
강제한다. CPU thread 8개로 model을 load하고 3회 warm-up한다. artifact 생성과 배포는 개발
Laptop의 MGW root에서 수행한다.

## 사람 보정 데이터 수집

`guided_policy_collector`는 모델의 angle/speed 두 head를 한 번에 추론하고 사람의
조향·속도 보정을 합쳐 하나의 motor publisher로 발행한다.

```text
executed_angle = clamp(model_angle + signed_left_stick * 200, -100, 100)
executed_speed = clamp(model_speed + RT * 2 - LT * 5, 0, speed_cap)
```

기본 Remote Gamepad에서는 조향 부호를 반전한다. 값은 모두 hold 동안만 적용되고
누적 trim으로 남지 않는다. CSV label은 모델 원본이 아니라 실제 발행된
`executed_angle/executed_speed`이며 model prediction, stick/trigger depth, residual,
사람 개입 여부와 inference latency도 함께 저장한다. angle/speed는 공통 ViT
backbone과 별도 출력 head를 공유하므로 한 번의 수집과 학습에서 동시에 다룬다.
첫 generation의 기본 `speed_cap=27`은 기존 speed 25 모델에 RT `+2`를 허용하기
위한 값이다. 다른 base model이면 그 모델의 검증된 속도와 이번 라운드의 증분에
맞춰 명시적으로 낮추거나 올린다.

- Y: DRIVE ON/OFF. 시작은 OFF이며 stick과 trigger를 중립으로 놓고 release 후
  눌러야 ON이 된다.
- A: 양수 speed 주행 중 새 session 녹화 시작.
- B: 주행을 유지하면서 현재 session 정상 저장.
- X: 주행을 유지하면서 최근 10 frame을 버리고 정상 저장.
- 녹화 중 Y OFF: 최근 10 frame을 버리고 정상 저장.
- stale Joy/camera, inference·writer 오류, motor subscriber 소실 또는 경쟁 motor
  publisher 출현: 즉시 정지하고 session을 incomplete로 마감한다.

raw session은 `recording_root_dir`에 모두 보존한다. `curriculum_generation`은 이
session을 어느 학습 세대로 취급할지 나타내며, `speed_cap`은 해당 라운드에서
사람과 모델이 합성한 전진 명령의 상한이다. 두 값은 launch 인자로 지정한다.

아래 명령은 camera, gamepad와 motor publisher를 시작하므로 매 실행 직전 사용자
승인이 필요하다. 바퀴 지지 또는 안전 주행 공간, motor 전원 차단 수단, Y와
`Ctrl+C` 정지, 경쟁 publisher 부재를 먼저 확인한다.

```bash
ssh xytron@xycar-gpu
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar
ros2 launch xycar_ai_drive jetson_guided_collection.launch.py \
  artifact_id:=<schema-v2-or-v3-artifact-id> \
  curriculum_generation:=1 speed_cap:=27.0 \
  use_camera:=true use_gamepad:=true allow_motion:=true
```

이미 준비된 `/image_raw` 또는 `/joy`를 재사용할 때만 각각
`use_camera:=false`, `use_gamepad:=false`를 지정한다. motor bridge는 이 launch가
시작하지 않는다. 종료는 Y로 DRIVE OFF한 뒤 `Ctrl+C` 순서다.

CUDA container 없이 host CPU inference를 점검할 때의 일반 launch는 다음과 같다.
이 명령도 camera와 motor publisher를 열 수 있어 동일한 실행 직전 승인이 필요하다.

```bash
ros2 launch xycar_ai_drive guided_policy_collection.launch.py \
  artifact_id:=<schema-v2-or-v3-artifact-id> \
  curriculum_generation:=1 speed_cap:=27.0 \
  inference_backend:=local inference_device:=cpu
```

camera와 Joy publisher가 이미 있고 collector만 직접 시작해야 할 때는 설치된
parameter file과 versioned artifact를 모두 명시한다.

```bash
ros2 run xycar_ai_drive guided_policy_collector --ros-args \
  --params-file /home/xytron/xycar_ws_mgw/install/xycar_ai_drive/share/xycar_ai_drive/config/guided_policy_collection.yaml \
  -p artifact_dir:=/home/xytron/xycar_ws_mgw/artifacts/models/<artifact-id> \
  -p curriculum_generation:=1 -p speed_cap:=27.0
```

기존 artifact의 `full_frame_bicubic_resize`와 새
`perspective_road_warp_then_bicubic_resize` geometry를 모두 지원한다. road-warp
artifact는 manifest에 정규화 source 사다리꼴, BEV 크기와 destination 경계를
내장하며 runtime은 학습과 같은 perspective warp를 RGB frame에 적용한 뒤
224x224로 resize한다. warp parameter가 누락되거나 범위를 벗어나면 node 시작을
거부한다. parameter 튜닝과 warp 학습 명령은 `ai/README.md`를 따른다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw/ai
/home/xytron/.local/bin/uv run --locked xycar-export-policy \
  --checkpoint artifacts/runs/front_cam_policy/baseline_seed20260810/best.pt \
  --artifact-id front-cam-policy-baseline-e6-20260810

cd /home/xytron/xycar_ws/apps/xycar_ws_mgw
./scripts/ai/deploy_model.sh front-cam-policy-baseline-e6-20260810 --dry-run
./scripts/ai/deploy_model.sh front-cam-policy-baseline-e6-20260810
```

`deploy_model.sh`는 versioned artifact만 배포하며 ROS node를 실행하지 않는다.

## Build와 테스트

실제 차량 checkout에서 다음과 같이 package만 build한다. 이 단계는 camera나
motor node를 실행하지 않는다.

```bash
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select xycar_ai_drive
colcon test --packages-select xycar_ai_drive --event-handlers console_direct+
source /home/xytron/xycar_ws_mgw/install/setup.bash
```

차량 Python에는 CPU PyTorch, NumPy, OpenCV, PyYAML과 ROS `cv_bridge`가 필요하다.
검증 당시 차량에는 PyTorch 2.6.0이 있었고, Ryzen 7 7730U synthetic memory-frame
benchmark에서 전처리 포함 steady-state p95는 11.802 ms였다. 이는 20 Hz budget
가능성만 확인한 값이며 실제 camera-to-motor 주행 검증은 아니다.

## 실차 실행

아래 명령들은 camera device와 motor publisher를 시작할 수 있으므로 매 실행 직전
사용자 승인이 필요하다. 승인 후에도 바퀴를 지면에서 분리하거나 안전 공간을
확보하고, `Ctrl+C`와 게임패드 A 재입력으로 정지할 수 있는지 먼저 확인한다.
동시에 `gamepad_teleop`, 다른 drive node 또는 motor publisher를 실행하지 않는다.

camera와 game controller까지 함께 시작하는 기본 실행:

```bash
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws_mgw/install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar
ros2 launch xycar_ai_drive front_cam_policy.launch.py
```

이미 `/image_raw` camera 또는 `/joy` publisher가 있으면 해당 장치를 중복으로
열지 않는다.

```bash
ros2 launch xycar_ai_drive front_cam_policy.launch.py \
  use_camera:=false use_gamepad:=false
```

두 publisher가 이미 있을 때 policy node만 직접 실행할 수도 있다.

```bash
ros2 run xycar_ai_drive front_cam_policy --ros-args \
  --params-file /home/xytron/xycar_ws_mgw/install/xycar_ai_drive/share/xycar_ai_drive/config/front_cam_policy.yaml \
  -p artifact_dir:=/home/xytron/xycar_ws_mgw/artifacts/models/front-cam-policy-baseline-e6-20260810
```

다른 versioned artifact를 선택할 때만 `artifact_id:=<id>`를 지정한다. 종료는
`Ctrl+C`이며 node는 종료 경로에서 정지 command를 반복 발행한다.

## Jetson CUDA inference

기존 차량의 기본 계약은 `inference_backend=local`, `inference_device=cpu`다.
Jetson에서는 ROS 2 Humble node를 host에 유지하고, NVIDIA PyTorch container의
CUDA policy server와 권한 `0600` Unix socket으로 RGB frame과 prediction을
교환한다. 새 parameter 계약은 다음과 같다.

- `inference_backend`: `local` 또는 `unix`
- `inference_device`: `cpu` 또는 `cuda`; server handshake와 반드시 일치
- `inference_socket_path`: policy server Unix socket
- `inference_rpc_timeout_sec`: 기본 0.20초이며 inference timeout보다 클 수 없음

artifact checksum·ID 또는 device가 다르거나 socket이 끊기고 응답이 timeout되면
CPU로 fallback하지 않는다. node는 inference failure로 처리해 motion OFF와
`[0,0]`을 발행한다. image build와 host 설치는
`deploy/jetson/README.md`를 따른다.

container image의 entrypoint와 같은 server를 직접 진단할 때의 executable은
아래와 같다. 실제 Jetson 운용은 device와 mount를 제한하는 wrapper를 사용한다.

```bash
ros2 run xycar_ai_drive front_cam_policy_gpu_server -- \
  --artifact-dir /artifacts/<artifact-id> \
  --socket-path /run/user/1000/xycar-ai/policy.sock \
  --device cuda
```

아래 wrapper는 camera, gamepad와 motor publisher를 시작할 수 있으므로 매 실행
직전 실차 승인이 필요하다. 바퀴 지지, 모터 전원 차단 수단, `Ctrl+C` 종료와
경쟁 publisher 부재를 먼저 확인한다.

```bash
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch xycar_ai_drive jetson_gpu_policy.launch.py \
  artifact_id:=front-cam-policy-tiny-hflip-p05-patience5-e5-20260811 \
  use_camera:=true use_gamepad:=true allow_motion:=true
```

이 Jetson 전용 launch는 checksum이 있는 versioned artifact와 CUDA server
container를 먼저 준비하고 Unix socket이 열린 뒤 camera, gamepad와 policy node를
시작한다. motor bridge는 시작하지 않으므로 별도 terminal의 `motor` 또는
`x27.desktop`으로 사용자가 직접 관리한다. 종료는 `Ctrl+C`이며 launch가
container도 함께 정리한다. 이미 camera나 gamepad publisher가 있으면
`use_camera:=false` 또는 `use_gamepad:=false`를 지정한다. 기존 `xycar-ai-gpu`
wrapper와 `ARTIFACT_ID=<id>` 방식도 호환을 위해 계속 지원한다.

wrapper는 camera를 열기 전에 host NumPy 1.x, OpenCV와 `cv_bridge` import를
검사한다. user site-packages의 NumPy 2.x가 ROS Humble ABI를 가리면 즉시 종료하며
camera나 CUDA container를 남기지 않는다. policy node가 오류로 종료돼도 같은
launch의 camera와 gamepad를 종료하고 CUDA container를 정리한다.
