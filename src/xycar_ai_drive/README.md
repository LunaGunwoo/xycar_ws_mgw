# xycar_ai_drive

고정 224x224 TorchScript ViT 정책으로 `/image_raw`를 계속 추론하고,
게임패드로 일반 AI 주행 또는 사람 보정 기반 고속 데이터 수집을 수행하는
ROS 2 패키지다.
기존 stateless 경로의 모터 메시지는
`std_msgs/Float32MultiArray([angle, speed])`이며 ON/OFF와 관계없이 20 Hz로
발행한다. 새 history 경로는 별도 `xycar_msgs/XycarMotor`와 camera header를 써서
새 frame 추론 완료 즉시 한 번만 발행한다.

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
- 새 증분 수집 기본은 schema v1 stateless artifact다. AR schema v2/v3 경로는
  rollback 호환성만 유지하며 새 split이나 기본 실행 예시에 사용하지 않는다.
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
강제한다. schema v4는 `[0,0]` 네 쌍에서 시작하고 `sample_clock=camera_frame`,
`update=externally_executed_commands`를 추가로 강제한다. CPU thread 8개로 model을
load하고 3회 warm-up한다. artifact 생성과 배포는 개발 Laptop의 MGW root에서
수행한다.

## Camera-frame history 주행과 Guided 수집

`history_policy`는 schema v4 AR4 artifact 전용이다. 추론 worker는 한 번에 하나만
처리하며 처리 중 도착한 frame은 최신 한 장만 남긴다. prediction을
`/xycar_motor_command`로 발행한 뒤 같은 camera stamp의
`/xycar_motor_executed`가 확인될 때까지 다음 추론을 보류하고, 확인된 실제
angle/speed만 네-command history에 추가한다. 미발행 prediction과 echo 없는
prediction은 history에 들어가지 않는다. 중복·역순·불일치 echo, 0.25초 stale,
경쟁 publisher와 native gateway not-ready는 DRIVE OFF와 `[0,0]` reset 조건이다.

일반 history CUDA 주행 명령은 다음과 같다. 별도 terminal에서 먼저 승인된
`xycar_motor_native/vesc_motor.launch.py`를 실행해야 한다. 이 launch도 camera,
gamepad와 motor publisher를 시작하므로 실행 직전 다시 승인받고 바퀴 지지 또는
안전 공간, 전원 차단, A와 `Ctrl+C` 정지를 확인한다.

```bash
ros2 launch xycar_ai_drive jetson_history_policy.launch.py \
  artifact_id:=<schema-v4-history-artifact-id> \
  params_file:=/home/xytron/.config/xycar/history_policy.yaml \
  use_camera:=true use_gamepad:=true allow_motion:=true
```

30Hz 가능성 검증은 대표 schema v3 AR4 artifact도 사용할 수 있다. 이때 실제
수집이나 주행으로 오인하지 않도록 `allow_motion:=false force_speed_zero:=true
require_schema4:=false`를 함께 지정하고, 60초 profile에서 camera/inference/
command/executed와 native gateway의 ERPM command rate, skipped/duplicate/
out-of-order, 단계별 p95를 확인한다. 이 zero-only 실행도 motor publisher와
serial을 사용하므로 승인이 필요하다.

`history_guided_collector`는 같은 frame의 model angle/speed에 사람의 절대 조향과
RT/LT speed 보정을 합성하고, native execution echo가 확인된 값만
`history_guided`에 기록한다. 모든 generation은 `speed_cap=30`을 명시해야 하며
다른 값이면 launch 전에 거부한다. Y는 DRIVE ON/OFF, A는 녹화 시작, B는 저장,
X는 tail 폐기 후 저장이다.

```bash
ros2 launch xycar_ai_drive jetson_history_guided_collection.launch.py \
  artifact_id:=<validated-schema-v4-history-artifact-id> \
  params_file:=/home/xytron/.config/xycar/guided_history_collection.yaml \
  curriculum_generation:=0 speed_cap:=30 \
  use_camera:=true use_gamepad:=true allow_motion:=true
```

종료는 먼저 Y로 DRIVE OFF하고 `/xycar_motor_executed`의 speed 0을 확인한 뒤 Guided
launch `Ctrl+C`, 마지막으로 native motor launch `Ctrl+C` 순서다. 일반 history
주행은 A로 OFF한 뒤 같은 순서를 따른다. 기존 stateless/ROS 1 bridge launch와
동시에 실행하지 않는다.

host CPU 진단용 일반 launch와 component entry point는 다음과 같다. 모두 native
motor command publisher를 만들 수 있으므로 hardware graph에서는 실행 직전 승인을
받는다.

```bash
ros2 launch xycar_ai_drive history_policy.launch.py \
  artifact_id:=<schema-v4-history-artifact-id> allow_motion:=false
ros2 launch xycar_ai_drive history_guided_collection.launch.py \
  artifact_id:=<schema-v4-history-artifact-id> \
  curriculum_generation:=0 speed_cap:=30 allow_motion:=false
ros2 run xycar_ai_drive history_policy --ros-args \
  --params-file /home/xytron/.config/xycar/history_policy.yaml \
  -p artifact_dir:=/home/xytron/xycar_ws_mgw/artifacts/models/<artifact-id>
ros2 run xycar_ai_drive history_guided_collector --ros-args \
  --params-file /home/xytron/.config/xycar/guided_history_collection.yaml \
  -p artifact_dir:=/home/xytron/xycar_ws_mgw/artifacts/models/<artifact-id> \
  -p collection_profile_path:=/home/xytron/.config/xycar/guided_history_collection.yaml
```

## 사람 보정 데이터 수집

`guided_policy_collector`는 모델의 angle/speed 두 head를 한 번에 추론한다. 왼쪽
stick이 중립이면 모델 angle을 사용하고, stick을 움직이는 동안에는 모델 angle을
버리고 Controller의 절대 조향을 사용한다. speed는 모델 예측과 사람의 RT/LT
보정을 합쳐 하나의 motor publisher로 발행한다.

```text
executed_angle = (
  model_angle                                      if left_stick == 0
  clamp(signed_left_stick * max_steering_angle)   otherwise
)
executed_speed = clamp(
  model_speed
  + RT_depth * rt_speed_increment
  - LT_depth * lt_speed_decrement,
  0,
  speed_cap)
```

기본 Remote Gamepad에서는 조향 부호를 반전하고 `max_steering_angle=100`으로 전체
조향 범위를 사용한다. `game_controller_node`의 deadzone을 거친 `steering_axis`가
정확히 0일 때만 모델 조향을 유지한다. CSV label은 모델 원본이 아니라 실제 발행된
`executed_angle/executed_speed`이며 model prediction, stick/trigger depth,
`executed_angle - model_angle`, 사람 개입 여부와 inference latency도 함께 저장한다.
새 CSV field는 추가하지 않으며 학습 label은 계속 `angle`과 `speed`다.
angle/speed는 공통 ViT backbone과 별도 출력 head를 공유하므로 한 번의 수집과
학습에서 동시에 다룬다.
새 stateless curriculum의 모든 guided round는 `speed_cap=30`을 사용한다. 이 값은
목표 속도나 최소 속도가 아니라 합성 결과에 대한 hard ceiling이다. 세대별 실제
속도 분포는 parent model의 speed와 YAML의 RT/LT 보정량으로 점진적으로 넓히며,
사용자 지시 없이 cap을 30보다 낮추지 않는다. 예를 들어 speed 15 Base에 RT/LT
`+5/-5`를 적용하면 첫 round의 실행 범위는 대체로 10~20이지만 30까지는 계속
허용된다.

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
외부 profile `/home/xytron/.config/xycar/guided_stateless_collection.yaml`에는
`max_steering_angle`, trigger 증감, deadzone, 버튼, timeout과
`/home/xytron/xycar_data/stateless_guided` 저장 root를 둔다. session metadata는
profile 경로·SHA-256과 최종 적용된 보정, inference, recording, 안전 parameter를
기록한다. 수집 이미지는 30 Hz camera보다 충분한 writer 처리량을 확보하도록 기본
JPEG 품질 95로 저장하며 `recording_image_format`과
`recording_jpeg_quality`를 외부 profile에서 조정할 수 있다.

기존 `residual_gain` profile은 호환 실행하지 않는다. 배포 전에 해당 key를 제거하고
같은 위치에 `max_steering_angle: 100.0`을 명시해야 하며, legacy key가 남거나 새
key가 빠지면 collector가 DRIVE OFF 상태에서 시작을 거부한다.

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
  params_file:=/home/xytron/.config/xycar/guided_stateless_collection.yaml \
  artifact_id:=<schema-v1-stateless-artifact-id> \
  curriculum_generation:=1 speed_cap:=30.0 \
  use_camera:=true use_gamepad:=true allow_motion:=true
```

이미 준비된 `/image_raw` 또는 `/joy`를 재사용할 때만 각각
`use_camera:=false`, `use_gamepad:=false`를 지정한다. motor bridge는 이 launch가
시작하지 않는다. 종료는 Y로 DRIVE OFF한 뒤 `Ctrl+C` 순서다.

CUDA container 없이 host CPU inference를 점검할 때의 일반 launch는 다음과 같다.
이 명령도 camera와 motor publisher를 열 수 있어 동일한 실행 직전 승인이 필요하다.

```bash
ros2 launch xycar_ai_drive guided_policy_collection.launch.py \
  params_file:=/home/xytron/.config/xycar/guided_stateless_collection.yaml \
  artifact_id:=<schema-v1-stateless-artifact-id> \
  curriculum_generation:=1 speed_cap:=30.0 allow_motion:=false \
  inference_backend:=local inference_device:=cpu
```

camera와 Joy publisher가 이미 있고 collector만 직접 시작해야 할 때는 설치된
parameter file과 versioned artifact를 모두 명시한다.

```bash
ros2 run xycar_ai_drive guided_policy_collector --ros-args \
  --params-file /home/xytron/.config/xycar/guided_stateless_collection.yaml \
  -p collection_profile_path:=/home/xytron/.config/xycar/guided_stateless_collection.yaml \
  -p artifact_dir:=/home/xytron/xycar_ws_mgw/artifacts/models/<schema-v1-artifact-id> \
  -p curriculum_generation:=1 -p speed_cap:=30.0 -p allow_motion:=false
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
  --checkpoint artifacts/runs/front_cam_policy/<stateless-run>/best.pt \
  --artifact-id <schema-v1-stateless-artifact-id> \
  --require-schema-version 1

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
  artifact_id:=<schema-v1-stateless-artifact-id> \
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

## 대회 signal + shortcut 통합 runtime

Competition runtime은 하나의 bundle에서 기존 Base Lap, signal temporal policy와
shortcut temporal policy를 CUDA memory에 모두 preload하고 warm-up한다. 상태별로
필요한 branch만 실행하며 주행 도중 model file을 다시 load하지 않는다.

- `signal_shadow`: signal만 실행하며 motor publisher를 만들지 않는다.
- `normal`/`signal_stop`: Base와 signal을 함께 실행한다.
- `shortcut`: shortcut만 실행한다.
- `handoff_verify`: Base와 shortcut을 함께 실행한다.

FSM은 STOP(red/yellow) > LEFT > STRAIGHT 우선순위, stop 2/3 vote, go 4/5 vote,
결정 deadline unknown-stop, shortcut 1회 latch와 12초 timeout을 적용한다.
handoff는 REACQUIRE와 probability 0.9, Base/shortcut angle 차이 25 이하가 5 frame
연속일 때만 완료한다. `shortcut_only`에서는 완료 즉시 zero command와 DRIVE OFF,
`combined`에서는 Base로 복귀한다.

Unix GPU server를 직접 진단하는 entry point는 다음과 같다. 일반 운용은 Jetson
wrapper가 network-none container, read-only artifact와 mode `0600` socket을
구성한다.

```bash
ros2 run xycar_ai_drive competition_policy_gpu_server -- \
  --artifact-dir /artifacts/<competition-bundle-id> \
  --socket-path /run/user/1000/xycar-ai/competition.sock \
  --device cuda --warmup-count 3
```

recorded session replay는 camera와 motor를 열지 않는다. `signal_only`,
`shortcut_only`, `combined` 중 하나를 선택하고 transition, fault와 p50/p95/p99
latency JSON을 확인한다.

```bash
ros2 run xycar_ai_drive competition_replay -- \
  --artifact-dir /home/xytron/xycar_ws_mgw/artifacts/models/<competition-bundle-id> \
  --session /home/xytron/xycar_data/competition_manual/<session-id> \
  --run-mode combined --device cuda \
  --output /home/xytron/competition-replay.json
```

### 실시간 shadow와 실제 주행

설치된 `xycar-ai-competition` wrapper는 먼저 GPU container에서 세 model을
preload/warm-up하고 socket 준비를 확인한 뒤 host launch를 시작한다.
`signal_shadow`는 motor publisher가 없고 gamepad도 시작하지 않지만 camera device를
열기 때문에 매 실행 직전 승인이 필요하다.

```bash
COMPETITION_BUNDLE_ID=<competition-bundle-id> \
COMPETITION_RUN_MODE=signal_shadow \
xycar-ai-competition
```

아래 두 mode는 camera, gamepad와 motor publisher를 시작하므로 각각 매 실행 직전
사용자 승인이 필요하다. 별도 승인한 motor bridge, 바퀴 지지/안전 공간, 전원
차단, A와 `Ctrl+C` 정지, 경쟁 publisher 부재를 먼저 확인한다.

```bash
COMPETITION_BUNDLE_ID=<competition-bundle-id> \
COMPETITION_RUN_MODE=shortcut_only ALLOW_MOTION=true \
xycar-ai-competition

COMPETITION_BUNDLE_ID=<competition-bundle-id> \
COMPETITION_RUN_MODE=combined ALLOW_MOTION=true \
xycar-ai-competition
```

두 moving mode는 항상 DRIVE OFF로 시작하며 A release 뒤 rising edge가 있어야
움직인다. A 재입력, stale Joy/camera/inference, socket 오류, motor subscriber
유실과 경쟁 publisher는 zero command와 DRIVE OFF를 만든다. 자동 완료나 fault
뒤에도 A를 놓았다가 다시 눌러야 재활성화할 수 있다. CPU로 자동 fallback하지
않는다.

host launch를 직접 구성해야 할 때는 다음 명령을 사용할 수 있지만 GPU server가
같은 artifact와 socket으로 이미 준비돼 있어야 한다. 움직이는 실행은 동일한
승인 규칙을 적용한다.

```bash
ros2 launch xycar_ai_drive competition_policy.launch.py \
  artifact_id:=<competition-bundle-id> \
  run_mode:=signal_shadow allow_motion:=false \
  inference_socket_path:=/run/user/1000/xycar-ai/competition.sock
```

관측 topic은 `/competition_ai/enabled`, `/competition_ai/mode`,
`/competition_ai/active_command`, `/competition_ai/signal_probabilities`,
`/competition_ai/shortcut_state`, `/competition_ai/fault`다. signal probability 배열
순서는 approach, visible, readable, red, yellow, left, green, progress이고 shortcut
배열은 phase index와 handoff probability다. 전체 설계와 검증 단계는 상위
하네스의 `docs/competition_ai_architecture.md`를 따른다.
