# xycar_ai_drive

고정 224x224 TorchScript ViT 또는 ResNet18 정책으로 `/image_raw`를 계속 추론하고,
게임패드로 일반 AI 주행 또는 사람 보정 기반 고속 데이터 수집을 수행하는
ROS 2 패키지다.
모터 메시지는 `std_msgs/Float32MultiArray([angle, speed])`이며 ON/OFF와 관계없이
20 Hz로 발행한다.

새 moving runtime의 angle은 normalized percent `-100..100`이다. ROS 1 safety
adapter가 `×0.5`로 변환한 driver command `-50..50`만
`/xycar_motor_safe`에 발행한다. artifact manifest에 정확한
`normalized_percent_v2` 계약이 없으면 일반 AI node는 시작 후에도 DRIVE ON을
거부하고 Guided collector는 parent artifact로 받아들이지 않는다.

## 동작 계약

- 시작 상태는 항상 OFF다. A를 누르는 동안에만 DRIVE ON이고 놓으면 DRIVE OFF다.
  실측된 controller의 빠른 `1/0` 반복 입력이 주행을 끊지 않도록 마지막 A press 뒤
  `a_release_grace_sec: 0.12` 이내의 짧은 false pulse는 무시한다. 실제 release가
  계속되면 grace가 지난 첫 제어 주기에 OFF가 된다.
- 추론은 OFF 상태에서도 최신 camera frame으로 계속 수행한다. ON 상태에서만 최신
  유효 예측을 motor command로 사용하고, OFF 상태에서는 `[0, 0]`을 발행한다.
- legacy AR artifact는 `(angle=0, speed=25)` class 네 쌍으로 history를 시작하고
  schema v2에서는 argmax 예측, schema v3에서는 실제 motor에 발행된 명령만
  queue에 추가한다. schema v5 compact AR은 artifact에 따라 UNKNOWN 또는
  canonical 실제 명령 `(angle=0, dataset mean speed)` 네 쌍으로 시작하고 실제 발행 명령을 angle token `0..100`, speed
  token `50..80`으로 바꿔 오래된 slot부터 교체한다. 추론됐지만 발행되지 않은 명령은
  schema v3/v5/v6 history에 넣지 않는다. schema v6도 compact history를 사용하지만
  연속 angle/speed를 범위 검사한 뒤 실제 발행 명령을 round해 token으로 만든다.
  추론 실패 또는 성공한 추론 사이가 0.25초 이상 벌어지면 초기 history로 reset한다.
- A hold로 ON이 되는 순간에도 AR history와 저장된 예측을 reset한다. reset 뒤
  새 camera frame의 첫 예측이 완료될 때까지 motor output은 `[0, 0]`을 유지한다.
- 새 stateless 증분 수집 기본은 schema v1 artifact다. 이 수집 curriculum은
  nice_adaptive 기본 주행 정책과 별도다. AR schema v2/v3 경로와 schema v5 분류형은
  rollback 호환성만 유지하고, nice_adaptive 일반 주행은 schema v6 회귀형을 쓴다.
- schema v1/v2/v3 angle은 `angle_class_id - 100`이다. speed는
  `max(0, speed_class_id - 100)`이므로 reverse는 금지하지만 양수 예측에는 별도
  cap을 적용하지 않아 label 계약의 최대 `100`까지 전달한다. schema v5 angle은
  `(class_id - 50) ÷ 0.5`로 normalized 명령을 복원하고 speed는 class `0..30`을
  그대로 사용한다.
- schema v6는 float scalar `angle_driver [-50,50]`과 manifest가 선언한
  `speed [0,maximum]`을 받고 angle을 `×2`하여 normalized motor 명령으로 바꾼다.
  maximum은 양의 정수 `1..50`이며 기존 artifact의 기본값은 `30`이다. shape,
  NaN/Inf 또는 범위 위반은 fail-closed다. launch `speed_cap` 기본값은 `30`이고
  해당 artifact maximum을 넘을 수 없다. external AR history에도 이 capped 실제
  명령만 넣는다.
- schema v7은 정사각형 road warp RGB `[1,3,224,224]`와 명시 속도 `[1,1]`을
  받아 normalized angle `[1,1]`만 출력한다. manifest의 고정 속도를 `25`로 나눠
  model input에 넣고 angle은 `×100`한다. 실제 `speed_cap`이 고정 속도와 다르면
  입력 속도와 실행 속도가 어긋나므로 node 시작을 거부한다.
- Joy 또는 camera prediction이 0.25초 이상 stale, 추론/변환 오류, motor
  subscriber 소실, 다른 motor publisher 출현 시 즉시 OFF와 `[0, 0]`으로
  전환한다. 조건이 복구돼도 A release 후 새 press가 있어야 다시 ON된다.
- Jetson의 양방향 parameter bridge `/ros_bridge`는
  `allowed_motor_relay_nodes`에 명시된 필수 relay라 경쟁 publisher 판정에서
  제외한다. Fast DDS가 bridge node 이름을 `UNKNOWN`으로 보고할 때는 같은 DDS
  participant GID를 가진 unnamed publisher/subscriber 쌍만 bridge로 인정한다.
  이름 없는 publisher 하나만 있거나 participant가 다른 경우와 `gamepad_teleop`을
  포함한 그 밖의 publisher는 계속 fail-closed로 주행을 차단한다.
- 기본 launch의 `allow_motion:=true`는 A를 누르는 동안 실제로 움직일 수 있다.
  `allow_motion:=false`는 nonzero command를 차단하는 점검용 gate지만 node 자체가
  motor publisher이므로 실행 전 승인 규칙은 그대로 적용된다.

상태 확인용 `/front_cam_policy/prediction`은 `[angle, speed, inference_ms]`,
`/front_cam_policy/enabled`는 현재 A hold drive gate 상태를 발행한다. 이 debug topic을
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

node는 시작 전에 checksum, input, schema별 output,
normalization, label decode와 steering 계약을 검증한다. schema v1 stateless artifact는 고정
RGB `[1,3,224,224]` 하나를 받고 계속 지원한다. schema v2 AR artifact는 RGB image와
int64 history `[1,4,2]` tuple을 받으며 history pair 순서는
`[angle_class_id, speed_class_id]`, 시간 순서는 오래된 값부터 최신 값까지다. 이후
schema v3는 같은 tensor shape를 사용하지만 history source를 실제 실행 명령으로
강제한다. schema v5는 `history_token_ids [1,4,2]`, angle `[1,101]`, speed `[1,31]`,
UNKNOWN 또는 canonical `(0,dataset mean speed)×4` 초기화와 공용 numeric token
mapping을 검증하고 실제 발행 history만 받는다. canonical 명령은 embedding
경계에서만 내부 token ID로 encode되며 물리 명령의 초기 의미는 바뀌지 않는다.
Angle-only schema v5 artifact는 tuple shape를 바꾸지 않고 speed logits를 검증된
dataset 고정 class로 강제하며, manifest의 `training_objective.speed_output_trained`와
`speed_output`에 미학습 head와 실제 고정 출력을 함께 기록한다.
schema v6는 같은 compact history input과 manifest의 canonical 초기 명령을
사용하고, tuple output을 driver angle `[1,1]`, speed `[1,1]` float scalar로
바꾼다. manifest의 단위·범위·`angle_driver × 2` mapping을 검사하고 speed output
maximum과 history token 상한이 일치하지 않거나 runtime 출력이 범위를 벗어나면
발행하지 않는다. 기존 `(0,25)×4`/`[50,75]×4`, speed maximum 30 artifact와
`(0,35)×4`/`[50,85]×4`, speed maximum 35 artifact를 모두 지원한다.
schema v7은 ResNet18 `image + speed/25 → angle` 전용 계약이다. 기존 normalized
road-warp와 ImageNet 정규화를 사용하며 speed를 예측하거나 AR history를 만들지
않는다. angle tensor shape, finite 값과 `[-1,1]` 범위를 검사한 뒤 `×100`으로
normalized steering을 복원한다.
CPU thread 8개로 model을 load하고 3회 warm-up한다. artifact 생성과 배포는 개발
Laptop의 MGW root에서 수행한다.

## Jetson 실시간 road-warp 튜너

`live_warp_tuner`는 `/image_raw`를 sensor QoS로 구독해 현재 원본 영상의 source
사다리꼴과 perspective BEV 결과를 보여 준다. 기존 오프라인 warp tuner와 동일한
Tkinter layout을 사용해 왼쪽에는 원본 ROI와 warped preview, 오른쪽에는 각
parameter 이름, 실제 실수값 slider와 `bev_width`/`bev_height` 정수 입력칸을
표시한다. slider를 움직이면 현재 camera frame의 두 preview에 즉시 반영된다.

- `Space`: 현재 frame 일시정지/실시간 영상 복귀
- `S`: 현재 유효 parameter를 YAML로 저장
- `R`: 마지막 저장값으로 preview 초기화
- `Q` 또는 `Esc`: 종료

이 노드는 Joy를 구독하거나 motor topic을 publish하지 않는다. 다만 아래 기본
launch는 실제 camera device를 여므로 **매 실행 직전 사용자 승인이 필요하다.**
Jetson의 local GNOME desktop 또는 `docs/jetson_operations.md`에 설명한 RDP 공유
화면의 terminal에서 실행한다. 다른 process가 camera를 사용하지 않는지도 먼저
확인한다.

```bash
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws_mgw/install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar
ros2 launch xycar_ai_drive live_warp_tuner.launch.py use_camera:=true
```

이미 `/image_raw` publisher가 실행 중일 때만 device 중복 접근을 피하도록
`use_camera:=false`를 쓴다.

```bash
ros2 launch xycar_ai_drive live_warp_tuner.launch.py use_camera:=false
```

기존 publisher를 재사용하면서 launch 없이 tuner node만 실행할 수도 있다. 이
명령 자체는 camera driver를 시작하지 않는다.

```bash
ros2 run xycar_ai_drive live_warp_tuner --ros-args \
  -p camera_topic:=/image_raw \
  -p initial_config_path:=/home/xytron/xycar_ws_mgw/ai/config/front_cam_policy_preprocess.yaml \
  -p output_config_path:=/home/xytron/.config/xycar/front_cam_policy_preprocess.yaml
```

매 실행의 초기값은 checkout의 read-only canonical 학습 설정
`/home/xytron/xycar_ws_mgw/ai/config/front_cam_policy_preprocess.yaml`에서 읽는다.
차량 checkout을 수정하지 않도록 `S`의 기본 출력은
`/home/xytron/.config/xycar/front_cam_policy_preprocess.yaml`이다. 출력이 이미
있어도 다음 실행은 canonical 학습 설정에서 다시 시작한다. 출력은 개발 PC에서
검토하기 위한 candidate이며 runtime이나 다음 학습에 자동 반영되지 않는다. 다른
topic이나 경로가 필요하면
`camera_topic:=...`, `initial_config_path:=...`, `output_config_path:=...` launch
인자로 명시한다. 저장한 값을 학습 source에 반영하는 절차는 `ai/README.md`를
따른다.

고정 canonical 값은 `top_y=0.500`, `bottom_y=0.933`,
`top_left_x=0.340`, `top_right_x=0.660`, `bottom_left_x=0.000`,
`bottom_right_x=1.000`, `bev_width=224`, `bev_height=224`,
`dst_left_x=0.000`, `dst_right_x=1.000`이다. 실제 policy runtime은 GUI output이
아니라 배포 artifact manifest에 내장된 동일한 road-warp 계약을 사용한다.

## 사람 보정 데이터 수집

`guided_policy_collector`는 모델의 angle/speed 두 head를 한 번에 추론한다. RB를
놓으면 왼쪽 stick 위치와 무관하게 모델 angle을 사용하고, RB를 누르는 동안에만
Controller의 왼쪽 stick 절대 조향을 사용한다. speed는 RB 상태와 무관하게 모델
예측과 사람의 RT/LT 보정을 합쳐 하나의 motor publisher로 발행한다.

```text
executed_angle = (
  clamp(signed_left_stick * max_steering_angle)   if RB is held
  model_angle                                      otherwise
)
executed_speed = clamp(
  model_speed
  + RT_depth * rt_speed_increment
  - LT_depth * lt_speed_decrement,
  0,
  speed_cap)
```

Angle-only checkpoint의 raw speed head는 Guided에 사용하지 않는다. 고정 speed로
export된 artifact만 예외이며, 이 경우 `model_speed`는 학습된 head가 아니라
manifest `speed_output.class_id`가 지정한 상수다. 현재 teleop_15 angle-only
artifact는 이를 `15`로 고정하므로 기본 profile에서 실제 합성 범위는 LT full
`10`, trigger 중립 `15`, RT full `17`이고 hard ceiling은 계속 `30`이다. angle
class 자체는 driver `-50..50`이며 runtime이 normalized `-100..100`으로 복원한다.
현재 artifact를 만든 PyTorch 2.8 TorchScript는 Jetson host의 system PyTorch 1.8과
archive 호환이 되지 않는다. 실제 Guided에는 검증된 PyTorch 2.8 CUDA container를
쓰는 `jetson_guided_collection.launch.py`만 사용하고 아래 host-local CPU 점검
명령에는 이 artifact를 넣지 않는다.

기본 Remote Gamepad에서는 조향 부호를 반전하고 `max_steering_angle=100`으로 전체
조향 범위를 사용하며 `game_controller_node`의 RB `buttons[10]`을 hold takeover로
쓴다. RB를 놓은 상태에서 들어오는 `steering_axis`는 조향에 적용하지 않는다. CSV
label은 모델 원본이 아니라 실제 발행된 `executed_angle/executed_speed`이며 model
prediction, stick/trigger depth, `executed_angle - model_angle`, 실제 적용된 사람
개입 여부와 inference latency도 함께 저장한다.
새 CSV field는 추가하지 않으며 학습 label은 계속 `angle`과 `speed`다.
angle/speed는 공통 ViT backbone과 별도 출력 head를 공유하므로 한 번의 수집과
학습에서 동시에 다룬다.
새 stateless curriculum의 모든 guided round는 `speed_cap=30`을 사용한다. 이 값은
목표 속도나 최소 속도가 아니라 합성 결과에 대한 hard ceiling이다. 세대별 실제
속도 분포는 parent model의 speed와 YAML의 RT/LT 보정량으로 점진적으로 넓히며,
사용자 지시 없이 cap을 30보다 낮추지 않는다. 예를 들어 speed 15 Base에 기본
RT/LT `+2/-5`를 적용하면 첫 round의 실행 범위는 10~17이지만 30까지는 계속
허용된다.

- Y: DRIVE ON/OFF. 시작은 OFF이며 RB와 trigger를 놓고 Y를 release 후 눌러야
  ON이 된다. RB를 놓았다면 왼쪽 stick 위치는 ON 조건에 영향을 주지 않는다.
  녹화 중 다시 누르면 즉시 zero command와 DRIVE OFF로 전환한다.
- RB: 누르는 동안만 왼쪽 stick 절대 조향으로 takeover한다. 놓으면 다음 control
  tick부터 모델 조향으로 돌아간다. RT/LT 속도 보정은 두 모드에서 모두 동작한다.
- A: 양수 speed 주행 중 새 session 녹화 시작.
- B: 주행을 유지하면서 현재 session 정상 저장.
- X: 즉시 zero command와 DRIVE OFF로 전환하면서 현재 session 전체 삭제.
  metadata나 incomplete directory를 남기지 않으며 복구할 수 없다.
- 녹화 중 Y 긴급 정지: 최근 10 frame을 버리고 나머지를
  `stop_reason=y_emergency_stop`으로 정상 저장.
- stale Joy/camera, inference·writer 오류, motor subscriber 소실 또는 경쟁 motor
  publisher 출현: 즉시 정지하고 session을 incomplete로 마감한다.

정상 저장하거나 fault로 마감한 raw session은 `recording_root_dir`에 모두
보존한다. 사용자가 X로 명시적으로 삭제한 현재 session만 이 보존 대상이 아니다.
`curriculum_generation`은 session을 어느 학습 세대로 취급할지 나타내며,
`speed_cap`은 해당 라운드에서
사람과 모델이 합성한 전진 명령의 상한이다. 두 값은 launch 인자로 지정한다.
외부 base profile
`/home/xytron/.config/xycar/guided_policy_collection_normalized_v2.yaml`에는
`max_steering_angle`, `steering_takeover_button`, trigger 증감, deadzone, 버튼, timeout과
AR4용 기본 저장 root `/home/xytron/xycar_data/teleop_15`를 둔다. session 순서와
`initial_history_token_ids`, frame별 실제 실행 angle/speed가 함께 저장되므로 학습 시
네 개의 실행 명령 history를 복원할 수 있다. 기존
`guided_stateless_collection_normalized_v2.yaml`과 `stateless_guided` root는
stateless curriculum rollback/reference로만 보존한다.
session metadata는
profile 경로·SHA-256과 최종 적용된 보정, inference, recording, 안전 parameter를
기록한다. 수집 이미지는 30 Hz camera보다 충분한 writer 처리량을 확보하도록 기본
JPEG 품질 95로 저장하며 `recording_image_format`과
`recording_jpeg_quality`를 외부 profile에서 조정할 수 있다.

기존 `residual_gain` 또는 steering 계약이 없는 profile은 호환 실행하지 않는다.
새 versioned profile에는 `max_steering_angle: 100.0`,
`steering_takeover_button: 10`, `steering_contract: normalized_percent_v2`가 모두
있어야 한다. Jetson과 일반 Guided launch는 이 외부 profile 계약을 camera,
gamepad 또는 CUDA wrapper를 시작하기 전에 검사하고, 누락되면 실행 전체를
거부한다.

아래 명령은 camera, gamepad와 motor publisher를 시작하므로 매 실행 직전 사용자
승인이 필요하다. 바퀴 지지 또는 안전 주행 공간, motor 전원 차단 수단, Y와
`Ctrl+C` 정지, 경쟁 publisher 부재를 먼저 확인한다.

```bash
ssh xytron@xycar-gpu
set -euo pipefail
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar
GENERATION=<next-generation>
COLLECTION_ID=<generation-and-collection-id>
PROFILE=/home/xytron/.config/xycar/guided_collections/generation_${GENERATION}/${COLLECTION_ID}.yaml
test -f "${PROFILE}"
grep -Fqx '    steering_takeover_button: 10' "${PROFILE}"
ros2 launch xycar_ai_drive jetson_guided_collection.launch.py \
  params_file:="${PROFILE}" \
  artifact_id:=<versioned-policy-artifact-id> \
  curriculum_generation:="${GENERATION}" speed_cap:=30.0 \
  use_camera:=true use_gamepad:=true allow_motion:=true
```

collection별 profile 생성과 `recording_root_dir` 검증 명령은 workspace root
`COMMANDS.md`의 Guided 절차를 그대로 사용한다.

이미 준비된 `/image_raw` 또는 `/joy`를 재사용할 때만 각각
`use_camera:=false`, `use_gamepad:=false`를 지정한다. motor bridge는 이 launch가
시작하지 않는다. 종료는 Y로 DRIVE OFF한 뒤 `Ctrl+C` 순서다.

CUDA container 없이 host CPU inference를 점검할 때의 일반 launch는 다음과 같다.
이 명령도 camera와 motor publisher를 열 수 있어 동일한 실행 직전 승인이 필요하다.

```bash
ros2 launch xycar_ai_drive guided_policy_collection.launch.py \
  params_file:="${PROFILE}" \
  artifact_id:=<versioned-policy-artifact-id> \
  curriculum_generation:="${GENERATION}" speed_cap:=30.0 allow_motion:=false \
  inference_backend:=local inference_device:=cpu
```

camera와 Joy publisher가 이미 있고 collector만 직접 시작해야 할 때는 설치된
parameter file과 versioned artifact를 모두 명시한다.

```bash
ros2 run xycar_ai_drive guided_policy_collector --ros-args \
  --params-file "${PROFILE}" \
  -p collection_profile_path:="${PROFILE}" \
  -p artifact_dir:=/home/xytron/xycar_ws_mgw/artifacts/models/<schema-v1-artifact-id> \
  -p curriculum_generation:="${GENERATION}" -p speed_cap:=30.0 -p allow_motion:=false
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
  --promotion-report artifacts/runs/front_cam_policy/<stateless-run>/promotion_gate.json \
  --artifact-id <schema-v1-stateless-artifact-id> \
  --require-schema-version 1

cd /home/xytron/xycar_ws/apps/xycar_ws_mgw
./scripts/ai/deploy_model.sh <normalized-steering-artifact-id> --dry-run
./scripts/ai/deploy_model.sh <normalized-steering-artifact-id>
```

`deploy_model.sh`는 versioned artifact만 배포하며 ROS node를 실행하지 않는다.
G1 이상에서 `--promotion-report`는 선택 사항이며 `passed`, `failed` 또는 report를
생략한 `not_evaluated` 상태가 manifest에 기록된다. offline 결과는 export와 배포를
차단하지 않으므로 실차 시험 전에 해당 상태와 실패 항목을 직접 확인한다.

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
확보하고, `Ctrl+C`와 게임패드 A release로 정지할 수 있는지 먼저 확인한다.
동시에 `gamepad_teleop`, 다른 drive node 또는 motor publisher를 실행하지 않는다.

camera와 game controller까지 함께 시작하는 기본 실행:

```bash
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws_mgw/install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar
ros2 launch xycar_ai_drive front_cam_policy.launch.py \
  artifact_id:=<normalized-steering-artifact-id>
```

이미 `/image_raw` camera 또는 `/joy` publisher가 있으면 해당 장치를 중복으로
열지 않는다.

```bash
ros2 launch xycar_ai_drive front_cam_policy.launch.py \
  artifact_id:=<normalized-steering-artifact-id> \
  use_camera:=false use_gamepad:=false
```

두 publisher가 이미 있을 때 policy node만 직접 실행할 수도 있다.

```bash
ros2 run xycar_ai_drive front_cam_policy --ros-args \
  --params-file /home/xytron/xycar_ws_mgw/install/xycar_ai_drive/share/xycar_ai_drive/config/front_cam_policy.yaml \
  -p artifact_dir:=/home/xytron/xycar_ws_mgw/artifacts/models/<normalized-steering-artifact-id>
```

artifact ID는 항상 명시한다. 새 Base가 준비되기 전까지 기존 artifact는 offline
분석 전용이고 실제 motion에는 사용하지 않는다. 종료는 `Ctrl+C`이며 node는 종료
경로에서 정지 command를 반복 발행한다.

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
  artifact_id:=front-cam-policy-vit-small-ar4-v2-nice-adaptive-joint-regression-sequence-init25-window5-20260821 \
  speed_cap:=25.0 \
  use_camera:=true use_gamepad:=true allow_motion:=true
```

정사각형 road-warp로 재학습한 좌회전 전용 ResNet18을 고정 속도 `23`으로
연속 inference하며 시험 주행하는 명령은 다음과 같다.

```bash
ros2 launch xycar_ai_drive jetson_gpu_policy.launch.py artifact_id:=nice-shortcut-resnet18-squarewarp-speed23-45sessions-20260821 speed_cap:=23.0 use_camera:=true use_gamepad:=true allow_motion:=true
```

이 명령은 8초 timer나 좌회전 감지 FSM을 포함하지 않는다. 시작은 DRIVE OFF이며
A를 누르는 동안만 model output과 speed `23`을 발행하고 A를 놓으면 정지한다.
camera와 motor publisher를 시작할 수 있으므로 실행 직전 실차 승인, 바퀴 지지 또는
안전 공간, 전원 차단 수단, `Ctrl+C` 정지와 경쟁 publisher 부재를 확인한다.

이 Jetson 전용 launch는 checksum이 있는 versioned artifact와 CUDA server
container를 먼저 준비하고 Unix socket이 열린 뒤 camera, gamepad와 policy node를
시작한다. motor bridge는 시작하지 않으므로 별도 terminal의 `motor` 또는
`x27.desktop`으로 사용자가 직접 관리한다. 종료는 `Ctrl+C`이며 launch가
container도 함께 정리한다. 이미 camera나 gamepad publisher가 있으면
`use_camera:=false` 또는 `use_gamepad:=false`를 지정한다. 기존 `xycar-ai-gpu`
wrapper와 `ARTIFACT_ID=<id>` 방식도 호환을 위해 계속 지원한다.

nice_adaptive 기본 정책은 2026-08-21 실차 A/B에서 채택한 schema v6 joint 회귀
artifact
`front-cam-policy-vit-small-ar4-v2-nice-adaptive-joint-regression-sequence-init25-window5-20260821`
이다. angle과 speed를 모두 float scalar로 출력하고 `speed_cap:=25.0`을 명시한다.
cap은 prediction debug topic이 아니라 실제 motor publish와 external history에
적용된다. schema v5 분류 artifact는 삭제하지 않지만 명시적 rollback에만 사용하고,
정상 운용에서 회귀형과 자동 교대하거나 CPU로 fallback하지 않는다.

speed-35 미세학습 candidate는 schema v6 artifact
`front-cam-policy-vit-small-ar4-v2-nice-ada-very-fast-joint-regression-sequence-init35-window5-speed35-20260823`
이다. 이 artifact는 speed `[0,35]`와 initial history `(0,35)×4`를 선언하므로
`speed_cap:=35.0`까지 허용한다. offline held-out 결과와 배포 checksum은 실차
승격을 의미하지 않으며, 실행은 매번 별도 실차 승인을 받은 제한 검증으로 시작한다.

wrapper는 camera를 열기 전에 host NumPy 1.x, OpenCV와 `cv_bridge` import를
검사한다. user site-packages의 NumPy 2.x가 ROS Humble ABI를 가리면 즉시 종료하며
camera나 CUDA container를 남기지 않는다. policy node가 오류로 종료돼도 같은
launch의 camera와 gamepad를 종료하고 CUDA container를 정리한다.

## 신호등 기반 Base/ResNet18 통합 runtime

`traffic_shortcut_policy` 하나가 `/image_raw`와 `/joy`를 구독하고 최종
`/xycar_motor`를 독점 발행한다. 현재 bundle schema v13은 schema v6 speed-35
nice_ada_very_fast Base와 schema v7 `nice-shortcut` ResNet18, 사람 보정 GT로
재학습한 YOLO11s
detector와 3-class CNN 전체를 포함한다. detector ONNX SHA-256은
`7d1bd24a025c6b7851c396e9e0cbc38dfad6f6e852c712e2cebdf8547a428e64`, CNN은
`e126f4f3036bcd4e44ab0ca4b5cc70a2e46f87bc859ae434719eb5f08de122b5`다.
`/home/xytron/yolo_tl.py`나 `xycar_ws_minju`를 import하지 않는다.

- detector는 BGR frame을 종횡비를 유지한 640×640 centered letterbox로 만들고,
  padding을 `(114,114,114)`로 채운 뒤 RGB float32 NCHW로 변환한다. confidence
  `0.25` 이상 중 최고 confidence bbox 하나만 선택하며 폭 `40..225`를 gate하고
  YOLO는 3 frame마다 탐색·갱신한다.
- 선택한 bbox의 각 축에 `15%` padding을 더한 crop을 Pillow bilinear로
  416×128 RGB에 resize하고 ImageNet normalization한 뒤 CNN에 넣는다. raw class는
  `STOP`, `STRAIGHT`, `LEFT`이며 softmax 최고 확률이 `0.50` 미만이면 `UNKNOWN`이다.
- YOLO 검출 뒤 중간 frame에서는 직전 bbox를 현재 frame crop에 재사용해 CNN만
  매 frame 실행한다. 동일 `STOP/LEFT/STRAIGHT`는 처리 완료된 분류 15회
  연속이어야 확정된다. camera 30 Hz와 실제 주행 vote rate는 같지 않다. `STOP`은 latch하고,
  확정된 같은 `LEFT` 또는
  `STRAIGHT` 15회만 STOP latch를 해제한다. `UNKNOWN`은 후보
  vote를 초기화하지만 이미 확정된 STOP latch는 해제하지 않는다. 다음 scheduled
  YOLO가 box를 놓치면 bbox cache를 즉시 지우고 재검출 전까지 CNN 판정을 멈춘다.
- left frame에서는 한 제어 주기 STOP, 다음 fresh frame부터 speed `23` ResNet18을
  사용한다. 동시에 Base는 진입 전 history에서 시작한 self-AR shadow prediction을
  매 fresh frame 계속 갱신하지만 motor에는 발행하지 않는다. 첫 실제 shortcut
  명령부터 8초에 최신 0.25초 이내 shadow command를 즉시 발행해 STOP 없이 Base로
  복귀한다. 성공은 process당 한 번이고 red 취소는 shadow와 시도를 폐기하되 성공을
  소비하지 않는다.
- 실제 실행 history와 미발행 Base shadow history는 분리한다. schema v13 Base와
  shadow에는 cap `35`가
  적용된 Base prediction만 compact token으로 추가하고, handoff 성공 때만 활성
  history로 승격한다. 최초 history는 `(0,35)×4`, token `[50,85]×4`다. 두 policy는
  한 CUDA process에서 preload/warm-up하고 서로 다른
  mode `0600` socket을 공유 CUDA lock으로 직렬화한다. shortcut 중에는 Base socket의
  paired RPC가 RGB frame을 한 번만 받고 공통 warp·resize를 한 번만 수행한 뒤, 각
  artifact의 서로 다른 normalization을 적용해 shortcut과 Base를 차례로 추론한다.
  따라서 Base self-AR를 계속 갱신하면서도 실제 선택 policy decision을 함께 반환한다.
- `use_gamepad:=true`에서는 OFF로 시작하고 A hold/release grace `0.12s` 계약을
  유지한다. 빨간불 정지 중에도 A를 계속 누르고 있어야 15회 확정된 다음 신호에서
  자동 재출발한다. `use_gamepad:=false`에서는 A-hold/Joy stale 검사를 쓰지 않고,
  camera와 motor subscriber가 준비되면 자동으로 ON이 되어 `Ctrl+C` 또는 fault까지
  계속 주행한다. camera/IPC stale, ONNX shape·NaN/Inf, 경쟁 publisher와 motor
  subscriber 소실은 어느 mode에서나 FAULT와 `[0,0]`이다.
- schema v5/v6/v7/v8/v9/v10/v11/v12/v13의 STOP은 YOLO가 confidence `0.25` 이상 box를 10번 연속
  찾지 못하면(매 3 frame 판독, 총 30 camera frame·약 1초) 예외적으로 latch를
  해제하고 Base로 재출발한다. 폭 gate 실패나 classifier UNKNOWN은 YOLO box가
  검출된 것으로 보고 이 counter를 즉시 초기화하며, ONNX 오류는 release가 아니라
  FAULT/[0,0]이다.

Jetson host는 NumPy `1.26.4`, ONNX Runtime `1.24.0`, provider 순서
CUDA→CPU를 exact 검사한다. bundle checksum, 두 socket과 synthetic ONNX preflight가
끝난 뒤에만 camera/gamepad launch를 시작한다. motor bridge는 포함하지 않는다.
현재 YOLO/CNN pair는 동일 human-GT validation에서 기존 pair 대비 end-to-end
action accuracy가 `439/477 (92.03%)`에서 `476/477 (99.79%)`로 상승했지만,
dataset manifest가 `known_scene_leakage: true`, `development_only: true`이므로 신규
주행 session의 일반화 성능으로 해석하지 않는다. 실차 운행 검증 전 후보 상태다.
기존 schema v12 speed-35 stop10-go30, schema v11 speed-35 stop30-go30, schema v10 speed-25 bundle, schema v9
stop10-go15, schema v8
stop3-go15 search3/classify1 bundle,
schema v6 votes2 bundle,
schema v3 HSV bundle,
`traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-tl45-votes5-every3-20260821`,
schema v1 종료 STOP bundle과
`traffic-shortcut-nice-regression-resnet18-8s-shadow-ar-handoff-20260821`
schema v2 red-3 bundle은 rollback용으로 삭제하거나 덮어쓰지 않는다.

```bash
cd /home/xytron/xycar_ws_mgw && source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch xycar_ai_drive jetson_traffic_shortcut.launch.py bundle_id:=traffic-shortcut-nice-ada-very-fast-speed35-regression-resnet18-8s-shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-tl40to225-stop15-go15-search3-classify1-yolo-miss30-release-45sessions-20260823 use_camera:=true use_gamepad:=true allow_motion:=true
```

대회 연속 주행은 Gamepad를 시작하지 않고 다음처럼 실행한다.

```bash
cd /home/xytron/xycar_ws_mgw && source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch xycar_ai_drive jetson_traffic_shortcut.launch.py bundle_id:=traffic-shortcut-nice-ada-very-fast-speed35-regression-resnet18-8s-shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-tl40to225-stop15-go15-search3-classify1-yolo-miss30-release-45sessions-20260823 use_camera:=true use_gamepad:=false allow_motion:=true
```

이 launch는 실제 camera와 motor publisher를 열 수 있으므로 매번 직전 승인을 받고
바퀴 지지/안전 공간, 전원 차단, A release와 `Ctrl+C`, 경쟁 publisher 부재를
확인한다. 성공한 좌회전을 다시 실행하려면 node를 재시작한다. 오프라인 성공은
실차 주행 적합성을 보장하지 않는다.

### 수동 신호등 예측 GUI

`traffic_light_viewer`는 schema v4..v13 bundle에서 신호등 YOLO와 CNN classifier만
load한다. Base·shortcut 정책 container와 socket을 시작하지 않고 Joy를 구독하거나
ROS topic을 publish하지 않는다. 따라서 `/xycar_motor` endpoint와 motion gate가
없다. GUI에는 3 frame마다 갱신한 YOLO bbox, 중간 frame의 cached-bbox CNN mode,
padding crop,
bundle에 선언된 raw class 확률, bbox 폭 gate, action별 vote, stop latch와 최종
`UNKNOWN/RED/LEFT/STRAIGHT`를 함께 표시한다. `Reset vote / stop latch`는 현재
진단 session을 초기화하며 frame이나 결과를 저장하지 않는다.

Jetson local GNOME desktop 또는 RDP 공유 화면의 terminal에서 다음처럼 실행한다.
기본 launch는 `/dev/videoCAM`을 여므로 매 실행 직전 camera 사용 승인을 받고 다른
camera node가 없는지 확인한다. 이 진단에는 motor bridge를 시작하지 않는다.

```bash
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch xycar_ai_drive traffic_light_viewer.launch.py \
  bundle_id:=traffic-shortcut-nice-ada-very-fast-speed35-regression-resnet18-8s-shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-tl40to225-stop15-go15-search3-classify1-yolo-miss30-release-45sessions-20260823 \
  use_camera:=true
```

이미 승인받아 실행 중인 `/image_raw` publisher가 있을 때는 camera device를 다시
열지 않도록 `use_camera:=false`를 쓴다. 이 mode도 같은 host NumPy `1.26.4`,
ONNX Runtime `1.24.0`, CUDA→CPU provider 순서와 bundle checksum을 검사한다.

```bash
ros2 launch xycar_ai_drive traffic_light_viewer.launch.py \
  bundle_id:=traffic-shortcut-nice-ada-very-fast-speed35-regression-resnet18-8s-shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-tl40to225-stop15-go15-search3-classify1-yolo-miss30-release-45sessions-20260823 \
  use_camera:=false
```

launch 없이 기존 `/image_raw`만 구독할 때는 bundle directory를 직접 지정한다. 이
명령은 camera driver도 시작하지 않으며 topic publisher가 없으면 대기 화면을
유지한다.

```bash
ros2 run xycar_ai_drive traffic_light_viewer --ros-args \
  -p bundle_dir:=/home/xytron/xycar_ws_mgw/artifacts/models/traffic-shortcut-nice-ada-very-fast-speed35-regression-resnet18-8s-shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-tl40to225-stop15-go15-search3-classify1-yolo-miss30-release-45sessions-20260823 \
  -p camera_topic:=/image_raw
```

wrapper 내부 경계를 각각 진단할 때만 다음 명령을 사용한다. 첫 명령은 container
안에서 두 CUDA policy server를 띄우는 entry point다. host launch와 node 직접
실행은 같은 bundle로 두 socket이 이미 준비된 경우에만 유효하다. 뒤 두 명령은
camera 또는 motor publisher를 열 수 있으므로 정상 wrapper와 같은 실행 직전 승인
규칙을 적용한다.

```bash
ros2 run xycar_ai_drive traffic_shortcut_gpu_server -- \
  --bundle-dir /artifacts/traffic-shortcut-nice-ada-very-fast-speed35-regression-resnet18-8s-shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-tl40to225-stop15-go15-search3-classify1-yolo-miss30-release-45sessions-20260823 \
  --base-socket-path /run/user/1000/xycar-ai/traffic-base.sock \
  --shortcut-socket-path /run/user/1000/xycar-ai/traffic-shortcut.sock \
  --device cuda

ros2 launch xycar_ai_drive traffic_shortcut_policy.launch.py \
  bundle_id:=traffic-shortcut-nice-ada-very-fast-speed35-regression-resnet18-8s-shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-tl40to225-stop15-go15-search3-classify1-yolo-miss30-release-45sessions-20260823 \
  use_camera:=true use_gamepad:=true allow_motion:=true

ros2 run xycar_ai_drive traffic_shortcut_policy --ros-args \
  --params-file /home/xytron/xycar_ws_mgw/install/xycar_ai_drive/share/xycar_ai_drive/config/traffic_shortcut_policy.yaml \
  -p bundle_dir:=/home/xytron/xycar_ws_mgw/artifacts/models/traffic-shortcut-nice-ada-very-fast-speed35-regression-resnet18-8s-shadow-ar-handoff-yolo11s-humanbbox-cnn416-actions3-conf50-tl40to225-stop15-go15-search3-classify1-yolo-miss30-release-45sessions-20260823
```

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
