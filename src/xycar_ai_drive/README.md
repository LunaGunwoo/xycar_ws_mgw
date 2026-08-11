# xycar_ai_drive

고정 224x224 TorchScript ViT-tiny 정책으로 `/image_raw`를 계속 추론하고,
게임패드 A 버튼으로 실제 `/xycar_motor` 발행을 ON/OFF하는 ROS 2 패키지다.
모터 메시지는 `std_msgs/Float32MultiArray([angle, speed])`이며 ON/OFF와 관계없이
20 Hz로 발행한다.

## 동작 계약

- 시작 상태는 항상 OFF다. 시작 후 A를 한 번 놓은 다음 누르면 ON, 다시 누르면
  OFF가 되는 상승 edge toggle이다. A를 계속 누르고 있는 것은 추가 toggle이 아니다.
- 추론은 OFF 상태에서도 최신 camera frame으로 계속 수행한다. ON 상태에서만 최신
  유효 예측을 motor command로 사용하고, OFF 상태에서는 `[0, 0]`을 발행한다.
- angle은 `angle_class_id - 100`이다. speed는
  `max(0, speed_class_id - 100)`이므로 reverse는 금지하지만 양수 예측에는 별도
  cap을 적용하지 않아 label 계약의 최대 `100`까지 전달한다.
- Joy 또는 camera prediction이 0.25초 이상 stale, 추론/변환 오류, motor
  subscriber 소실, 다른 motor publisher 출현 시 즉시 OFF와 `[0, 0]`으로
  전환한다. 조건이 복구돼도 A release 후 새 press가 있어야 다시 ON된다.
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

node는 시작 전에 checksum, 고정 RGB `[1,3,224,224]` input, angle/speed
`[1,201]` tuple output, normalization과 label decode 계약을 검증한다. 이후 CPU
thread 8개로 model을 load하고 3회 warm-up한다. artifact 생성과 배포는 개발
Laptop의 MGW root에서 수행한다.

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
