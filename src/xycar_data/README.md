# xycar_data

`xycar_data`는 게임패드로 Xycar를 조종하고 카메라 기반 behavior-cloning 모델
학습용 데이터를 수집하는 패키지다. 카메라 frame마다 그 순간 실제 발행한 연속형
`angle`, `speed`를 label로 저장하고, 시간상 가장 가까운 선행 LiDAR scan을 선택적
NPZ metadata로 연결한다. 게임패드 주행은 카메라나 LiDAR가 없어도 유지되며,
camera가 없으면 frame을 건너뛰고 LiDAR가 없으면 `lidar_valid=false`로 저장한다.

새 수집의 `/xycar_motor` angle은 물리 driver 단위가 아니라 normalized percent
`-100..100`이다. ROS 1 safety adapter가 이를 `×0.4`로 변환해
`/xycar_motor_safe`에 driver command `-40..40`을 발행한다. CSV `angle`은 변환 전
실제 normalized publish 값이며 session metadata의
`steering_contract.name=normalized_percent_v1`에 이 의미를 기록한다.

새 수집 명령은 활성 Jetson 차량
`xytron@xycar-gpu:/home/xytron/xycar_ws_mgw` 기준이다. 기존 `xytron@xycar`는
CPU inference rollback 재현에만 사용한다.
카메라·LiDAR driver 또는 motor publisher를 시작하므로 각각 실행 직전에 별도
승인을 받고, 차량 지지 상태·비상 정지 공간·기존 motor publisher 종료 상태를
확인해야 한다.

## SSH, 빌드와 환경 적용

```bash
ssh -t xytron@xycar-gpu
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
git pull --ff-only
colcon build --symlink-install \
  --packages-select xycar_cam xycar_lidar xycar_data
source /home/xytron/xycar_ws_mgw/install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar
export ROS_LOCALHOST_ONLY=1
```

## Remote Gamepad Teleop

Remote Gamepad 휴대폰 앱과 PC 앱을 먼저 연결한다. 차량 바퀴를 지면에서 띄우고
다른 `/xycar_motor` publisher가 없는지 확인한 뒤 다음 launch를 실행한다.

새 stateless generation 0 수집은 설치 시 한 번 생성되는 외부 profile을 사용한다.
`install_runtime.sh`는 이 파일이 이미 있으면 덮어쓰지 않는다. profile의 기본
전진 최고속도는 `25`, 후진 최고속도 절댓값은 `15`, 저장 root는
`/home/xytron/xycar_data/stateless_manual`이다. 각 session metadata에는 profile의
절대 경로와 SHA-256, 실제 적용된 gamepad·recording·timeout 값이 기록된다.
기존 외부 profile은 배포 전에 SHA-256 이름으로 백업한 뒤 25/15와
`lidar_topic`, `lidar_timeout_sec`, `max_lidar_skew_sec`를 별도로 반영해야 한다.
차량의 tracked seed 파일을 현장에서 수정해 대신하지 않는다.
Gamepad camera frame은 Jetson에서 30 Hz를 지속 저장할 수 있도록 기본 JPEG 품질
95로 저장한다. 이미지 형식과 품질은 외부 profile에서 조정할 수 있다.

```bash
ros2 launch xycar_data gamepad_teleop.launch.py \
  params_file:=/home/xytron/.config/xycar/gamepad_stateless_manual_normalized_v1.yaml
```

아래 인자 없는 명령도 normalized v1 generic profile과 기존
`/home/xytron/xycar_data/teleop` 저장 root를 사용한다. Generic profile은 기존
전진 `7`/후진 `-5`를 유지하며, 아래 입력 표와 25/15 값은 위 stateless manual
profile 기준이다.

```bash
ros2 launch xycar_data gamepad_teleop.launch.py
```

이 한 명령은 camera와 LiDAR driver, ROS 2 공식 `joy/game_controller_node`,
`gamepad_teleop`을 함께 시작한다. 따라서 게임패드 입력, 차량 주행, camera 기반
데이터와 선택적 LiDAR scan 수집을 별도 terminal 없이 모두 수행한다. controller와
teleop은 전용 `/gamepad_teleop/joy`만 사용하므로, 별도로 남아 있는 `/joy`
publisher의 축 배열과 명령이 섞이지 않는다. launch도 각 child process에
`ROS_LOCALHOST_ONLY=1`을 강제해 같은 Wi-Fi의 다른 ROS 2 participant를 발견하지
않는다. 입력과 출력은 다음과 같다.

| Gamepad 입력 | 변환 | 범위 |
| --- | --- | --- |
| 왼쪽 스틱 좌우 `axes[0]` | `angle = -100 * axes[0]` | `-100 ~ 100` |
| LT `axes[4]` | `speed -= 15 * depth` | `0 ~ -15` |
| RT `axes[5]` | `speed += 25 * depth` | `0 ~ 25` |
| A `buttons[0]` | 양수 속도 녹화 대기·시작 | — |
| B `buttons[1]` | 현재 세션 정상 저장 | — |

차량 Remote Gamepad 실측 기준은 trigger release `0`, full press `-1`이며
LT는 `axes[4]`, RT는 `axes[5]`다. 따라서 `trigger_axis_mode: negative`가
기본값이고 각 depth는 `-raw_axis`다. raw `0`, `-0.5`, `-1`은 각각
depth `0`, `0.5`, `1`로 변환된다. LT와 RT는 합산하므로 둘을 끝까지 누르면
speed는 `10`이다. 두 트리거가 모두 `0`이면 speed만 0이 되고 angle은 왼쪽
스틱을 계속 따라간다.

A를 누르고 있는 동안 녹화를 대기한다. 실제 `/xycar_motor`에 `speed > 0`이
발행되는 순간 세션이 시작되며, 시작 전에 A를 놓으면 대기가 취소된다. 녹화
시작 뒤에는 A를 놓아도 계속 기록한다. B는 보류 중인 frame까지 정상 저장하고
녹화만 끝내므로 차량은 현재 gamepad 명령으로 계속 주행한다.

녹화 중 실제 발행 speed가 `0` 또는 음수가 되면 최신 camera frame 15개를
폐기하고 그 이전 frame을 학습 가능한 정상 세션으로 저장한다. 이 동작은 녹화만
종료하며 별도 정지나 disarm을 하지 않으므로 LT에 의한 후진은 계속 발행될 수
있다. B 또는 speed 종료 후 A를 계속 누르고 있어도 자동 재시작하지 않으며,
A를 한 번 놓았다가 다시 눌러야 새 세션을 시작한다.

전용 `/gamepad_teleop/joy`가 0.25초 이상 끊기거나, 축 배열이 잘못됐거나, motor subscriber가 없거나,
다른 motor publisher가 발견되면 `[0, 0]`을 발행한다. Fast DDS가 로컬 ROS 1
motor bridge 이름을 `UNKNOWN`으로 표시할 때는 동일 DDS participant GID의 이름
없는 publisher/subscriber 쌍만 필수 relay로 인정한다. 이름 있는 다른 publisher,
짝 없는 이름 없는 publisher와 participant가 다른 쌍은 계속 정지 조건이다.
종료할 때도 정지 명령을 5회 발행한다. 시작할 때와 위 안전 정지에서 복구할 때는
LT와 RT가 모두
`neutral_trigger_threshold` 이하인 입력을 한 번 확인해야 다시 주행할 수 있다.
연결이 유효하고 전용 Joy topic이 갱신되는 동안 마지막 유효 명령은 20 Hz로 반복된다.

휴대폰 연결이 끊겨도 PC의 가상 controller가 마지막 Joy 값을 계속 갱신하면 이
node는 실제 Remote 앱 단절과 정상적인 고정 입력을 구분할 수 없다. 이 동작은
실차 주행 전에 별도로 확인해야 하며, 마지막 값이 계속 갱신되는 환경은 앱
heartbeat 또는 별도 deadman 없이는 단절 안전이 검증된 것으로 보지 않는다.

이미 `/image_raw` camera가 실행 중이면 중복 장치 접근을 피하도록 camera 자동
시작을 끌 수 있다. 이미 `/scan`이 실행 중이면 같은 방식으로 LiDAR 자동 시작을
끈다.

```bash
ros2 launch xycar_data gamepad_teleop.launch.py use_camera:=false
ros2 launch xycar_data gamepad_teleop.launch.py use_lidar:=false
```

`/joy`도 이미 다른 승인된 `Joy` node에서 발행 중일 때는 teleop만 단독 실행할
수 있다. 이 명령도 `/xycar_motor` publisher를 만들므로 실행 직전 승인이
필요하다.

```bash
ros2 run xycar_data gamepad_teleop --ros-args \
  --params-file /home/xytron/.config/xycar/gamepad_stateless_manual_normalized_v1.yaml \
  -p collection_profile_path:=/home/xytron/.config/xycar/gamepad_stateless_manual_normalized_v1.yaml
```

기본값은 `config/gamepad_teleop.yaml`에서 바꿀 수 있다. 다른 SDL 장치를 쓸 때는
device ID를 launch 인자로 지정한다.

```bash
ros2 launch xycar_data gamepad_teleop.launch.py device_id:=1
```

trigger가 `0`에서 `+1`로 움직이는 controller는 별도 YAML에서 `positive`,
`+1`에서 `-1`로 전체 축 범위를 쓰는 controller는 `signed`로 설정하고
`params_file` launch 인자로 전달한다. 차량에서는 LT/RT 축 번호와 `0`에서
`-1`로 변하는 raw 입력을 확인했다.
조향 `±100`, 전진 `25`, 후진 `-15`의 실제 회전과 motor scale, release·stale·종료
정지는 raised-car 상태에서 아직 검증되지 않았다. 첫 실차 시험은 낮은 trigger
깊이부터 시작한다.

## 센서 실행

Gamepad 녹화 또는 센서만 수동으로 실행할 때 첫 터미널에서 카메라와 선택적
LiDAR driver를 시작한다. 이 launch는 motor publisher를 포함하지 않는다.

```bash
ros2 launch xycar_data teleop_sensors.launch.py
```

LiDAR가 없더라도 카메라 주행 데이터를 수집하려면 아래처럼 카메라만 시작한다.

```bash
ros2 launch xycar_data teleop_sensors.launch.py use_lidar:=false
```

이미 `/image_raw`와 `/scan`이 다른 승인된 node에서 publish 중이면 이 launch를
중복 실행하지 않는다.

기본 Gamepad launch는 camera와 LiDAR를 직접 포함하므로 별도의 sensor launch가
필요하지 않다. `use_camera:=false` 또는 `use_lidar:=false`로 실행했다면 각각
`/image_raw` 또는 `/scan`을 기존 node에서 받아야 한다. camera가 없거나 중간에
끊기면 gamepad 주행과 녹화 상태는 유지되고 해당 구간의 frame만 저장되지 않는다.
LiDAR가 없거나 `0.30`초 timeout 또는 `0.20`초 최대 skew를 넘으면 camera frame은
계속 저장하되 `lidar_valid=false`로 남긴다.

## 데이터 형식

새 gamepad stateless 저장 위치는
`/home/xytron/xycar_data/stateless_manual`이다. 기존 호환 profile의 기본값
`/home/xytron/xycar_data/teleop`은 `config/gamepad_teleop.yaml`에서 바꿀 수 있다.
첫 유효 sample 전에는 directory를 만들지 않으므로 빈 세션은 남지 않는다.

```text
YYYYMMDD_HHMMSS_mmm_session/
  Images/1.jpg 또는 Images/1.png
  Lidar/000001.npz
  samples.csv
  metadata.yaml
```

`samples.csv`의 각 행은 하나의 camera frame이다. `image`, `angle`, `speed`,
`input_key`, camera stamp/수신 시각과 `lidar_valid`를 기본 학습 label로 사용한다.
실제 이미지 상대 경로와 확장자는 `image` 열이 기준이다. Gamepad 기본은 JPEG
품질 95이고 과거 PNG session도 같은 CSV 계약으로 읽는다.
유효한 LiDAR가 있으면 `lidar` 경로와 scan timestamp·skew도 기록하며, 하나의
LiDAR scan이 여러 camera frame에 대응할 때 NPZ 파일을 재사용한다. NPZ에는
전체 `ranges`, `intensities`, LaserScan geometry·timing·frame metadata가 담긴다.
Gamepad sample은 `input_key=gamepad`와 `metadata.yaml`의
`control_mode=gamepad`로 기록된다. 유효 scan 연결 여부는 각 행의
`lidar_valid`와 metadata의 `lidar_linked_count`, `lidar_missing_count`로 확인한다.

정상 B 종료 또는 speed 0 이하 종료는 임시 `_recording_...` directory를
`YYYYMMDD_HHMMSS_mmm_session`으로 atomic rename한다. 쓰기 실패나 경쟁 motor
publisher로 중단되면
`YYYYMMDD_HHMMSS_mmm_incomplete`로 남고 `metadata.yaml`의 `stop_reason`에서
원인을 확인할 수 있다. Gamepad B 종료는 `stop_reason=b_button`, speed 0 이하 종료는
`stop_reason=speed_nonpositive`와 실제 `emergency_discard_count`를 남긴다.
폐기 후 유효 sample이 하나도 없으면 빈 session directory를 만들지 않는다.

## 튜닝

새 수집은 외부
`~/.config/xycar/gamepad_stateless_manual_normalized_v1.yaml`을 조정한다.
tracked seed는 `config/gamepad_stateless_manual_normalized_v1.yaml`이다. 계약이
없는 기존 profile과 raw session은 legacy/reference로 보존하며 새 학습에 넣지
않는다.

- camera, LiDAR, Joy와 motor topic 및 LiDAR timeout/skew
- gamepad axis·button mapping, trigger 부호 profile, 중립 재활성 임계값과
  출력 한계
- Joy freshness, graph 확인 주기, 20 Hz publish rate와 stop 반복 횟수
- dataset root, emergency tail 15 frame, `jpeg|png` 이미지 형식, JPEG 품질,
  PNG compression, writer queue, 최소 디스크 여유

YAML의 빈 topic, 잘못된 speed/steering 부호·범위, `NaN`·`Inf` 수치, 음수
timeout, queue 크기, 이미지 형식, JPEG 품질과 PNG compression 범위 오류는 node
시작 시 거부된다. profile이 없거나 `steering_contract`가
`normalized_percent_v1`이 아니어도 시작을 거부한다.
실제 차에서는 raised-car 상태에서 왼쪽 stick 조향 부호와 후진 부호를 먼저
확인한 뒤 값을 조정한다.

## 대회 mission sequence 수집

`mission_sequence_collection.launch.py`는 신호등 정지·재출발과 지름길 전체를
한 session으로 보존하는 별도 수집 mode다. 기존 stateless 수집 동작은 그대로
유지한다. mission mode에서는 A rising edge에 즉시 시작하고 speed가 0 또는
음수가 되어도 자동 종료하지 않으며 B를 눌렀을 때만 정상 저장한다.

아래 launch는 camera, gamepad와 `/xycar_motor` publisher를 시작한다. 매 실행
직전 camera launch와 motor publisher 각각 사용자 승인을 받고 바퀴 지지/안전
공간, 전원 차단, `Ctrl+C`, 경쟁 publisher 부재를 확인한다. motor bridge는 별도
terminal에서 승인 후 실행한다.

```bash
ssh xytron@xycar-gpu
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws_mgw/install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar

ros2 launch xycar_data mission_sequence_collection.launch.py \
  params_file:=/home/xytron/.config/xycar/competition_mission_collection_normalized_v1.yaml \
  capture_kind:=signal
```

완전한 지름길 시범은 별도 실행에서 다음처럼 선택한다.

```bash
ros2 launch xycar_data mission_sequence_collection.launch.py \
  params_file:=/home/xytron/.config/xycar/competition_mission_collection_normalized_v1.yaml \
  capture_kind:=shortcut
```

camera와 Joy가 이미 승인된 별도 node에서 발행 중일 때 collector만 직접 실행할
수 있다. 이 명령도 motor publisher를 만들므로 실행 직전 별도 승인이 필요하다.

```bash
ros2 run xycar_data mission_sequence_collector --ros-args \
  --params-file /home/xytron/.config/xycar/competition_mission_collection_normalized_v1.yaml \
  -p capture_kind:=signal \
  -p collection_profile_path:=/home/xytron/.config/xycar/competition_mission_collection_normalized_v1.yaml
```

- A를 놓았다가 한 번 누르면 현재 speed와 관계없이 녹화가 시작된다.
- RT/LT와 왼쪽 stick으로 계속 사람이 운전한다.
- 빨간불에서 RT를 놓아도 `speed=0` frame이 같은 session에 남는다.
- B는 session을 저장하지만 vehicle을 자동 정지하지 않는다.
- B 저장 뒤 trigger 중립, launch `Ctrl+C`, motor `Ctrl+C` 순서로 종료한다.

기본 저장 root는 `/home/xytron/xycar_data/competition_manual`이다. 정상 metadata는
`dataset_kind=competition_mission_sequence`,
`control_mode=gamepad_mission_sequence`, `mission.capture_kind`와
`mission.records_stationary_frames=true`를 기록한다. 수집 profile seed는
`config/competition_mission_collection_normalized_v1.yaml`이고 Jetson runtime
설치 시 같은 이름의 외부 file이 없을 때만 복사한다.

정확히 모아야 할 상태별 session 수, 각 sequence 시작/종료 지점과 라벨 검수
항목은 상위 하네스의 `docs/competition_ai_data_checklist.md`를 따른다. active
`_recording_*`과 `_incomplete`는 학습 도구가 제외하며, 실패 자료를 정상
session처럼 rename하지 않는다.
