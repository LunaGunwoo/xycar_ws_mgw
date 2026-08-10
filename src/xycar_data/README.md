# xycar_data

`xycar_data`는 게임패드·터미널로 Xycar를 조종하고 카메라 기반
behavior-cloning 모델 학습용 데이터를 수집하는 패키지다. 두 Teleop 모두
카메라 frame을 학습 입력으로 사용하고 각 PNG에는 그 순간 실제 발행한 연속형
`angle`, `speed`를 label로 저장한다. 게임패드 주행은 카메라가 없어도 유지되며
녹화 중 누락된 camera frame만 건너뛴다.

모든 명령은 실제 차량 `xytron@10.42.0.1:/home/xytron/xycar_ws_mgw` 기준이다.
카메라·LiDAR driver 또는 motor publisher를 시작하므로 각각 실행 직전에 별도
승인을 받고, 차량 지지 상태·비상 정지 공간·기존 motor publisher 종료 상태를
확인해야 한다.

## SSH, 빌드와 환경 적용

```bash
ssh -t xytron@10.42.0.1
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
git pull --ff-only
colcon build --symlink-install \
  --packages-select xycar_cam xycar_lidar xycar_data
source /home/xytron/xycar_ws_mgw/install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar
```

`teleop_recorder`는 GUI나 창을 만들지 않는 CLI 전용 node다. 키보드 TTY가
필요하므로 `ros2 launch`로 실행하지 않고 SSH 연결에도 `-t`를 유지한다.

## Remote Gamepad Teleop

Remote Gamepad 휴대폰 앱과 PC 앱을 먼저 연결한다. 차량 바퀴를 지면에서 띄우고
다른 `/xycar_motor` publisher가 없는지 확인한 뒤 다음 launch를 실행한다.

```bash
ros2 launch xycar_data gamepad_teleop.launch.py
```

이 한 명령은 camera driver, ROS 2 공식 `joy/game_controller_node`,
`gamepad_teleop`을 함께 시작한다. 따라서 게임패드 입력, 차량 주행, camera 기반
데이터 수집을 별도 terminal 없이 모두 수행한다. 입력과 출력은 다음과 같다.

| Gamepad 입력 | 변환 | 범위 |
| --- | --- | --- |
| 왼쪽 스틱 좌우 `axes[0]` | `angle = -100 * axes[0]` | `-100 ~ 100` |
| LT `axes[4]` | `speed -= 5 * depth` | `0 ~ -5` |
| RT `axes[5]` | `speed += 7 * depth` | `0 ~ 7` |
| A `buttons[0]` | 양수 속도 녹화 대기·시작 | — |
| B `buttons[1]` | 현재 세션 정상 저장 | — |

차량 Remote Gamepad 실측 기준은 trigger release `+1`, full press `-1`이며
LT는 `axes[4]`, RT는 `axes[5]`다. 따라서 `trigger_axis_mode: signed`가
기본값이고 각 depth는 `(1 - raw_axis) / 2`다. raw `+1`, `0`, `-1`은 각각
depth `0`, `0.5`, `1`로 변환된다. LT와 RT는 합산하므로 둘을 끝까지 누르면
speed는 `2`다. 두 트리거가 모두 `+1`이면 speed만 0이 되고 angle은 왼쪽
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

`/joy`가 0.25초 이상 끊기거나, 축 배열이 잘못됐거나, motor subscriber가 없거나,
다른 motor publisher가 발견되면 `[0, 0]`을 발행한다. 종료할 때도 정지 명령을
5회 발행한다. 시작할 때와 위 안전 정지에서 복구할 때는 LT와 RT가 모두
`neutral_trigger_threshold` 이하인 입력을 한 번 확인해야 다시 주행할 수 있다.
연결이 유효하고 `/joy`가 갱신되는 동안 마지막 유효 명령은 20 Hz로 반복된다.

휴대폰 연결이 끊겨도 PC의 가상 controller가 마지막 `/joy`를 계속 갱신하면 이
node는 실제 Remote 앱 단절과 정상적인 고정 입력을 구분할 수 없다. 이 동작은
실차 주행 전에 별도로 확인해야 하며, 마지막 값이 계속 갱신되는 환경은 앱
heartbeat 또는 별도 deadman 없이는 단절 안전이 검증된 것으로 보지 않는다.

이미 `/image_raw` camera가 실행 중이면 중복 장치 접근을 피하도록 camera 자동
시작을 끌 수 있다.

```bash
ros2 launch xycar_data gamepad_teleop.launch.py use_camera:=false
```

`/joy`도 이미 다른 승인된 `Joy` node에서 발행 중일 때는 teleop만 단독 실행할
수 있다. 이 명령도 `/xycar_motor` publisher를 만들므로 실행 직전 승인이
필요하다.

```bash
ros2 run xycar_data gamepad_teleop
```

기본값은 `config/gamepad_teleop.yaml`에서 바꿀 수 있다. 다른 SDL 장치를 쓸 때는
device ID를 launch 인자로 지정한다.

```bash
ros2 launch xycar_data gamepad_teleop.launch.py device_id:=1
```

trigger가 `0`에서 `+1`로 움직이는 controller는 별도 YAML에서 `positive`,
`0`에서 `-1`로 움직이는 controller는 `negative`로 설정하고 `params_file`
launch 인자로 전달한다. 차량에서는 LT/RT 축 번호와 `+1`에서 `-1`로 변하는
raw 입력까지만 확인했다.
조향 `±100`, 전진 `7`, 후진 `-5`의 실제 회전과 motor scale, release·stale·종료
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
중복 실행하지 않는다. `/scan`은 없어도 recorder가 동작하며 해당 sample은
`lidar_valid=false`로 저장된다. Terminal `teleop_recorder`는 필요한 headless
camera를 자체 준비하므로 이 sensor launch를 먼저 실행할 필요가 없다.

기본 Gamepad launch는 camera를 직접 포함하므로 별도의 sensor launch가 필요하지
않다. `use_camera:=false`로 실행했다면 `/image_raw`를 기존 camera node에서 받아야
한다. camera가 없거나 중간에 끊기면 gamepad 주행과 녹화 상태는 유지되고 해당
구간의 frame만 저장되지 않는다. camera가 복구되면 같은 세션에서 새 frame 저장을
계속한다.

## Terminal Teleop과 수집

SSH TTY에서 environment를 적용한 뒤 실행한다. 이 한 명령은 필요하면 camera
driver를 시작하고 `/xycar_motor`에 `Float32MultiArray([angle, speed])`를
publish할 수 있다. 따라서 camera 장치 접근과 motor 명령을 합친 실차 실행으로
간주하며 매 실행 직전에 두 동작 모두에 대한 승인이 필요하다.

```bash
ros2 run xycar_data teleop_recorder
```

recorder는 키 입력을 받기 전에 ROS graph를 1초간 확인한다.

- `/image_raw` publisher와 해석 가능한 최신 frame이 있으면 기존 headless
  camera를 재사용한다.
- publisher가 없으면 GUI 없는
  `ros2 launch xycar_cam xycar_cam.launch.py`를 자동 시작하고 최대 10초간
  해석 가능한 frame을 기다린다.
- publisher는 있지만 frame이 없거나 decode할 수 없으면 camera를 중복 시작하지
  않고 원인을 출력한 뒤 종료한다.
- `/examine_image`가 발견되면 GUI viewer가 실행 중인 것으로 보고 시작을
  거부한다. `xycar_cam_viewer.launch.py`를 종료한 뒤 recorder를 다시 실행한다.

recorder가 직접 시작한 camera만 종료 시 SIGINT, SIGTERM 순서로 정리한다.
사용자가 따로 시작한 headless camera는 재사용만 하고 종료하지 않는다. 현재
차량에서 확인한 `/image_raw` 계약은 `sensor_msgs/Image`, `rgb8`, 약 30 Hz이며
recorder는 이를 OpenCV BGR frame으로 변환한다.

다른 YAML을 시험할 때는 ROS parameter로 경로를 바꾼다.

```bash
ros2 run xycar_data teleop_recorder --ros-args \
  -p tuning_file:=/home/xytron/xycar_ws_mgw/src/xycar_data/config/teleop_recorder.yaml
```

기본 키와 명령은 다음과 같다.

| 키 | `[angle, speed]` | 동작 |
| --- | --- | --- |
| W | speed `8` | 현재 angle을 유지하고 전진 속도 즉시 적용 |
| Shift+W | speed `12` | 전진 속도 `8`의 1.5배를 즉시 적용 |
| S, Shift+S | speed `-8` | 현재 angle을 유지하고 후진 속도 즉시 적용 |
| A | angle `-10`, 최소 `-100` | 현재 speed를 유지하고 좌측 조향 누적 |
| D | angle `+10`, 최대 `100` | 현재 speed를 유지하고 우측 조향 누적 |
| E | — | 새 기록 세션 시작 |
| Q | 현재 명령 유지 | 현재 세션 저장·마감 |
| Space | `[0, 0]` | 세션을 유지한 즉시 정지 |
| Esc, Ctrl+C | `[0, 0]` | 정지 명령 5회 후 종료 |

W/S에는 증분이 없으며 한 번의 입력으로 각각 `8`과 `-8`을 설정한다. CLI/SSH
TTY는 Shift 단독 event를 전달하지 않으므로 대문자 `W`를 Shift+W로 해석해
`12`를 설정한다. 대문자·소문자 S는 모두 `-8`이며 후진 boost는 없다.

A/D는 OS key-repeat으로 누적 조정된다. 기본 angle step `10`은 10회 입력으로
중립에서 `±100`에 도달하며, terminal repeat가 30 Hz이면 repeat 시작 뒤 약
0.3초가 걸린다. 실제 시간은 OS/SSH key-repeat 설정에 따라 달라진다. 모든
WASD 입력이 0.25초 끊기면 angle과 speed 상태를 모두 초기화하고 즉시 `[0, 0]`을
발행한다. W로 정한 speed는 A/D가 계속 입력되는 동안 유지된다. 방향키는 주행
입력으로 사용하지 않는다.

기록 세션이 없어도 수동 주행은 가능하지만, `E` 이후 speed가 0이 아닌 유효
WASD 명령과 최신 camera frame이 동시에 있을 때만 파일을 저장한다. speed가
0인 조향-only frame과 입력 timeout 상태는 저장하지 않는다. `Q`는 세션만
저장·마감하며 현재 차량 명령을 별도로 정지시키지 않는다.

시작 전과 주행 중에는 다음 안전 gate가 적용된다.

- 카메라가 0.25초 이상 stale이면 즉시 정지하고 새 WASD 입력이 필요하다.
- recorder가 자동 시작한 camera process가 종료되면 즉시 `[0, 0]`으로 정지하고
  활성 세션을 incomplete로 마감한 뒤 오류 종료한다.
- 다른 `/xycar_motor` publisher가 있으면 주행을 거부한다. 주행 중 발견되면
  정지 후 오류 종료한다.
- motor subscriber가 없으면 command와 sample 저장을 거부한다.
- LiDAR는 선택적이다. 최신 scan이 없으면 경고만 남기고 camera sample을 저장한다.
- writer queue 포화, 디스크 여유 부족, 이미지·NPZ 쓰기 오류는 정지와
  `YYYYMMDD_HHMMSS_mmm_incomplete` 세션 보존으로 처리한다.

## 데이터 형식

기본 저장 위치는 저장소 밖의 `/home/xytron/xycar_data/teleop`이며
터미널은 `config/teleop_recorder.yaml`, gamepad는
`config/gamepad_teleop.yaml`에서 바꿀 수 있다. 첫 유효 sample 전에는
directory를 만들지 않으므로 빈 세션은 남지 않는다.

```text
YYYYMMDD_HHMMSS_mmm_session/
  Images/1.png
  Lidar/000001.npz
  samples.csv
  metadata.yaml
```

`samples.csv`의 각 행은 하나의 camera frame이다. `image`, `angle`, `speed`,
`input_key`, camera stamp/수신 시각과 `lidar_valid`를 기본 학습 label로 사용한다.
유효한 LiDAR가 있으면 `lidar` 경로와 scan timestamp·skew도 기록하며, 하나의
LiDAR scan이 여러 camera frame에 대응할 때 NPZ 파일을 재사용한다. NPZ에는
전체 `ranges`, `intensities`, LaserScan geometry·timing·frame metadata가 담긴다.
Gamepad sample은 `input_key=gamepad`, `lidar_valid=false`로 기록되고
`metadata.yaml`의 `control_mode=gamepad`로 터미널 세션과 구분한다.

정상 `Q` 또는 종료는 임시 `_recording_...` directory를
`YYYYMMDD_HHMMSS_mmm_session`으로
atomic rename한다. 쓰기 실패나 경쟁 motor publisher로 중단되면
`YYYYMMDD_HHMMSS_mmm_incomplete`로 남고 `metadata.yaml`의 `stop_reason`에서
원인을 확인할 수 있다. Gamepad B 종료는 `stop_reason=b_button`, speed 0 이하 종료는
`stop_reason=speed_nonpositive`와 실제 `emergency_discard_count`를 남긴다.
폐기 후 유효 sample이 하나도 없으면 빈 session directory를 만들지 않는다.

## 튜닝

`config/teleop_recorder.yaml`과 `config/gamepad_teleop.yaml`은 다음을 분리한다.

- camera, LiDAR, motor topic
- WASD angle step, 즉시 speed와 boost, 20 Hz publish rate, key timeout,
  stop 반복 횟수
- gamepad axis·button mapping, trigger 부호 profile, 중립 재활성 임계값과
  출력 한계
- 필수 camera freshness와 선택 LiDAR 연결 허용 시간
- headless camera 자동 시작 여부와 graph discovery, frame startup, process
  shutdown timeout
- dataset root, emergency tail 15 frame, PNG compression, writer queue, 최소
  디스크 여유

YAML의 빈 topic, 잘못된 speed/steering 부호·범위, `NaN`·`Inf` 수치, 음수
timeout, queue 크기와 PNG compression 범위 오류는 node 시작 시 거부된다.
실제 차에서는 raised-car 상태에서 A/D 조향 부호와 후진 부호를 먼저
확인한 뒤 값을 조정한다.
