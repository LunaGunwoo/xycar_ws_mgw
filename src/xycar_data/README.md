# xycar_data

`xycar_data`는 게임패드·터미널로 Xycar를 조종하고 카메라 기반
behavior-cloning 모델 학습용 데이터를 수집하는 패키지다. 게임패드 조종은
카메라·LiDAR·녹화와 독립적이다. 터미널 Teleop의 학습 입력 기준은 카메라
frame이며, 각 PNG frame에는 그 순간 발행한 연속형 `angle`, `speed`가 label로
저장된다.

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

`teleop_recorder`는 키보드 TTY가 필요하므로 `ros2 launch`로 실행하지 않는다.
SSH 연결에도 `-t`를 유지한다.

## Remote Gamepad Teleop

Remote Gamepad 휴대폰 앱과 PC 앱을 먼저 연결한다. 차량 바퀴를 지면에서 띄우고
다른 `/xycar_motor` publisher가 없는지 확인한 뒤 다음 launch를 실행한다.

```bash
ros2 launch xycar_data gamepad_teleop.launch.py
```

launch는 ROS 2 공식 `joy/game_controller_node`와 `gamepad_teleop`을 함께
시작한다. 입력과 출력은 다음과 같다.

| Gamepad 입력 | 변환 | 범위 |
| --- | --- | --- |
| 왼쪽 스틱 좌우 `axes[0]` | `angle = -100 * axes[0]` | `-100 ~ 100` |
| LT `axes[4]` | `speed -= 5 * depth` | `0 ~ -5` |
| RT `axes[5]` | `speed += 7 * depth` | `0 ~ 7` |

Remote Gamepad 실측 기준은 trigger release `0`, full press `+1`이므로
`trigger_axis_mode: positive`가 기본값이다. `0`에서 `-1`로 움직이는 다른 SDL
controller는 `negative` profile을 사용한다. LT와 RT는 합산하므로 둘을 끝까지
누르면 speed는 `2`다. 두 트리거가 모두 0이면 speed만 0이 되고 angle은 왼쪽
스틱을 계속 따라간다. A/B 버튼은 사용하지 않는다.

`/joy`가 0.25초 이상 끊기거나, 축 배열이 잘못됐거나, motor subscriber가 없거나,
다른 motor publisher가 발견되면 `[0, 0]`을 발행한다. 종료할 때도 정지 명령을
5회 발행한다. 시작할 때와 위 안전 정지에서 복구할 때는 LT와 RT가 모두
`neutral_trigger_threshold` 이하인 입력을 한 번 확인해야 다시 주행할 수 있다.
연결이 유효하고 `/joy`가 갱신되는 동안 마지막 유효 명령은 20 Hz로 반복된다.

휴대폰 연결이 끊겨도 PC의 가상 controller가 마지막 `/joy`를 계속 갱신하면 이
node는 실제 Remote 앱 단절과 정상적인 고정 입력을 구분할 수 없다. 이 동작은
실차 주행 전에 별도로 확인해야 하며, 마지막 값이 계속 갱신되는 환경은 앱
heartbeat 또는 별도 deadman 없이는 단절 안전이 검증된 것으로 보지 않는다.

`/joy`가 이미 다른 승인된 `Joy` node에서 발행 중일 때는 teleop만 단독 실행할
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

trigger가 `0`에서 `-1`로 움직이는 controller는 별도 YAML에서
`trigger_axis_mode: negative`로 설정하고 `params_file` launch 인자로 전달한다.
조향 `±100`, 전진 `7`, 후진 `-5`는 현재 기본값이지만 실제 motor scale과
조향 끝값은 raised-car 상태에서 아직 검증되지 않았다. 첫 실차 시험은 낮은
trigger 깊이부터 시작한다.

## 센서 실행

첫 터미널에서 카메라와 선택적 LiDAR driver만 시작한다. 이 launch는 motor
publisher를 포함하지 않는다.

```bash
ros2 launch xycar_data teleop_sensors.launch.py
```

LiDAR가 없더라도 카메라 주행 데이터를 수집하려면 아래처럼 카메라만 시작한다.

```bash
ros2 launch xycar_data teleop_sensors.launch.py use_lidar:=false
```

이미 `/image_raw`와 `/scan`이 다른 승인된 node에서 publish 중이면 이 launch를
중복 실행하지 않는다. `/scan`은 없어도 recorder가 동작하며 해당 sample은
`lidar_valid=false`로 저장된다.

## Terminal Teleop과 수집

두 번째 SSH TTY에서 environment를 적용한 뒤 실행한다. 이 명령은
`/xycar_motor`에 `Float32MultiArray([angle, speed])`를 publish할 수 있으므로
매 실행 직전 승인이 필요하다.

```bash
ros2 run xycar_data teleop_recorder
```

다른 YAML을 시험할 때는 ROS parameter로 경로를 바꾼다.

```bash
ros2 run xycar_data teleop_recorder --ros-args \
  -p tuning_file:=/home/xytron/xycar_ws_mgw/src/xycar_data/config/teleop_recorder.yaml
```

기본 키와 명령은 다음과 같다.

| 키 | `[angle, speed]` | 동작 |
| --- | --- | --- |
| Up | `[0, 5]` | 전진 |
| Down | `[0, -3]` | 저속 후진 |
| Left | `[-30, 3]` | 좌회전 전진 |
| Right | `[30, 3]` | 우회전 전진 |
| R | — | 새 기록 세션 시작 |
| W | `[0, 0]` | 현재 세션 저장·마감 후 정지 |
| Space | `[0, 0]` | 세션을 유지한 즉시 정지 |
| Q, Esc, Ctrl+C | `[0, 0]` | 정지 명령 5회 후 종료 |

방향키는 OS key-repeat으로 유지되며 0.25초 동안 새 입력이 없으면 즉시 정지한다.
기록 세션이 없어도 수동 주행은 가능하지만, `R` 이후 유효한 방향키 명령과 최신
카메라 frame이 동시에 있을 때만 파일을 저장한다. 방향키가 없거나 정지 상태인
frame은 저장하지 않는다.

시작 전과 주행 중에는 다음 안전 gate가 적용된다.

- 카메라가 0.25초 이상 stale이면 즉시 정지하고 새 방향키 입력이 필요하다.
- 다른 `/xycar_motor` publisher가 있으면 주행을 거부한다. 주행 중 발견되면
  정지 후 오류 종료한다.
- motor subscriber가 없으면 command와 sample 저장을 거부한다.
- LiDAR는 선택적이다. 최신 scan이 없으면 경고만 남기고 camera sample을 저장한다.
- writer queue 포화, 디스크 여유 부족, 이미지·NPZ 쓰기 오류는 정지와
  `_incomplete_...` 세션 보존으로 처리한다.

## 데이터 형식

기본 저장 위치는 저장소 밖의 `/home/xytron/xycar_data/teleop`이며
`config/teleop_recorder.yaml`에서 바꿀 수 있다. 첫 유효 sample 전에는 directory를
만들지 않으므로 빈 세션은 남지 않는다.

```text
session_YYYYMMDD_HHMMSS_mmm/
  Images/000001.png
  Lidar/000001.npz
  samples.csv
  metadata.yaml
```

`samples.csv`의 각 행은 하나의 camera frame이다. `image`, `angle`, `speed`,
`input_key`, camera stamp/수신 시각과 `lidar_valid`를 기본 학습 label로 사용한다.
유효한 LiDAR가 있으면 `lidar` 경로와 scan timestamp·skew도 기록하며, 하나의
LiDAR scan이 여러 camera frame에 대응할 때 NPZ 파일을 재사용한다. NPZ에는
전체 `ranges`, `intensities`, LaserScan geometry·timing·frame metadata가 담긴다.

정상 `W` 또는 종료는 임시 `_recording_...` directory를 `session_...`으로
atomic rename한다. 쓰기 실패나 경쟁 motor publisher로 중단되면
`_incomplete_...`으로 남고 `metadata.yaml`의 `stop_reason`에서 원인을 확인할 수
있다.

## 튜닝

`config/teleop_recorder.yaml`과 `config/gamepad_teleop.yaml`은 다음을 분리한다.

- camera, LiDAR, motor topic
- 방향키 angle/speed, 20 Hz publish rate, key timeout, stop 반복 횟수
- gamepad axis·trigger 부호 profile, 중립 재활성 임계값과 출력 한계
- 필수 camera freshness와 선택 LiDAR 연결 허용 시간
- dataset root, PNG compression, writer queue, 최소 디스크 여유

YAML의 빈 topic, 잘못된 speed/steering 부호·범위, `NaN`·`Inf` 수치, 음수
timeout, queue 크기와 PNG compression 범위 오류는 node 시작 시 거부된다.
실제 차에서는 raised-car 상태에서 Left/Right 조향 부호와 후진 부호를 먼저
확인한 뒤 값을 조정한다.
