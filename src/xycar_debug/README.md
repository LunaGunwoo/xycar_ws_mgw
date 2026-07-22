# xycar_debug

실차 LiDAR 라바콘 인식 결과와 중앙 경로를 GUI로 확인하고, 같은 창에서 Space
키로 저속 실차 주행을 켜고 끄는 ROS 2 디버그 패키지다. 모든 실행 방법은 실제
차량 `xytron@10.42.0.1:/home/xytron/xycar_ws_mgw`를 기준으로 한다.

## SSH 접속, 빌드와 환경 적용

```bash
ssh -X xytron@10.42.0.1
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
git pull --ff-only
colcon build --symlink-install \
  --packages-select xycar_lidar track_drive xycar_debug
source /home/xytron/xycar_ws_mgw/install/setup.bash
ros2 pkg prefix xycar_lidar
```

마지막 명령은 `/home/xytron/xycar_ws_mgw/install/xycar_lidar`를 출력해야 한다.
`/home/xytron/xycar_ws/install/xycar_lidar`가 출력되면 새 workspace 빌드가
완료되지 않았거나 새 `install/setup.bash`를 source하지 않은 상태다.
SSH에서 GUI를 띄우려면 X forwarding이 필요하므로 `ssh -X`로 접속하고
`echo "$DISPLAY"` 결과가 비어 있지 않은지 확인한다.

## LiDAR 포함 통합 Viewer 실행

다음 launch는 `/dev/ttyLIDAR`를 여는 LiDAR driver와 motor publisher를 가진
Viewer를 함께 시작한다. 실행할 때마다 승인을 받고, 차량 바퀴를 지면에서
분리하거나 안전 공간을 확보하며, 기존 주행 노드를 먼저 종료한다.

```bash
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws_mgw/install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar
ros2 launch xycar_debug cone_path_viewer.launch.py
```

이미 별도로 `/scan`이 publish되고 있어 LiDAR를 중복 실행하지 않을 때는 아래
명령을 사용한다. 이 명령도 motor publisher를 만들므로 동일하게 승인이 필요하다.

```bash
ros2 run xycar_debug cone_path_viewer
```

GUI에는 원시 LiDAR 점, 필터 점, 라바콘 cluster와 좌우 경계, 중앙 경로,
lookahead, 예상 회전 궤적, preview 명령과 실제 motor 명령이 계속 표시된다.

## 주행 조작과 안전 동작

- 시작 상태는 항상 `DRIVE OFF`이며 자동 출발하지 않는다.
- 유효한 최신 콘 경로가 보일 때 Space를 누르면 `DRIVE ON`이 된다.
- 다시 Space를 누르면 즉시 `[0, 0]`을 5회 publish하고 `DRIVE OFF`가 된다.
- OFF 상태에서 다시 Space를 누르면 현재 경로를 다시 검사한 뒤 재출발한다.
- 경로·scan 유실 시 즉시 `[0, 0]`으로 멈춘다. 0.5초 안에 복구되면 ON 상태로
  자동 재개하고, 0.5초를 넘으면 OFF로 전환돼 다시 Space를 눌러야 한다.
- 단일 콘, 낮은 confidence, stale scan과 재사용 경로에서는 움직이지 않는다.
- 다른 motor publisher가 있으면 ON 요청을 거부한다. 주행 중 발견되면 즉시
  정지하고 OFF로 전환하며 GUI에 node 이름을 표시한다.
- `Q`, `Esc`, 창 닫기 또는 `Ctrl+C`는 정지 명령을 5회 보낸 뒤 종료한다.

현재 차량에서 `/driver`가 `/xycar_motor`를 publish 중이면 Space 활성화가
거부되는 것이 정상이다. 통합 Viewer가 다른 주행 node를 자동 종료하지 않는다.

## 튜닝

화면과 Space 토글 생명주기는 `config/cone_viewer.yaml`에서 조정한다.

- `path_loss_timeout_sec`: 경로 유실 후 OFF로 전환할 시간
- `stop_publish_count`: OFF·종료 시 정지 명령 반복 횟수
- `key_debounce_sec`: Space 길게 누름에 의한 중복 전환 방지 시간

LiDAR 필터, 차로 폭, confidence, motor topic, steering과 speed 3~5는
`../track_drive/config/cone_drive.yaml`에서 조정한다.
실차에서 확인된 기본 motor topic은 절대 경로 `/xycar_motor`다.

## GUI shell

`etc/gui-shell/x28.sh`도 실제 차량의
`/home/xytron/xycar_ws_mgw/install/setup.bash`를 source한 뒤 통합 Viewer launch를
실행한다. 이 shell은 LiDAR와 motor publisher를 시작하므로 매 실행 전 승인이
필요하다.
