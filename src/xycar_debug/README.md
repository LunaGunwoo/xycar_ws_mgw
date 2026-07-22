# xycar_debug

LiDAR 라바콘 경로를 화면에서 확인하거나, 라바콘 구간 하나만 저속으로
주행하기 위한 ROS 2 디버그 패키지다. launch 파일은 LiDAR나 motor driver를
자동으로 시작하지 않는다.

## 빌드와 환경 적용

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw
colcon build --symlink-install --packages-select track_drive xycar_debug
source install/setup.bash
```

## 경로 Viewer

이미 `/scan`이 publish되고 있을 때 다음 중 하나로 실행한다.

```bash
ros2 run xycar_debug cone_path_viewer
ros2 launch xycar_debug cone_path_viewer.launch.py
```

Viewer는 `/scan`만 구독하며 motor publisher를 만들지 않는다. GUI에는 원시
LiDAR 점, 라바콘 cluster와 좌우 경계, 중앙 경로, lookahead, 예상 회전 궤적과
예상 `[angle, speed]`가 표시된다. `Ctrl+C`를 누르거나 창을 닫아 종료한다.

화면 범위와 표시 주기는 `config/cone_viewer.yaml`에서 조정한다. 콘 인식과
경로 설정은 `../track_drive/config/cone_drive.yaml`을 공용으로 사용한다.

## 콘 단일 미션 주행

아래 명령은 기본 `xycar_motor` 토픽에
`std_msgs/Float32MultiArray([angle, speed])`를 publish할 수 있다. 실행할 때마다
사용자 승인을 받고 차량 지지 상태, 비상 정지 방법, LiDAR 및 motor topic을
확인해야 한다.

```bash
ros2 run xycar_debug cone_debug_drive
ros2 launch xycar_debug cone_debug_drive.launch.py
```

노드는 최신 유효 라바콘 경로가 3 frame 연속 확인되면 별도 서비스 없이 자동
출발한다. 속도는 3~5로 제한되고 경로가 유실된 동안에는 즉시 정지한다. 경로가
0.5초 안에 회복되지 않으면 정지 명령을 5회 publish한 뒤 정상 종료한다. 경쟁
motor publisher 또는 처리 예외가 확인되면 정지 후 실패 종료한다. 수동 종료는
`Ctrl+C`이며 이때도 정지 명령을 반복한다.

자동 출발 frame 수, 경로 유실 시간과 종료 정지 횟수는
`config/cone_debug_drive.yaml`에서 조정한다. LiDAR 필터, 차로, 조향과 속도는
`../track_drive/config/cone_drive.yaml`에서 조정한다.
