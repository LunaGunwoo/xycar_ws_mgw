# track_drive

실차 LiDAR 라바콘 인식, 중앙 경로 계획과 저속 조향 제어를 제공하는 공용
Python 라이브러리다. ROS node, motor publisher, `console_scripts` 또는 launch
파일을 제공하지 않는다.

## 빌드

```bash
ssh xytron@10.42.0.1
cd /home/xytron/xycar_ws_mgw
colcon build --symlink-install --packages-select track_drive
source install/setup.bash
```

`track_drive.cone_following`, `track_drive.control`, `track_drive.tuning`은
`xycar_debug`의 Viewer 겸 Space-key 콘 주행기가 함께 사용한다. 전체 맵 주행용
`ros2 run track_drive ...` 또는 `ros2 launch track_drive ...` 명령은 현재 없다.

공용 튜닝 값은 `config/cone_drive.yaml`에 있으며 `/scan`, `/xycar_motor`, LiDAR
필터, cluster, 차로 폭, lookahead, confidence, speed 3~5, steering gain과 clamp를
관리한다. 설정은 노드 시작 시 검증되며 음수 속도, 역전된 최솟값·최댓값,
최대 속도보다 빠른 급커브 속도와 ±100을 넘는 조향 한계는 거부된다.
