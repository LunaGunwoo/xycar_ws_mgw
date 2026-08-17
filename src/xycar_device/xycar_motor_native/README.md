# xycar_motor_native

ROS 1 bridge 없이 Jetson ROS 2 Humble에서 VESC를 직접 구동하는 history 전용
gateway다. 기존 `/xycar_motor` stateless 경로는 변경하지 않는다.

## Build

차량 source 기준으로 고정된 F1TENTH VESC ROS 2 source를 준비하고 필요한 package만
빌드한다. 준비 script는 `_deps/`의 기존 checkout이 다른 commit이거나 dirty하면
덮어쓰지 않고 중단한다.

```bash
cd /home/xytron/xycar_ws_mgw
./deploy/jetson/prepare_native_vesc_source.sh
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --base-paths src _deps/src/f1tenth_vesc \
  --packages-select vesc_msgs vesc_driver xycar_msgs xycar_motor_native
source install/setup.bash
```

Upstream은 `f1tenth/vesc` ROS 2 commit
`c47fccbbd10fb66db3faaaa6e469f2eedba2586f`로 고정한다. 이 commit은 차량의
VESC firmware 2.18과 같은 legacy values protocol을 사용하며, firmware 5.2용
packet layout과 IMU polling이 추가되기 전 버전이다. BSD-3-Clause license 사본은
`deploy/jetson/F1TENTH_VESC_LICENSE`에 보존한다.

## Topics

- 입력 `/xycar_motor_command`: `xycar_msgs/msg/XycarMotor`
- 실행 echo `/xycar_motor_executed`: clamp와 ramp 뒤 실제 VESC command로 변환한 값
- 준비 상태 `/xycar_motor_native/ready`: fresh VESC feedback과 안전한 graph일 때 true
- 내부 `/xycar_native/commands/motor/speed`: ERPM `Float64`
- 내부 `/xycar_native/commands/servo/position`: servo `Float64`
- feedback `/xycar_native/sensors/core`: VESC state

정상 command는 callback에서 한 번만 즉시 전달한다. 30 Hz timer는 command를
재발행하지 않고 stale·feedback·graph 상태만 검사한다. source header stamp가 0,
중복 또는 역순이거나 command가 NaN/Inf이면 즉시 정지한다.
fault는 ready false로 latch되고, 환경이 안전해진 뒤 새 source stamp의 zero command를
받아야 native gateway가 다시 ready가 된다. 상위 policy/collector도 별도 A/Y 또는
trigger 중립 재-arm을 요구하므로 fault 직전 nonzero가 자동 재개되지 않는다.

## Real-car launch

아래 명령은 `/dev/ttyMOTOR`를 열고 motor command를 publish하므로 실행할 때마다
사용자 승인, 바퀴 지지, 전원 차단 방법, `[0,0]` 정지와 `Ctrl+C` 경로를 먼저
확인한다. 기존 `motor`/ROS1 container를 동시에 실행하지 않는다.

```bash
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
source install/setup.bash
ROS_DOMAIN_ID=7 ROS_LOCALHOST_ONLY=1 \
ros2 launch xycar_motor_native vesc_motor.launch.py \
  params_file:=/home/xytron/xycar_ws_mgw/install/xycar_motor_native/share/xycar_motor_native/config/native_vesc.yaml
```

종료는 command source를 먼저 DRIVE OFF로 바꾸고 `[0,0]` echo를 확인한 뒤 이
terminal에서 `Ctrl+C`를 누른다. `mock_driver:=true`도 motor publisher를 시작하므로
실차 checkout에서는 승인 없는 실행 용도로 사용하지 않는다.

launch preflight 없이 gateway component만 시작하는 아래 명령은 mock test harness
외에는 사용하지 않는다. 실제 graph에서는 serial driver 소유권과 legacy container
검사를 우회하므로 금지하며, motor publisher를 생성하므로 실행 직전 승인 대상이다.

```bash
ros2 run xycar_motor_native native_motor_gateway --ros-args \
  --params-file /home/xytron/xycar_ws_mgw/install/xycar_motor_native/share/xycar_motor_native/config/native_vesc.yaml
```
