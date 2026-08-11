# xycar_cam

`xycar_cam`은 차량 USB camera의 RGB 영상을 `/image_raw`로 발행한다. 모든 명령은
차량 `xytron@xycar:/home/xytron/xycar_ws_mgw`에 Tailscale SSH로 접속한 상태를 기준으로
한다.

camera 장치를 여는 명령이므로 매 실행 직전에 사용자 승인을 받고, 다른 camera
node가 `/dev/video*` 또는 `/image_raw`를 사용 중이지 않은지 확인한다.

## Build와 환경 적용

```bash
cd /home/xytron/xycar_ws_mgw
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select xycar_cam
source /home/xytron/xycar_ws_mgw/install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar
```

## Headless camera

```bash
ros2 launch xycar_cam xycar_cam.launch.py
```

기본 `usb_cam` parameter로 640x480, 30 Hz, `rgb8` `/image_raw`를 발행한다.
publisher plugin은 `image_transport/raw`만 활성화한다. RGB frame을 depth 전용
`compressedDepth` plugin으로 보내면서 발생하는 반복 compression 오류를 막고,
recorder가 사용하는 raw topic 계약을 유지하기 위한 설정이다.

## Camera viewer

```bash
ros2 launch xycar_cam xycar_cam_viewer.launch.py
```

camera와 `show_image.py` GUI를 함께 실행하며 raw transport 제한은 headless
launch와 같다. GUI와 camera 장치를 모두 사용하므로 이 명령도 매번 별도 승인이
필요하다.

두 launch 모두 `Ctrl+C`로 종료한다.
