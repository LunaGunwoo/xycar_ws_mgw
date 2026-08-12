# xycar_cam

`xycar_cam`은 차량 USB camera의 RGB 영상을 `/image_raw`로 발행한다. 모든 명령은
활성 차량 `xytron@xycar-gpu:/home/xytron/xycar_ws_mgw`에 Tailscale SSH로 접속한
상태를 기준으로 한다.

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

Humble `v4l2_camera`와 udev alias `/dev/videoCAM`을 사용해 카메라가 지원하는
YUYV 640x480, 30 Hz를 요청하고 `rgb8` `/image_raw`를 발행한다. Jetson에서
`usb_cam` 0.8.1의 FFmpeg 색상 변환 경로가 `char*` 예외로 abort한 현장 결과 때문에
FFmpeg를 사용하지 않는 V4L2 변환 경로를 기준으로 한다. AI와 recorder가 사용하는
raw topic과 image 계약은 바뀌지 않는다.

## Camera viewer

```bash
ros2 launch xycar_cam xycar_cam_viewer.launch.py
```

camera와 `image_view` GUI를 함께 실행한다. GUI와 camera 장치를 모두 사용하므로
이 명령도 매번 별도 승인이 필요하다.

두 launch 모두 `Ctrl+C`로 종료한다.
