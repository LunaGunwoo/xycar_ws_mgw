# src/study 패키지 상세 정리

이 문서는 `src/study/` 아래에 있는 ROS 2 Python 패키지들이 각각 어떤 역할을 하는지, 어떤 토픽을 사용하고, 어떤 실행 파일과 launch 파일을 제공하는지 코드 기준으로 정리한 문서입니다. 아래 실차용 패키지는 학습 예제와 별도로 관리합니다.

## 실차 LiDAR 라바콘 패키지

### `track_drive`

- 공용 설정: `track_drive/config/cone_drive.yaml`
- 실행 파일과 launch 없음

LaserScan metadata로 점을 차량 좌표로 변환하고 cone cluster와 좌우 경계를
구성한 뒤 중앙 경로의 Pure Pursuit lookahead를 계산합니다. 정상 양쪽 경로는
speed 3~5, 현재 frame의 한쪽 경계 추정은 speed 3으로 제한합니다. 기본
steering clamp는 ±30이고 YAML에서 최대 ±100까지 설정할 수 있습니다.

이 패키지는 ROS sensor message나 motor publisher를 소유하지 않는 공용
planner/control library입니다. 전체 코스용 실행 진입점은 아직 구현하지 않습니다.

### `xycar_debug`

- 공용 입력: 기본 `/scan`
- 통합 Viewer/주행: `ros2 launch xycar_debug cone_path_viewer.launch.py`
- 기존 `/scan` 사용: `ros2 run xycar_debug cone_path_viewer`
- 화면 설정: `xycar_debug/config/cone_viewer.yaml`

원시 점군, cone 후보와 좌우 경계, 중앙 경로, lookahead, 예상 회전 arc와
preview/actual angle·speed를 하나의 GUI에 표시합니다. 시작은 항상 DRIVE OFF이며
Space로 실제 주행을 ON/OFF 합니다. ON 상태의 양쪽 경계는 speed 3~5, 급커브와
한쪽 경계는 speed 3으로 주행합니다. 경로가 사라지면 즉시 정지하고 0.5초 안에
복구되면 재개하며, 그 시간을 넘으면 OFF로 바뀝니다. launch는 LiDAR driver를
포함하고 node는 motor command를 publish할 수 있으므로 매 실행 전 승인이
필요합니다.

실행 문서와 경로는 실제 차량
`xytron@10.42.0.1:/home/xytron/xycar_ws_mgw`를 기준으로 관리합니다.

## 전체 구조 요약

`src/study/`는 자율주행 자동차 실습을 센서별/기능별로 나눈 학습용 ROS 2 패키지 묶음입니다. 모든 패키지는 `ament_python` 방식으로 작성되어 있고, `setup.py`의 `console_scripts`를 통해 Python 노드를 실행합니다.

| 패키지 | 핵심 역할 | 주요 입력 | 주요 출력 | 성격 |
| --- | --- | --- | --- | --- |
| `my_cam` | 카메라 영상 수신, OpenCV 전처리, 노출값 변경 실습 | `/image_raw` | OpenCV 창 표시, 카메라 exposure 설정 | 카메라/영상처리 기초 |
| `my_lidar` | LiDAR 거리 데이터 확인 및 2D 시각화 | `/scan` 또는 `scan` | 로그 출력, Matplotlib 시각화 | LiDAR 센서 확인 |
| `my_ultra` | 초음파 센서 배열값 수신 및 출력 | `xycar_ultrasonic` | 로그 출력 | 초음파 센서 확인 |
| `my_imu` | IMU quaternion을 roll/pitch/yaw로 변환해 출력 | `/imu` | 로그 출력 | 자세 센서 확인 |
| `my_motor` | 모터 명령 토픽에 조향각/속도 명령 발행 | 없음 | `xycar_motor` | 구동 명령 발행 실습 |
| `my_hough` | 카메라 영상에서 Hough Transform으로 차선을 검출하고 주행 명령 발행 | `/image_raw` | `xycar_motor`, OpenCV 디버그 창 | 차선 인식 주행 |

## 공통 패키지 패턴

각 패키지는 대체로 다음 구조를 갖습니다.

```text
패키지명/
  package.xml
  setup.py
  setup.cfg
  resource/패키지명
  launch/*.launch.py
  패키지명/*.py
  test/test_*.py
```

공통 구성의 의미는 다음과 같습니다.

- `package.xml`: ROS 2 패키지 메타데이터입니다. 대부분 `build_type`은 `ament_python`입니다.
- `setup.py`: Python 패키지 설치 설정과 `ros2 run`으로 실행 가능한 `console_scripts`를 정의합니다.
- `setup.cfg`: 개발/설치 시 실행 스크립트가 `$base/lib/<package>`에 들어가도록 지정합니다.
- `resource/<package>`: ROS 2 ament index에 패키지를 등록하기 위한 marker 파일입니다.
- `launch/*.launch.py`: 관련 하드웨어 드라이버 패키지를 함께 실행하거나, 해당 패키지 노드를 쉽게 실행하기 위한 launch 파일입니다.
- `test/test_copyright.py`, `test/test_flake8.py`, `test/test_pep257.py`: ROS 2 Python 패키지 생성 시 만들어지는 기본 품질 검사 테스트입니다.
- `__pycache__/`: Python 실행 중 생성된 캐시 파일입니다. 소스 역할은 없으므로 문서에서는 제외합니다.

코드에서 실제로 쓰는 Python 모듈과 `package.xml`의 의존성 선언이 완전히 일치하지 않는 부분도 있습니다. 예를 들어 여러 패키지에서 `sensor_msgs`, `std_msgs`, `cv_bridge`, `OpenCV`, `numpy`, `matplotlib`, `tf_transformations` 등을 사용하지만, 일부 `package.xml`에는 이 의존성이 명시되어 있지 않습니다. 학습 환경에 이미 설치되어 있어 동작할 수는 있지만, 배포/재현성을 높이려면 의존성 선언을 보강하는 것이 좋습니다.

## 토픽 흐름 요약

주요 토픽 흐름은 다음과 같습니다.

| 토픽 | 메시지 타입 | 사용하는 패키지 | 방향 | 의미 |
| --- | --- | --- | --- | --- |
| `/image_raw` | `sensor_msgs/msg/Image` | `my_cam`, `my_hough` | subscribe | 카메라 원본 이미지 |
| `/scan` 또는 `scan` | `sensor_msgs/msg/LaserScan` | `my_lidar` | subscribe | LiDAR 거리 스캔 |
| `xycar_ultrasonic` | `std_msgs/msg/Int32MultiArray` | `my_ultra` | subscribe | 초음파 센서 배열 데이터 |
| `/imu` | `sensor_msgs/msg/Imu` | `my_imu` | subscribe | IMU 자세 데이터 |
| `xycar_motor` | `std_msgs/msg/Float32MultiArray` | `my_motor`, `my_hough` | publish | `[angle, speed]` 형태의 구동 명령 |

`xycar_motor`는 예전 코드 주석상 `xycar_msgs/msg/XycarMotor`를 쓰던 흔적이 있지만, 현재 소스는 `std_msgs/msg/Float32MultiArray`를 사용합니다. 데이터는 `[조향각, 속도]` 순서로 채워집니다.

## 패키지별 상세 설명

## 1. `my_cam`

### 한 줄 역할

카메라 토픽 `/image_raw`를 받아 OpenCV 이미지로 변환한 뒤, grayscale, Gaussian blur, Canny edge 결과를 화면에 보여주는 카메라/영상처리 기초 실습 패키지입니다. 추가로 카메라 노출값을 `v4l2-ctl`로 변경하는 예제도 포함합니다.

### 주요 파일

| 파일 | 역할 |
| --- | --- |
| `my_cam/my_cam/edge_cam.py` | 카메라 이미지 수신 후 기본 영상처리 결과를 OpenCV 창에 표시 |
| `my_cam/my_cam/exposure_cam.py` | `edge_cam.py`와 같은 영상처리를 수행하면서 카메라 노출값을 주기적으로 변경 |
| `my_cam/launch/edge_cam.launch.py` | `xycar_cam` launch를 포함하고 `edge_cam` 노드를 실행 |
| `my_cam/launch/exposure_cam.launch.py` | `xycar_cam` launch를 포함하고 `exposure_cam` 노드를 실행 |
| `my_cam/setup.py` | `edge_cam`, `exposure_cam` 실행 엔트리 등록 |

### 실행 엔트리

`setup.py`에는 다음 실행 파일이 등록되어 있습니다.

```text
edge_cam = my_cam.edge_cam:main
exposure_cam = my_cam.exposure_cam:main
```

따라서 빌드 후 다음처럼 실행할 수 있습니다.

```bash
ros2 run my_cam edge_cam
ros2 run my_cam exposure_cam
```

launch 파일을 사용할 경우 다음처럼 실행합니다.

```bash
ros2 launch my_cam edge_cam.launch.py
ros2 launch my_cam exposure_cam.launch.py
```

### `edge_cam.py` 동작 흐름

`edge_cam.py`의 핵심 클래스는 `CamTuneNode`입니다.

1. ROS 2 노드 이름을 `cam_tune`으로 초기화합니다.
2. `CvBridge`를 생성해 `sensor_msgs/msg/Image`를 OpenCV 이미지로 변환할 준비를 합니다.
3. `/image_raw` 토픽을 구독합니다.
4. 첫 카메라 이미지가 들어올 때까지 `rclpy.spin_once()`를 반복하며 대기합니다.
5. 이미지가 들어오면 `run()` 루프에서 계속 최신 이미지를 처리합니다.
6. `process_images()`에서 다음 전처리 결과를 만듭니다.
   - 원본 BGR 이미지
   - grayscale 이미지
   - `cv2.GaussianBlur(gray, (5, 5), 0)` 결과
   - `cv2.Canny(..., 60, 70)` edge 이미지
7. 각 결과를 `cv2.imshow()`로 표시합니다.

이 노드는 퍼블리셔가 없고, 결과 이미지를 ROS 토픽으로 다시 내보내지도 않습니다. 목적은 화면 확인과 전처리 실험입니다.

### `exposure_cam.py` 동작 흐름

`exposure_cam.py`는 `edge_cam.py`와 거의 같은 구조지만, 카메라 노출값을 바꾸는 기능이 추가되어 있습니다.

추가 속성:

```text
self.exposure_value = 0
```

추가 함수:

```text
cam_exposure(value)
```

이 함수는 다음 명령을 OS shell로 실행합니다.

```bash
v4l2-ctl -d /dev/videoCAM -c auto_exposure=1
v4l2-ctl -d /dev/videoCAM -c exposure_time_absolute=<value>
```

`run()` 루프에서는 이미지 처리 후 다음을 반복합니다.

1. `exposure_value`를 0부터 255까지 순환시킵니다.
2. 현재 노출값을 로그로 출력합니다.
3. `cam_exposure()`로 실제 카메라 장치 노출값을 변경합니다.
4. 0.5초 대기합니다.

### launch 파일 구조

두 launch 파일은 모두 `xycar_cam` 패키지의 `xycar_cam.launch.py`를 먼저 include합니다.

```text
xycar_cam/launch/xycar_cam.launch.py
```

그 뒤 `my_cam`의 노드를 실행합니다.

```text
edge_cam.launch.py      -> executable='edge_cam'
exposure_cam.launch.py  -> executable='exposure_cam'
```

즉, 카메라 드라이버 실행과 학습용 영상처리 노드 실행을 한 번에 묶어둔 launch입니다.

### 주의점

- OpenCV GUI 창을 띄우므로 그래픽 디스플레이 환경이 필요합니다.
- `/image_raw` 토픽이 들어오지 않으면 노드 초기화 중 계속 대기합니다.
- `exposure_cam.py`는 `/dev/videoCAM` 장치 이름과 `v4l2-ctl` 설치를 전제로 합니다.
- `exposure_cam.py`는 실제 카메라 설정을 변경하므로 다른 카메라 노드와 동시에 사용할 때 영향이 있을 수 있습니다.
- `package.xml`에는 `rclpy`, `xycar_msgs`만 의존성으로 적혀 있지만, 코드상 `sensor_msgs`, `cv_bridge`, `cv2`, `numpy`도 필요합니다.

## 2. `my_lidar`

### 한 줄 역할

LiDAR의 `LaserScan` 데이터를 받아 거리값을 로그로 확인하거나, 2D 평면에 점으로 그려 센서 스캔 형태를 시각화하는 패키지입니다.

### 주요 파일

| 파일 | 역할 |
| --- | --- |
| `my_lidar/my_lidar/lidar_scan.py` | LiDAR 거리 배열 일부를 cm 단위로 변환해 주기적으로 로그 출력 |
| `my_lidar/my_lidar/lidar_viewer.py` | LiDAR 거리 배열을 극좌표에서 직교좌표로 변환해 Matplotlib으로 표시 |
| `my_lidar/launch/lidar_scan.launch.py` | `xycar_lidar` launch를 포함하고 `lidar_scan` 실행 |
| `my_lidar/launch/lidar_viewer.launch.py` | `xycar_lidar` launch를 포함하고 `lidar_viewer` 실행 |
| `my_lidar/setup.py` | `lidar_scan`, `lidar_viewer` 실행 엔트리 등록 |

### 실행 엔트리

```text
lidar_scan = my_lidar.lidar_scan:main
lidar_viewer = my_lidar.lidar_viewer:main
```

실행 예시는 다음과 같습니다.

```bash
ros2 run my_lidar lidar_scan
ros2 run my_lidar lidar_viewer
ros2 launch my_lidar lidar_scan.launch.py
ros2 launch my_lidar lidar_viewer.launch.py
```

### `lidar_scan.py` 동작 흐름

`lidar_scan.py`의 핵심 클래스는 `LidarNode`입니다.

1. 노드 이름을 `lidar_node`로 초기화합니다.
2. LiDAR 데이터에 적합하도록 QoS를 `BEST_EFFORT`로 설정합니다.
3. `scan` 토픽을 `sensor_msgs/msg/LaserScan` 타입으로 구독합니다.
4. 콜백에서 `msg.ranges[1:505]`만 저장합니다.
5. 첫 데이터가 올 때까지 `wait_for_message()`에서 대기합니다.
6. 1초 타이머로 거리값을 출력합니다.

출력 방식:

- 저장된 거리 개수를 출력합니다.
- meter 단위의 `ranges` 값을 cm 단위 정수로 변환합니다.
- 전체 배열을 모두 출력하지 않고 약 18개 간격 샘플만 출력합니다.

`msg.ranges[1:505]`를 쓰므로 일반적으로 504개의 거리값을 사용합니다. 센서 전체 스캔 중 특정 구간만 잘라 보는 학습 코드로 볼 수 있습니다.

### `lidar_viewer.py` 동작 흐름

`lidar_viewer.py`의 핵심 클래스는 `LidarVisualizer`입니다.

1. 노드 이름을 `lidar_visualizer`로 초기화합니다.
2. QoS는 `BEST_EFFORT`입니다.
3. `/scan` 토픽을 구독합니다.
4. 콜백에서 `msg.ranges[1:505]`를 `numpy.array`로 저장합니다.
5. Matplotlib figure를 만들고 x/y 축을 `-150cm ~ 150cm` 범위로 설정합니다.
6. 0.1초 타이머마다 최신 LiDAR 값을 2D 점으로 갱신합니다.

좌표 변환 방식:

```text
angles = linspace(0, 2*pi, len(ranges)) - pi/2
x = ranges * cos(angles) * 100
y = ranges * sin(angles) * 100
```

즉, LaserScan 거리값을 원형 스캔으로 보고 cm 단위 직교좌표로 변환합니다. `-pi/2` 보정은 화면에서 스캔 방향을 보기 좋게 회전시키기 위한 값입니다.

### launch 파일 구조

두 launch 파일 모두 `xycar_lidar`의 `xycar_lidar.launch.py`를 include합니다.

```text
xycar_lidar/launch/xycar_lidar.launch.py
```

그 뒤 각각 `lidar_scan` 또는 `lidar_viewer`를 실행합니다.

### 주의점

- `lidar_scan.py`는 구독 토픽을 `scan`으로 쓰고, `lidar_viewer.py`는 `/scan`으로 씁니다. namespace를 쓰는 환경에서는 이 차이가 의미를 가질 수 있습니다.
- Matplotlib GUI가 필요한 `lidar_viewer.py`는 그래픽 환경이 있어야 제대로 표시됩니다.
- `lidar_viewer.py`는 0.1초마다 그래프와 로그를 갱신하므로 터미널 출력이 많을 수 있습니다.
- `package.xml`에는 `rclpy`, `xycar_msgs`만 적혀 있지만, 코드상 `sensor_msgs`, `numpy`, `matplotlib`도 필요합니다.

## 3. `my_ultra`

### 한 줄 역할

초음파 센서 배열 토픽 `xycar_ultrasonic`을 받아 값이 들어오는지 확인하고, 1초마다 로그로 출력하는 센서 확인용 패키지입니다.

### 주요 파일

| 파일 | 역할 |
| --- | --- |
| `my_ultra/my_ultra/ultra_scan.py` | 초음파 센서 배열값 구독 및 로그 출력 |
| `my_ultra/launch/ultra_scan.launch.py` | `xycar_ultrasonic` 관련 launch를 포함하고 `ultra_scan` 실행 |
| `my_ultra/setup.py` | `ultra_scan` 실행 엔트리 등록 |

### 실행 엔트리

```text
ultra_scan = my_ultra.ultra_scan:main
```

실행 예시는 다음과 같습니다.

```bash
ros2 run my_ultra ultra_scan
ros2 launch my_ultra ultra_scan.launch.py
```

### `ultra_scan.py` 동작 흐름

`ultra_scan.py`의 핵심 클래스는 `UltraNode`입니다.

1. 노드 이름을 `ultra_node`로 초기화합니다.
2. `xycar_ultrasonic` 토픽을 `std_msgs/msg/Int32MultiArray` 타입으로 구독합니다.
3. 콜백에서 `msg.data`를 `self.ultra_msg`에 저장합니다.
4. 첫 초음파 데이터가 들어올 때까지 `wait_for_message()`에서 대기합니다.
5. 첫 데이터 수신 후 1초 타이머를 생성합니다.
6. 타이머 콜백에서 최신 초음파 배열값을 로그로 출력합니다.

초음파 데이터는 여러 센서의 거리값이 배열로 들어오는 형태입니다. 이 패키지는 배열의 각 인덱스가 어느 방향 센서인지 해석하지는 않고, 원본 배열을 그대로 출력합니다.

### launch 파일 구조

`ultra_scan.launch.py`는 다음 launch를 include합니다.

```text
xycar_ultrasonic/launch/xycar_ultrasonic_viewer.launch.py
```

그 뒤 `my_ultra`의 `ultra_scan` 노드를 실행합니다.

### 주의점

- `xycar_ultrasonic` 토픽이 들어오지 않으면 노드 초기화 중 계속 대기합니다.
- 코드상 실제 메시지 타입은 `std_msgs/msg/Int32MultiArray`입니다.
- `package.xml`에는 `rclpy`, `xycar_msgs`가 적혀 있지만, 코드상 `std_msgs`도 필요합니다.

## 4. `my_imu`

### 한 줄 역할

IMU 토픽에서 quaternion 자세값을 받아 roll, pitch, yaw Euler angle로 변환한 뒤 1초마다 로그로 출력하는 패키지입니다.

### 주요 파일

| 파일 | 역할 |
| --- | --- |
| `my_imu/my_imu/roll_pitch_yaw.py` | IMU quaternion을 roll/pitch/yaw로 변환해 출력 |
| `my_imu/launch/roll_pitch_yaw.launch.py` | `xycar_imu` launch를 포함하고 `roll_pitch_yaw` 실행 |
| `my_imu/setup.py` | `roll_pitch_yaw` 실행 엔트리 등록 |

### 실행 엔트리

```text
roll_pitch_yaw = my_imu.roll_pitch_yaw:main
```

실행 예시는 다음과 같습니다.

```bash
ros2 run my_imu roll_pitch_yaw
ros2 launch my_imu roll_pitch_yaw.launch.py
```

### `roll_pitch_yaw.py` 동작 흐름

`roll_pitch_yaw.py`의 핵심 클래스는 `ImuNode`입니다.

1. 노드 이름을 `imu_print`로 초기화합니다.
2. `/imu` 토픽을 `sensor_msgs/msg/Imu` 타입으로 구독합니다.
3. 콜백에서 orientation quaternion의 `x`, `y`, `z`, `w` 값을 리스트로 저장합니다.
4. 1초 타이머에서 quaternion을 Euler angle로 변환합니다.
5. `Roll`, `Pitch`, `Yaw`를 소수점 4자리로 로그 출력합니다.

변환에는 다음 함수를 사용합니다.

```text
tf_transformations.euler_from_quaternion()
```

출력 단위는 별도 변환을 하지 않으므로 radian입니다.

### launch 파일 구조

`roll_pitch_yaw.launch.py`는 다음 launch를 include합니다.

```text
xycar_imu/launch/xycar_imu.launch.py
```

그 뒤 `my_imu`의 `roll_pitch_yaw` 노드를 실행합니다.

### 주의점

- 코드 주석에는 `'/imu/data' Topic`이라고 되어 있지만 실제 구독 토픽은 `/imu`입니다.
- `wait_for_message()` 함수는 구현되어 있지만 현재 호출이 주석 처리되어 있습니다. 그래서 첫 메시지를 기다리지 않고 바로 `IMU Ready` 로그를 출력합니다.
- IMU 메시지가 아직 없으면 타이머 콜백에서 아무 것도 출력하지 않습니다.
- `package.xml`에는 런타임 의존성이 거의 적혀 있지 않지만, 코드상 `rclpy`, `sensor_msgs`, `tf_transformations`가 필요합니다.

## 5. `my_motor`

### 한 줄 역할

`xycar_motor` 토픽으로 `[angle, speed]` 형태의 `Float32MultiArray` 모터 명령을 계속 발행하는 기본 구동 테스트 패키지입니다.

### 주요 파일

| 파일 | 역할 |
| --- | --- |
| `my_motor/my_motor/go.py` | 모터 명령 퍼블리셔. 정지 명령 후 직진 속도 명령 반복 |
| `my_motor/launch/go.launch.py` | `go` 노드를 실행하고 `speed` 파라미터를 전달 |
| `my_motor/config/xycar_bridge_config.yaml` | `ros1_bridge`에서 `xycar_msgs`를 whitelist하는 설정 |
| `my_motor/setup.py` | `go` 실행 엔트리 등록 |

### 실행 엔트리

```text
go = my_motor.go:main
```

실행 예시는 다음과 같습니다.

```bash
ros2 run my_motor go
ros2 launch my_motor go.launch.py
```

### `go.py` 동작 흐름

`go.py`의 핵심 클래스는 `DriverNode`입니다.

1. 노드 이름을 `driver`로 초기화합니다.
2. `xycar_motor` 토픽 퍼블리셔를 생성합니다.
3. 메시지 타입은 `std_msgs/msg/Float32MultiArray`입니다.
4. `speed` 파라미터를 선언하고 기본값은 `50`입니다.
5. `main_loop()`에서 먼저 20번 정지 명령을 보냅니다.
   - `angle = 0`
   - `speed = 0`
   - 0.1초 간격
   - 총 약 2초
6. 이후 ROS가 살아있는 동안 계속 직진 명령을 보냅니다.
   - `angle = 0`
   - `speed = 20`
   - 0.1초 간격

퍼블리시되는 데이터는 다음 형태입니다.

```text
Float32MultiArray.data = [float(angle), float(speed)]
```

현재 활성 코드에서는 조향각이 항상 0이므로 직진 명령입니다.

### launch 파일 구조

`go.launch.py`는 별도 하드웨어 launch를 include하지 않고 `my_motor`의 `go` 노드만 실행합니다.

```text
parameters=[{'speed': 12}]
```

다만 현재 `go.py`는 파라미터로 읽은 `self.speed`를 초기화 이후 루프에서 `20`으로 다시 덮어씁니다. 그래서 launch에서 전달한 `speed: 12`는 현재 지속 주행 속도에는 반영되지 않습니다.

### `xycar_bridge_config.yaml`

내용은 다음과 같습니다.

```yaml
ros1_bridge:
  ros1_package_whitelist:
    - "xycar_msgs"
  ros2_package_whitelist:
    - "xycar_msgs"
```

이는 ROS 1과 ROS 2 사이에서 `xycar_msgs`를 bridge 대상으로 삼기 위한 설정으로 보입니다. 하지만 현재 `go.py`는 `xycar_msgs/msg/XycarMotor`가 아니라 `std_msgs/msg/Float32MultiArray`를 사용합니다. 따라서 이 설정은 과거 구현 또는 다른 실행 환경을 위한 잔여 설정일 수 있습니다.

### 주의점

- 실제 차량/모터가 연결된 환경에서는 실행 즉시 2초 정지 후 `speed=20` 직진 명령이 반복됩니다.
- `speed` launch 파라미터가 현재 루프 속도에 반영되지 않습니다.
- 기존 `XycarMotor` 메시지를 쓰던 코드가 주석으로 남아 있으므로, 모터 드라이버가 어떤 타입을 기대하는지 확인이 필요합니다.
- `package.xml`에는 런타임 의존성이 적혀 있지 않지만, 코드상 `rclpy`, `std_msgs`가 필요합니다.

## 6. `my_hough`

### 한 줄 역할

카메라 영상에서 ROI를 잘라 Canny edge와 Probabilistic Hough Transform으로 좌우 차선을 검출하고, 차선 중앙과 화면 중앙의 차이를 조향각으로 변환해 `xycar_motor`에 주행 명령을 보내는 차선 인식 주행 패키지입니다.

### 주요 파일

| 파일 | 역할 |
| --- | --- |
| `my_hough/my_hough/hough_drive.py` | ROS 2 차선 검출 주행 노드 |
| `my_hough/launch/hough_drive.launch.py` | `xycar_cam` launch를 포함하고 `hough_drive` 실행 |
| `my_hough/my_hough/python_codes/my_hough.py` | 정지 이미지 기반 Hough 차선 검출 단계별 실험 |
| `my_hough/my_hough/python_codes/my_hough2.py` | polygon mask와 `np.polyfit`을 쓰는 대안적 차선 검출 실험 |
| `my_hough/my_hough/python_codes/hough_find.py` | 동영상 파일 기반 Hough 차선 검출 실험 |
| `my_hough/my_hough/python_codes/line_pic*.png` | 정지 이미지 실습 자료 |
| `my_hough/my_hough/python_codes/*.mp4` | 동영상 실습 자료 |
| `my_hough/setup.py` | `hough_drive` 실행 엔트리 등록 |

### 실행 엔트리

```text
hough_drive = my_hough.hough_drive:main
```

실행 예시는 다음과 같습니다.

```bash
ros2 run my_hough hough_drive
ros2 launch my_hough hough_drive.launch.py
```

### `hough_drive.py` 주요 상수

차선 인식에 쓰이는 핵심 상수는 다음과 같습니다.

| 상수 | 값 | 의미 |
| --- | --- | --- |
| `WIDTH` | `640` | 입력 이미지 가로 크기 |
| `HEIGHT` | `480` | 입력 이미지 세로 크기 |
| `ROI_START_ROW` | `300` | 차선을 찾을 ROI 시작 row |
| `ROI_END_ROW` | `380` | 차선을 찾을 ROI 끝 row |
| `ROI_HEIGHT` | `80` | ROI 세로 높이 |
| `L_ROW` | `40` | ROI 내부에서 차선 위치를 판단할 기준 수평선 |
| `View_Center` | `320` | 화면 중심 x 좌표 |
| `Fix_Speed` | `12` | 차선 주행 시 고정 속도 |

즉, 원본 640x480 이미지에서 y=300부터 y=380까지의 아래쪽 일부 영역만 잘라 차선을 찾습니다. 차선 위치는 ROI 내부의 y=40 지점에서 좌우 차선과 만나는 x 좌표로 판단합니다.

### `LaneDriverNode` 초기화 흐름

`hough_drive.py`의 핵심 클래스는 `LaneDriverNode`입니다.

1. 노드 이름을 `lane_detection_node`로 초기화합니다.
2. `CvBridge`를 준비합니다.
3. `/image_raw` 토픽을 `sensor_msgs/msg/Image`로 구독합니다.
4. `xycar_motor` 토픽 퍼블리셔를 생성합니다.
5. 퍼블리시 메시지 타입은 `std_msgs/msg/Float32MultiArray`입니다.
6. `stop_car(2)`로 약 2초 동안 정지 명령을 보냅니다.
7. 첫 카메라 이미지가 들어올 때까지 대기합니다.
8. 카메라 준비 후 생성자 안에서 바로 `main_loop()`를 실행합니다.

생성자 내부에서 `main_loop()`가 실행되므로, `main()` 아래쪽의 `rclpy.spin(node)`는 일반적인 상황에서는 도달하기 어렵습니다. 이 구조는 “노드를 만든 뒤 spin”하는 전형적인 ROS 2 패턴과는 조금 다르지만, 루프 내부에서 `rclpy.spin_once()`를 직접 호출하므로 콜백 처리는 이루어집니다.

### 영상처리 파이프라인

`lane_detect(image)`의 처리 순서는 다음과 같습니다.

1. 원본 이미지를 복사합니다.
2. ROI를 자릅니다.

```text
roi_img = image[300:380, 0:640]
```

3. ROI를 grayscale로 변환합니다.
4. Gaussian blur를 적용합니다.

```text
cv2.GaussianBlur(gray, (5, 5), 0)
```

5. Canny edge를 적용합니다.

```text
cv2.Canny(..., 60, 75)
```

6. Probabilistic Hough Transform으로 선분을 찾습니다.

```text
cv2.HoughLinesP(edge_img, 1, pi/180, 50, 50, 20)
```

파라미터 의미는 대략 다음과 같습니다.

| 인자 | 값 | 의미 |
| --- | --- | --- |
| `rho` | `1` | 거리 해상도 1 pixel |
| `theta` | `pi/180` | 각도 해상도 1 degree |
| `threshold` | `50` | 직선으로 인정할 누적 vote 기준 |
| `minLineLength` | `50` | 선분 최소 길이 |
| `maxLineGap` | `20` | 같은 선으로 연결할 최대 간격 |

### 좌우 차선 분류 방식

Hough로 찾은 선분에 대해 기울기를 계산합니다.

```text
slope = (y2 - y1) / (x2 - x1)
```

수직선인 경우 `slope = 1000.0`으로 처리합니다.

그 뒤 다음 조건으로 필터링합니다.

- `abs(slope) > 0.2`: 거의 수평인 선은 제거합니다.
- `slope < 0`이고 선분이 화면 왼쪽에 있으면 왼쪽 차선 후보입니다.
- `slope > 0`이고 선분이 화면 오른쪽에 있으면 오른쪽 차선 후보입니다.

분류 기준은 다음과 같습니다.

```text
left:  slope < 0 and x2 < WIDTH / 2
right: slope > 0 and x1 > WIDTH / 2
```

즉, 카메라 이미지에서 왼쪽 차선은 보통 오른쪽 아래에서 왼쪽 위로 올라가는 음의 기울기, 오른쪽 차선은 왼쪽 아래에서 오른쪽 위로 올라가는 양의 기울기라는 전제를 사용합니다.

### 대표 차선 계산 방식

왼쪽/오른쪽 후보 선분이 여러 개 있을 수 있으므로, 각각 대표 직선을 하나씩 만듭니다.

각 차선 후보에 대해 다음 값을 평균냅니다.

- 선분 양끝점의 x 좌표 합
- 선분 양끝점의 y 좌표 합
- 각 선분의 기울기

평균점 `(x_avg, y_avg)`와 평균 기울기 `m`을 구한 뒤 y절편을 계산합니다.

```text
b = y_avg - m * x_avg
```

대표 직선은 다음 형태입니다.

```text
y = m*x + b
```

이 대표 직선을 ROI의 위쪽 `y=0`과 아래쪽 `y=ROI_HEIGHT`에 맞춰 파란색 선으로 그립니다.

### 차선 위치와 조향각 계산

차선 위치는 ROI 내부 기준 수평선 `L_ROW=40`과 대표 차선 직선이 만나는 x 좌표입니다.

```text
x_left = int((L_ROW - b_left) / m_left)
x_right = int((L_ROW - b_right) / m_right)
```

한쪽 차선을 찾지 못한 경우에는 이전 값을 쓰거나, 반대쪽 차선에서 380px을 더하거나 빼서 추정합니다.

```text
if left missing and right exists:
    x_left = x_right - 380

if right missing and left exists:
    x_right = x_left + 380
```

두 차선의 중앙은 다음처럼 계산합니다.

```text
x_midpoint = (x_left + x_right) // 2
```

조향각은 차선 중앙과 화면 중앙의 차이를 그대로 사용합니다.

```text
new_angle = (x_midpoint - View_Center) * 1.0
```

차선 중앙이 화면 중심보다 오른쪽이면 양수, 왼쪽이면 음수 조향각이 됩니다. 속도는 `Fix_Speed=12`로 고정됩니다.

### 모터 명령 발행

모터 명령은 `Float32MultiArray`로 발행됩니다.

```text
motor_msg.data = [float(angle), float(speed)]
```

차선을 찾은 경우:

```text
drive(new_angle, 12)
```

차선을 찾지 못한 경우:

```text
drive(previous_angle, 12)
```

즉, 차선을 못 찾는 순간에도 바로 멈추지 않고 마지막 조향각과 속도를 유지합니다.

### 디버그 화면

`hough_drive.py`는 OpenCV 창으로 여러 중간 결과를 보여줍니다.

- `Lane Detection Canny Image`: ROI에 Canny edge를 적용한 이미지
- `Lanes positions`: 원본 이미지에 ROI 차선 검출 결과를 덮어쓴 이미지

`Lanes positions`에는 다음 요소가 표시됩니다.

- 기준 수평선 `L_ROW`
- 왼쪽 차선 교점
- 오른쪽 차선 교점
- 좌우 차선의 중점
- 화면 중앙점

### launch 파일 구조

`hough_drive.launch.py`는 카메라 launch만 include합니다.

```text
xycar_cam/launch/xycar_cam.launch.py
```

모터 launch include 코드는 있지만 현재 주석 처리되어 있습니다.

```text
# xycar_motor/launch/xycar_motor.launch.py
```

그 뒤 `my_hough`의 `hough_drive` 노드를 실행합니다. 실제 차량을 움직이려면 `xycar_motor` 토픽을 받아 처리하는 모터 드라이버 또는 bridge가 별도로 실행되어 있어야 합니다.

### `python_codes/` 실습 자료

`my_hough/my_hough/python_codes/` 폴더는 ROS 노드가 아니라 OpenCV 알고리즘을 독립적으로 실험하기 위한 자료입니다.

#### `my_hough.py`

정지 이미지 `line_pic1.png`를 읽어 차선 검출 과정을 단계별로 보여주는 코드입니다.

주요 특징:

- 원본 이미지 전체에서 먼저 Hough line을 찾아봅니다.
- 이후 ROI를 잘라 다시 Hough line을 찾습니다.
- 수평에 가까운 선을 기울기로 제거합니다.
- 왼쪽/오른쪽 차선 후보를 분류합니다.
- 대표 직선을 만들고 기준 수평선과의 교점을 계산합니다.
- 좌우 차선 중점과 화면 중심의 차이를 출력합니다.
- 각 단계마다 `cv2.waitKey()`로 멈춰 사람이 확인할 수 있게 되어 있습니다.

이 파일은 `hough_drive.py`의 실시간 주행 로직으로 가기 전에, 정지 이미지에서 Hough 차선 검출 원리를 이해하는 용도입니다.

#### `my_hough2.py`

정지 이미지 `line_pic2.png`를 읽고, polygon mask와 `numpy.polyfit`을 이용해 차선을 추정하는 대안 실험 코드입니다.

주요 특징:

- Canny threshold가 `50, 150`입니다.
- ROI를 직사각형 crop이 아니라 다각형 mask로 만듭니다.
- 왼쪽/오른쪽 선분을 기울기 기준으로 나눕니다.
- 각 차선 후보 선분의 점들을 모아 `np.polyfit(y_coords, x_coords, 1)`로 x 좌표를 y의 함수로 근사합니다.
- 외부에서 기준 수평선 `lane_row`를 인자로 넘깁니다.

`hough_drive.py`와 비교하면 구조가 더 함수형이고, 대표 직선 계산에 평균 기울기 대신 polynomial fitting을 씁니다.

#### `hough_find.py`

동영상 파일 `xycar_track1.mp4`를 반복 재생하며 차선을 검출하는 코드입니다.

주요 특징:

- `cv2.VideoCapture("xycar_track1.mp4")`로 영상을 읽습니다.
- 영상이 끝나면 frame 위치를 0으로 되돌려 반복 재생합니다.
- ROI는 `280:400`, 기준 수평선은 `L_ROW=50`입니다.
- `hough_drive.py`와 거의 같은 방식으로 Canny, Hough, 좌우 차선 분류, 대표 직선 계산을 수행합니다.
- 사다리꼴 mask와 dilation 코드는 들어 있지만 현재 주석 처리되어 있습니다.
- `TOLERANCE=50`을 두고 이전 차선 위치와 비교해 튀는 값을 완화하려는 의도가 보입니다.

이 파일은 실시간 ROS 카메라가 없어도 녹화 영상으로 차선 검출 알고리즘을 반복 실험하는 용도입니다.

#### 이미지/영상 자료

| 파일 | 역할 |
| --- | --- |
| `line_pic1.png` ~ `line_pic7.png` | 640x480 차선 정지 이미지 실험 자료 |
| `road_video1.mp4` | 도로 영상 실험 자료 |
| `road_video2.mp4` | 도로 영상 실험 자료 |
| `xycar_track1.mp4` | Xycar 트랙 영상 실험 자료 |

### 주의점

- OpenCV 창을 띄우므로 그래픽 환경이 필요합니다.
- `/image_raw`가 없으면 카메라 대기 상태에 머뭅니다.
- `cam_exposure(100)`이 실행되어 `/dev/videoCAM`의 노출 설정을 바꿉니다.
- `hough_drive.launch.py`는 모터 드라이버 launch를 실행하지 않습니다. 실제 구동에는 `xycar_motor` 토픽을 처리할 노드가 따로 필요합니다.
- 차선을 찾지 못했을 때 마지막 조향각/속도를 유지하므로, 실차 환경에서는 안전 조건을 별도로 넣는 것이 좋습니다.
- `package.xml`에는 런타임 의존성이 거의 적혀 있지 않지만, 코드상 `rclpy`, `sensor_msgs`, `std_msgs`, `cv_bridge`, `numpy`, `opencv-python` 등이 필요합니다.

## 패키지 간 관계

`src/study/`의 패키지들은 서로를 직접 import하지 않습니다. 대신 공통 토픽을 통해 느슨하게 연결됩니다.

### 센서 확인 계열

다음 패키지는 각각 센서 입력을 확인하는 성격입니다.

- `my_cam`: 카메라 이미지 확인
- `my_lidar`: LiDAR 거리 확인
- `my_ultra`: 초음파 배열 확인
- `my_imu`: IMU 자세 확인

이들은 대체로 “하드웨어 드라이버 launch include + 학습용 subscriber node” 형태입니다.

### 제어 계열

다음 패키지는 `xycar_motor`에 명령을 발행합니다.

- `my_motor`: 고정 직진 명령 발행
- `my_hough`: 차선 검출 결과를 조향각으로 바꾸어 명령 발행

두 패키지를 동시에 실행하면 같은 `xycar_motor` 토픽에 서로 다른 명령을 발행할 수 있으므로, 실제 구동 환경에서는 둘 중 하나만 실행하는 것이 안전합니다.

### 차선 주행 계열

`my_hough`는 `my_cam`의 개념을 확장한 패키지로 볼 수 있습니다.

- `my_cam`은 `/image_raw`를 받아 전처리 결과를 보여줍니다.
- `my_hough`는 `/image_raw`를 받아 차선을 검출하고, 검출 결과를 모터 명령으로 연결합니다.

즉, 학습 흐름상 `my_cam`에서 영상처리 기초를 확인한 뒤 `my_hough`에서 차선 주행으로 넘어가는 구조입니다.

## 실행 예시 모음

작업공간 루트에서 빌드 후 setup 파일을 source합니다.

```bash
colcon build --packages-select my_cam my_lidar my_ultra my_imu my_motor my_hough
source install/setup.bash
```

개별 실행 예시는 다음과 같습니다.

```bash
ros2 launch my_cam edge_cam.launch.py
ros2 launch my_cam exposure_cam.launch.py

ros2 launch my_lidar lidar_scan.launch.py
ros2 launch my_lidar lidar_viewer.launch.py

ros2 launch my_ultra ultra_scan.launch.py
ros2 launch my_imu roll_pitch_yaw.launch.py

ros2 launch my_motor go.launch.py
ros2 launch my_hough hough_drive.launch.py
```

## 코드 기준 개선 포인트

현재 코드는 학습용으로 이해하기 쉽지만, 실차 주행이나 재사용성을 높이려면 다음 부분을 정리할 수 있습니다.

| 위치 | 개선 포인트 |
| --- | --- |
| 여러 패키지의 `package.xml` | 코드에서 실제 사용하는 `sensor_msgs`, `std_msgs`, `cv_bridge`, `tf_transformations` 등 의존성 추가 |
| `my_motor/go.py` | launch의 `speed` 파라미터가 실제 반복 주행 속도에 반영되도록 수정 |
| `my_hough/hough_drive.py` | 차선 미검출 시 일정 시간 이후 감속/정지하는 안전 로직 추가 |
| `my_hough/hough_drive.py` | 생성자 안에서 무한 루프를 시작하는 구조를 timer 또는 명시적 spin 구조로 정리 |
| `my_hough/hough_drive.launch.py` | 실제 주행에 필요한 모터 드라이버 launch 포함 여부를 환경에 맞게 정리 |
| `my_cam/exposure_cam.py`, `my_hough/hough_drive.py` | `/dev/videoCAM` 장치명과 exposure 값을 launch parameter로 분리 |
| `my_lidar` | `scan`과 `/scan` 토픽 표기를 통일 |
| 모든 OpenCV GUI 노드 | headless 환경에서 실행할 수 있도록 display 옵션을 parameter로 분리 |

## 최종 요약

`src/study/`는 Xycar 기반 자율주행 실습을 위한 예제 패키지 모음입니다. 센서별로 데이터를 읽고 눈으로 확인하는 패키지들이 먼저 있고, 그 위에 모터 명령 발행과 Hough Transform 기반 차선 주행 예제가 얹혀 있습니다.

학습 순서로 보면 다음 흐름이 자연스럽습니다.

1. `my_cam`으로 카메라 이미지와 Canny edge를 확인합니다.
2. `my_lidar`, `my_ultra`, `my_imu`로 각 센서 토픽이 정상인지 확인합니다.
3. `my_motor`로 `xycar_motor` 명령 형식을 확인합니다.
4. `my_hough/python_codes`로 정지 이미지/동영상에서 차선 검출 알고리즘을 실험합니다.
5. `my_hough/hough_drive.py`로 실시간 카메라 차선 검출과 모터 명령 발행을 연결합니다.
