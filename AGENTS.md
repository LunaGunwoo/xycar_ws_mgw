# Xycar MGW Repository Harness

이 저장소는 실차 Xycar의 유일한 활성 source 저장소다. 작업을 시작할 때 이
파일과 작업 대상 package의 `README.md`를 먼저 읽는다.

## 세 환경의 역할

| 환경 | 기준 경로 | 역할 |
| --- | --- | --- |
| 개발 PC | `/home/xytron/xycar_ws/apps/xycar_ws_mgw` | source 작성, 정적 검증, commit, push |
| 차량 PC | `xytron@10.42.0.1:/home/xytron/xycar_ws_mgw` | pull, build, 승인된 실행, 데이터 수집, inference |
| 학습 PC | `gunwoo@5090-DESKTOP:/home/gunwoo/Documents/xycar-ai` | 전달된 코드로 학습·평가, artifact 생성 |

tracked source의 원본은 항상 개발 PC의 MGW checkout이다. 차량에서는 tracked
파일을 수정하거나 commit·push하지 않는다. 차량 변경이 필요하면 개발 PC에서
수정·검증·push한 뒤 차량에서 다음 순서로 반영한다.

```bash
cd /home/xytron/xycar_ws_mgw
git status --short --branch
git pull --ff-only
```

차량 checkout이 dirty하거나 branch·remote가 예상과 다르면 stash, reset,
checkout으로 자동 정리하지 말고 개발 PC에서 원인을 확인한다. 차량에서 생성할
수 있는 것은 `build/`, `install/`, `log/`, dataset과 ignored model artifact 같은
runtime 파일뿐이다.

학습 코드의 원본은 이 저장소의 `ai/`다. 개발 PC에서만 의존성 및 lockfile을
변경하고 `scripts/ai/sync_training_code.sh`로 그 내용을 5090 학습 root에
flatten한다. 5090에서 source를 영구 수정하지 않는다.

## 저장소와 실차 안전

- `src/yolo_ros`와 `src/xycar_device/xycar_lidar/YDLidar-SDK`는 별도 소유의
  중첩 저장소다. 사용자가 명시적으로 범위에 넣기 전에는 읽기만 한다.
- 모터 publisher, camera·LiDAR·IMU·초음파 장치, serial device,
  `v4l2-ctl`, `etc/gui-shell/*.sh`는 매 실행 직전에 사용자 승인을 받는다.
- 센서 미수신, stale data, 처리 예외와 종료의 기본 동작은 정지다.
- ROS callback에 blocking loop나 `time.sleep()`를 새로 넣지 않는다.
- ROS 2 Humble의 system Python을 변경하거나 system `pip install`을 하지 않는다.
- 실행 파일·launch·CMake 설치 명령이 바뀌면 같은 변경에서 package
  `README.md`를 갱신한다.

## AI 데이터와 모델

- `ai/`는 Python 3.12와 `uv`만 사용하는 독립 학습 프로젝트다.
- `.venv`, dataset, checkpoint와 model은 Git 또는 학습 코드 전송에 포함하지
  않는다.
- 5090에서는 `/home/gunwoo/.local/bin/uv`를 사용해 `uv lock --check`,
  `uv sync --locked`, `uv run --locked ...`만 실행한다.
- 차량 dataset의 활성 `_recording_*`은 전송하지 않는다. 종료된
  `*_session*`과 보존용 `*_incomplete*`은 전송하되 incomplete 데이터는
  학습에 자동 포함하지 않는다.
- 배포 model은 versioned directory, `manifest.yaml`, `SHA256SUMS`로 관리하고
  Git에 넣지 않는다.

## 검증과 commit

- 변경 package만 `colcon build --symlink-install --packages-select ...`로
  빌드한다.
- AI Python 명령은 `uv run --locked ...`로 실행한다.
- hardware 없이 할 수 없는 검증은 성공으로 보고하지 않는다.
- commit은 사용자가 명시적으로 요청한 경우에만 개발 PC에서 만든다.
- 상위 하네스 checkout이 있으면 작업 전후
  `python3 scripts/xycar_workspace_guard.py`를 실행한다.
