# Xycar AI Training Harness

이 directory는 MGW 저장소의 독립 Python 학습 project다.

## Source ownership

- source 작성과 학습의 기준은 4090 Laptop
  `/home/xytron/xycar_ws/apps/xycar_ws_mgw/ai`다.
- 이 위치에서 source, dependency, 학습·평가와 artifact 생성을 관리한다.

## Python과 uv

- Python 3.12와 `uv`만 사용한다.
- dependency 변경은 이 project에서 `uv add`, `uv remove`, `uv lock`으로 한다.
- 실행과 검증은 pin된 uv 0.11.24의 `uv lock --check`,
  `uv sync --locked --managed-python`, `uv run --locked ...`를 사용한다.
- `pip`, `pipx`, conda, Poetry, 수동 `python -m venv`, system Python package
  설치와 raw `python`/`pytest` 실행을 금지한다.
- `.venv`는 전송하거나 Git에 넣지 않는다.

## Data와 artifact

- 차량에서 종료된 `*_session*`은 모두 학습 후보 raw data다.
- `*_incomplete*`은 보존하되 명시적인 검토 없이 학습에 포함하지 않는다.
- 활성 `_recording_*`은 동기화하지 않는다.
- 학습 작업본은 `datasets/teleop`, 외장 SSD 공유본은 marker가 있는
  `$XYCAR_AI_SHARED_DATASET_ROOT/teleop`이다.
- 차량 dataset sync와 model deploy의 기본 SSH 대상은 Tailscale MagicDNS의
  `xytron@xycar`다. 직접 IP fallback을 기본 workflow로 사용하지 않는다.
- 한 명만 SSD에 publish하고 다른 팀원은 pull만 한다. 둘 다 no-delete다.
- `datasets/`, `artifacts/`, model, checkpoint는 Git에 넣지 않는다.
- 배포 후보 model은 `artifacts/models/<artifact-id>/` 아래에 두고
  `manifest.yaml`과 `SHA256SUMS`를 포함한다.
