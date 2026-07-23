# Xycar AI Training Harness

이 directory는 MGW 저장소에서는 `ai/`이고, 5090으로 전송되면
`/home/gunwoo/Documents/xycar-ai`의 root가 된다.

## Source ownership

- source의 원본은 개발 PC
  `/home/xytron/xycar_ws/apps/xycar_ws_mgw/ai`다.
- 5090은 학습·평가·artifact 생성 환경이다. 5090에서 source, `pyproject.toml`,
  `.python-version`, `uv.lock`을 영구 수정하지 않는다.
- 5090에서 발견한 수정 사항은 개발 PC의 MGW source에 반영한 뒤 다시
  동기화한다.

## Python과 uv

- Python 3.12와 `uv`만 사용한다.
- 개발 PC에서만 `uv add`, `uv remove`, `uv lock`으로 의존성과 lockfile을
  변경한다.
- 5090에서는 `/home/gunwoo/.local/bin/uv lock --check`,
  `/home/gunwoo/.local/bin/uv sync --locked --managed-python`,
  `/home/gunwoo/.local/bin/uv run --locked ...`만 사용한다.
- `pip`, `pipx`, conda, Poetry, 수동 `python -m venv`, system Python package
  설치와 raw `python`/`pytest` 실행을 금지한다.
- `.venv`는 전송하지 않는다. 각 host에서 `uv sync --locked`로 따로 만든다.

## Data와 artifact

- 차량에서 종료된 `*_session*`은 모두 학습 후보 raw data다.
- `*_incomplete*`은 보존하되 명시적인 검토 없이 학습에 포함하지 않는다.
- 활성 `_recording_*`은 동기화하지 않는다.
- `datasets/`, `artifacts/`, model, checkpoint는 Git에 넣지 않는다.
- 배포 후보 model은 `artifacts/models/<artifact-id>/` 아래에 두고
  `manifest.yaml`과 `SHA256SUMS`를 포함한다.
