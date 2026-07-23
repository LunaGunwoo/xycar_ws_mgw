# xycar-ai

5090 GPU 학습을 위한 독립 `uv` project scaffold다. source의 원본은 개발 PC의
MGW `ai/`이며, 이 directory의 내용만 5090
`/home/gunwoo/Documents/xycar-ai`에 flatten된다.

## 환경 계약

- Python: 3.12
- uv: 0.11.24
- 5090 uv: `/home/gunwoo/.local/bin/uv`
- dataset: `datasets/teleop/`
- training output: `artifacts/`
- 배포 후보: `artifacts/models/<artifact-id>/`

`.venv`, dataset과 artifact는 전송하거나 Git에 commit하지 않는다. 각 환경의
`.venv`는 lockfile에서 재생성한다.

개발 PC에서는 MGW root에서 다음 명령으로 환경을 준비한다.

```bash
./scripts/ai/bootstrap_env.sh
```

5090에서는 다음 명령만 사용한다.

```bash
cd /home/gunwoo/Documents/xycar-ai
/home/gunwoo/.local/bin/uv lock --check
/home/gunwoo/.local/bin/uv sync --locked --managed-python
/home/gunwoo/.local/bin/uv run --locked python -c \
  'import sys; print(sys.version)'
```

의존성 추가·삭제와 `uv lock`은 5090에서 하지 않는다. 먼저 개발 PC의 MGW
`ai/`에서 변경하고 code sync를 다시 실행한다.

## 전송

개발 PC의 MGW root에서 실행한다.

```bash
./scripts/ai/sync_training_code.sh --dry-run --init
./scripts/ai/sync_training_code.sh --init
./scripts/ai/sync_dataset.sh --dry-run
./scripts/ai/sync_dataset.sh
```

code sync는 source `.venv`를 보내지 않고 destination의 `.venv`, `datasets/`,
`artifacts/`를 보존한다. dataset sync는 개발 PC 디스크에 dataset을 저장하지
않고 일회성 SSH reverse tunnel을 통해 차량에서 5090으로 전송한다.

## Model artifact 계약

배포할 model directory는 최소한 다음 구조를 가진다.

```text
artifacts/models/<artifact-id>/
  manifest.yaml
  SHA256SUMS
  <model files>
```

`SHA256SUMS`에는 `manifest.yaml`과 모든 배포 파일의 상대 경로를 기록한다.
절대 경로, `..`, symlink는 허용하지 않는다.
