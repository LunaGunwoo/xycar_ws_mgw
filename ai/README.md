# xycar-ai

현재 `4090-LAPTOP`에서 source 작성, RTX 4090 Laptop GPU 학습·평가와 artifact
생성을 함께 수행하는 독립 `uv` project다.

## 환경 계약

- Python: 3.12
- uv: 0.11.24
- dataset: `datasets/teleop/`
- training output: `artifacts/`
- 배포 후보: `artifacts/models/<artifact-id>/`

`.venv`, dataset과 artifact는 Git에 commit하지 않는다. `.venv`는 lockfile에서
재생성한다. `datasets/teleop`은 내부 ext4 작업본 directory 또는 현재 Laptop의
dataset을 가리키는 local symlink로 둘 수 있다. `/mnt/c`를 직접 읽으면 최초
복사는 생략할 수 있지만 WSL filesystem 경계 때문에 image I/O가 더 느릴 수 있다.

MGW root에서 다음 명령으로 환경을 준비한다. pin된 uv가 없거나 버전이 다르면
0.11.24를 설치하고 Python 3.12, locked environment, CUDA와 GPU 이름까지
검증한다.

```bash
./scripts/ai/bootstrap_env.sh
```

환경 준비 후 AI 명령은 project directory에서 실행한다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw/ai
/home/xytron/.local/bin/uv lock --check
/home/xytron/.local/bin/uv sync --locked --managed-python
```

## Dataset 동기화와 외장 SSD 공유

차량에서 현재 Laptop의 내부 ext4 작업본으로 가져올 때는 MGW root에서 실행한다.
기본 동작은 dry-run이고, `--apply`만 파일을 복사한다. 크기·mtime 대신 내용을
정밀 비교해야 할 때만 비용이 큰 `--checksum`을 함께 쓴다.

개발 Laptop과 차량은 Tailscale이 설치된 같은 tailnet에 있어야 하며 차량의
MagicDNS 이름은 `xycar`다. sync와 model deploy script의 기본 SSH 대상은
`xytron@xycar`이고 직접 IP로 자동 fallback하지 않는다. 먼저
`getent hosts xycar`와 `ssh xytron@xycar`로 연결을 확인한다. WSL은 Windows
Tailscale을 통해 MagicDNS를 사용할 수 있어 Linux CLI가 없어도 된다.

```bash
./scripts/ai/sync_dataset.sh
./scripts/ai/sync_dataset.sh --apply
./scripts/ai/sync_dataset.sh --checksum
```

외장 SSD는 Linux 또는 WSL에서 mount한 뒤 공용 root를 절대 경로로 지정한다.
Windows 사용자는 PowerShell 복사 대신 WSL에서 실행한다. 최초 한 번만 한 명의
publisher가 `--init --apply`로 marker와 `teleop/`을 만들고, 이후에도 publisher
한 명만 publish한다. 다른 팀원은 pull만 사용한다.

```bash
export XYCAR_AI_SHARED_DATASET_ROOT=/mnt/e/xycar-ai-dataset

./scripts/ai/publish_dataset_ssd.sh --init
./scripts/ai/publish_dataset_ssd.sh --init --apply
./scripts/ai/publish_dataset_ssd.sh
./scripts/ai/publish_dataset_ssd.sh --apply

./scripts/ai/pull_dataset_ssd.sh
./scripts/ai/pull_dataset_ssd.sh --apply
```

모든 sync는 `_recording_*`, rsync partial과 partial 파일을 제외하고 새 파일과
변경 파일만 복사한다. destination-only 파일은 보존하며 `--delete`를 사용하지
않는다. SSD marker가 없거나 경로가 mount된 별도 filesystem이 아니면 작업을
거부한다.

현재 Desktop에 모아 둔 snapshot을 최초 작업본으로 가져올 때도 먼저 dry-run을
확인한 다음 같은 filter와 no-delete 방식으로 복사한다.

```bash
rsync -rtvn --modify-window=1 --omit-dir-times \
  --filter='merge scripts/ai/dataset-rsync-filter.rules' \
  /mnt/c/Users/gunwoo/Desktop/teleop/ ai/datasets/teleop/
rsync -rtv --modify-window=1 --omit-dir-times --partial \
  --partial-dir=.rsync-partial \
  --filter='merge scripts/ai/dataset-rsync-filter.rules' \
  /mnt/c/Users/gunwoo/Desktop/teleop/ ai/datasets/teleop/
```

## Front-camera policy 학습

`xycar-train`은 640x480 camera frame 한 장에서 연속 motor command를 정수
class로 변환한 angle/speed logits를 함께 학습한다. backbone은 ImageNet-21k
pretrain 뒤 ImageNet-1k로 fine-tuning된
`vit_tiny_patch16_224.augreg_in21k_ft_in1k`이며, pretrained weight를 받을 수
없으면 random initialization으로 대체하지 않고 학습을 중단한다.

입력은 전체 frame을 bicubic 224x224로 resize하고, timm model data config의
mean/std로 normalize한다. 학습 시 color jitter와 resize 전 horizontal flip을
사용한다. flip된 frame은 angle을 반전하고 speed는 유지한다. validation/test에는
두 augmentation을 모두 적용하지 않는다. 두 출력은 각각 201 class이고 계약은
다음과 같다.

```text
class_id = int(round(clamp(command, -100, 100))) + 100
command  = class_id - 100

horizontal flip:
  angle_raw      = -angle_raw
  angle          = -angle
  angle_class_id = 200 - angle_class_id
  speed          = unchanged
```

기본 candidate 설정은 `config/front_cam_policy_train.yaml`이며 flip 확률은
`0.5`, run name은 `hflip_p05_seed20260810`이다. A/B baseline인
`config/front_cam_policy_train_no_flip.yaml`은 flip 확률 `0.0`, run name
`baseline_seed20260810`이고 나머지 설정은 동일하다. 고정 session split은
`config/front_cam_policy_split.yaml`이다. trainer는 완료된 gamepad session 중
`metadata.yaml`의 `gamepad.max_forward_speed: 25.0`과 일치하는 11개 session만
허용한다. split은 session 간 image가 섞이지 않는 7/2/2이고 현재 기준 sample은
train 3,227, validation 685, test 702개다. 새 session을 자동 포함하지 않으므로
dataset snapshot을 바꾸려면 split YAML을 명시적으로 검토한다.
모델·data path·augmentation·optimizer·scheduler·loss·seed·output은 모두 학습
YAML에 기록되며 CLI override는 제공하지 않는다.

dataset 전송 뒤 먼저 source와 manifest 계약을 검사한다. 이 명령은 model이나
pretrained weight를 load하지 않는다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw/ai
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train.yaml \
  --validate-only
```

검증이 성공하면 같은 locked 환경에서 baseline과 candidate를 각각 학습한다.

```bash
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_no_flip.yaml

/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train.yaml
```

중단된 run은 해당 directory의 `last.pt`에서 이어 간다. model/data/labelling
계약과 split이 현재 config와 다르면 resume를 거부한다.

```bash
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train.yaml \
  --resume artifacts/runs/front_cam_policy/<run-id>/last.pt
```

각 run은 다음 파일을 남긴다. checkpoint와 metric은 ignored artifact이며 Git에
넣지 않는다.

```text
artifacts/runs/front_cam_policy/<run-id>/
  resolved_config.yaml
  dataset_stats.json
  split.json
  metrics.csv
  best.pt
  last.pt
  test_metrics.json
  summary.json
```

best checkpoint 선택식은 `val_angle_mae + 0.25 * val_speed_mae`다. A/B winner도
두 `summary.json`의 `best_score`가 더 낮은 run으로 정하고 동률이면 baseline을
선택한다. test 결과는 winner를 validation으로 고른 뒤 최종 확인에만 사용한다.
metrics에는 train의 실제 flip 비율과 hard-left/left/near-zero/right/hard-right별
angle MAE와 within-10 accuracy가 포함된다. 실제 차량은 Ryzen 7 7730U CPU-only
환경이다. epoch 6 baseline의 synthetic memory-frame benchmark는 전처리 포함
steady-state p95 `11.802 ms`로 20 Hz의 50 ms budget 가능성을 확인했다. 실제
camera-to-motor latency와 주행 품질은 아직 실차에서 검증하지 않았다.

## Model artifact 계약

배포할 model directory는 최소한 다음 구조를 가진다.

```text
artifacts/models/<artifact-id>/
  model.ts
  manifest.yaml
  SHA256SUMS
```

`SHA256SUMS`에는 `manifest.yaml`과 모든 배포 파일의 상대 경로를 기록한다.
절대 경로, `..`, symlink는 허용하지 않는다. 학습 checkpoint를 fixed-shape
TorchScript tuple-output artifact로 변환할 때는 다음 명령을 사용한다. 기존
artifact ID는 덮어쓰지 않는다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw/ai
/home/xytron/.local/bin/uv run --locked xycar-export-policy \
  --checkpoint artifacts/runs/front_cam_policy/baseline_seed20260810/best.pt \
  --artifact-id front-cam-policy-baseline-e6-20260810
```

exporter는 checkpoint model state를 strict load하고 eager/trace/reload 결과와 두
`[1,201]` 출력을 확인한 뒤 `model.ts`, `manifest.yaml`, `SHA256SUMS`를 atomic하게
생성한다. manifest에는 checkpoint/source/dataset, RGB input, timm preprocessing,
label decode, CPU thread와 warm-up 계약이 포함된다. 차량 배포와 실행 방법은
`src/xycar_ai_drive/README.md`를 따른다.
