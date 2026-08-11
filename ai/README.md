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
재생성한다. 학습은 내부 ext4의 실제 `datasets/teleop` directory만 사용한다.
Windows `/mnt/c`를 가리키는 symlink는 최초 Windows sync에서 안전하게 실제
directory로 전환해 학습 시 WSL filesystem 경계를 넘는 image I/O를 제거한다.

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

### 기존 Windows C: snapshot 미러 (선택 사항)

새 기본 workflow는 아래의 T7 direct pull이다. 기존 C: snapshot을 원본으로 계속
사용해야 할 때만 MGW root에서 Windows dataset을 WSL 작업본으로 가져온다. 기본
source는 `/mnt/c/Users/gunwoo/Desktop/teleop`, destination은
`ai/datasets/teleop`이다. 완료된 gamepad session 중 metadata의
`max_forward_speed >= 20`인 session만 선택하며 현재 기준 11 sessions,
4,614 samples다. metadata가 없거나 비어 있거나 손상된 session은 warning과 함께
미완료로 간주해 복사하지 않는다.

최초 한 번은 Windows source를 가리키는 symlink를 ext4 directory로 전환한다.
첫 명령은 dry-run이고 두 번째 명령만 symlink를 제거하고 실제 파일을 복사한다.
Windows source는 변경하지 않는다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw
./scripts/ai/sync_dataset_windows.sh --init
./scripts/ai/sync_dataset_windows.sh --init --apply
```

이후에는 Windows의 추가·변경·삭제를 WSL에 증분 반영한다. Windows가 authoritative
source인 완전 미러이므로 destination-only 파일과 더 이상 조건에 맞지 않는
session도 `--apply`에서 삭제한다. 항상 dry-run을 먼저 확인하고 같은 크기·mtime의
내용 변경까지 비교할 때만 비용이 큰 `--checksum`을 사용한다.

```bash
./scripts/ai/sync_dataset_windows.sh
./scripts/ai/sync_dataset_windows.sh --apply
./scripts/ai/sync_dataset_windows.sh --checksum
./scripts/ai/sync_dataset_windows.sh --checksum --apply
```

source를 바꿀 때만 `XYCAR_AI_WINDOWS_DATASET_ROOT`를 절대 경로로 지정한다.
destination은 marker가 있는 ext4 directory여야 하며 system root, Windows mount,
잘못된 symlink와 source/destination 동일 경로는 거부한다.

### 현재 T7 `D:\teleop`에서 직접 pull

학습 환경은 WSL2로 고정하며 T7을 C:에 다시 복사하지 않는다. Windows에서
`D:\teleop`은 WSL의 `/mnt/d/teleop`으로 접근하고, 필요한 session만 내부 ext4
`datasets/teleop`으로 증분 복사한다. 실제 학습은 `/mnt/d`가 아니라 ext4
작업본에서 수행한다.

현재 확인된 T7 exFAT volume은 물리 disk는 Healthy지만 filesystem이
`Warning / Full Repair Needed`다. 중요한 데이터를 먼저 다른 곳에 백업하고,
Windows의 관리자 PowerShell에서 검사·복구한 뒤 `Healthy / OK`를 확인한다.
복구 명령은 파일 entry를 변경할 수 있으므로 백업 전에 실행하지 않는다.

```powershell
Get-Volume -DriveLetter D |
  Format-List DriveLetter,FileSystem,HealthStatus,OperationalStatus
chkdsk D:

# 위 검사에서 복구가 필요하고 백업을 마친 뒤에만 실행한다.
chkdsk D: /f
```

SSD를 연결한 뒤 새 WSL shell에서도 `/mnt/d`가 보이지 않을 때만 다음 명령으로
DrvFs mount를 만든다.

```bash
sudo mkdir -p /mnt/d
sudo mount -t drvfs D: /mnt/d
findmnt -T /mnt/d/teleop
```

direct 모드의 기본 source는 `/mnt/d/teleop`이다. drive letter가 바뀌면 예를 들어
`XYCAR_AI_SSD_TELEOP_ROOT=/mnt/e/teleop`으로 덮어쓴다. 인자 없는 direct 명령은
dry-run이고 `--apply`만 새 파일과 변경 파일을 복사한다. 완료된 gamepad session
중 `max_forward_speed >= 20`만 선택하며 WSL-only 파일과 session은 보존한다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw
./scripts/ai/pull_dataset_ssd.sh --direct
./scripts/ai/pull_dataset_ssd.sh --direct --apply
./scripts/ai/pull_dataset_ssd.sh --direct --checksum
./scripts/ai/pull_dataset_ssd.sh --direct --checksum --apply
```

SSD에서 삭제된 관리 대상 session과 session 내부 파일까지 반영할 때만 명시적인
mirror를 사용한다. mirror도 첫 명령은 dry-run이다. 속도 20 미만 session,
dataset marker와 root의 비관리 파일은 삭제하지 않는다.

```bash
./scripts/ai/pull_dataset_ssd.sh --direct --mirror
./scripts/ai/pull_dataset_ssd.sh --direct --mirror --apply
```

### 차량과 marker 기반 외장 SSD 공유

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

차량 sync와 marker 기반 SSD publish/pull은 `_recording_*`, rsync partial과 partial 파일을
제외하고 새 파일과 변경 파일만 복사한다. 이 두 경로는 destination-only 파일을
보존하며 `--delete`를 사용하지 않는다. SSD marker가 없거나 경로가 mount된 별도
filesystem이 아니면 작업을 거부한다. Windows→WSL 명령만 앞 절의 완전 미러
계약을 사용한다. `D:\teleop` direct 모드는 marker 기반 공유 동작을 변경하지
않으며 삭제는 `--direct --mirror --apply`에서만 수행한다.

## Front-camera policy 학습

`xycar-train`은 640x480 camera frame 한 장에서 연속 motor command를 정수
class로 변환한 angle/speed logits를 함께 학습한다. backbone은 ImageNet-21k
pretrain 뒤 ImageNet-1k로 fine-tuning된 timm ViT-tiny 또는 ViT-small이며,
pretrained weight를 받을 수 없으면 random initialization으로 대체하지 않고
학습을 중단한다.

기존 YAML은 전체 frame을 bicubic 224x224로 resize한다. 새 road-warp YAML은 원본
RGB frame의 도로 사다리꼴을 bird's-eye-view로 먼저 perspective warp한 뒤
224x224로 resize한다. 두 방식 모두 timm model data config의 mean/std로
normalize한다. warp 학습의 처리 순서는 `RGB → road warp → horizontal flip →
resize → color jitter → normalize`다. flip된 frame은 angle을 반전하고 speed는
유지한다. validation/test에도 같은 road warp를 적용하지만 flip과 color jitter는
적용하지 않는다. 두 출력은 각각 201 class이고 계약은 다음과 같다.

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

ViT-small 설정은 `config/front_cam_policy_train_small.yaml`이다. 완료된 gamepad
session 중 `max_forward_speed >= 20`을 허용하고 별도
`front_cam_policy_split_min_speed20.yaml`을 사용한다. 현재 snapshot에서는 같은
11 sessions과 3,227/685/702 split이지만 이후 조건에 맞는 session이 추가되면
split manifest에 배치하기 전까지 validation이 실패한다. small은 horizontal flip
`0.5`, batch 128, AMP, 20 epochs이며 run name은
`vit_small_hflip_p05_seed20260810`이다.

### Road warp YAML과 오프라인 튜닝 GUI

warp parameter는 `config/front_cam_policy_preprocess.yaml`에 저장한다. 모든 source
좌표는 원본 camera frame에 대한 0~1 정규화 좌표다. `top_*`/`bottom_*`은 원본에서
도로로 사용할 사다리꼴, `bev_width`/`bev_height`는 warp 출력 크기,
`dst_left_x`/`dst_right_x`는 출력에서 도로 좌우 경계다.

GUI는 ROS, camera device와 motor를 열지 않고 WSL ext4 dataset의 저장된 image만
읽는다. 왼쪽에서 원본 ROI와 warped 결과를 확인하고 slider를 움직인다. 변경은
preview에만 적용되며 **Save YAML** 버튼이나 `S`를 눌러야 파일에 기록된다.
Reset/`R`은 마지막 저장값으로 되돌리고 화살표 또는 `P`/`N`으로 sample을 바꾼다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw/ai
/home/xytron/.local/bin/uv run --locked xycar-warp-tuner \
  --config config/front_cam_policy_preprocess.yaml \
  --dataset-root datasets/teleop
```

특정 image 한 장으로 바로 열 수도 있다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw/ai
/home/xytron/.local/bin/uv run --locked xycar-warp-tuner \
  --config config/front_cam_policy_preprocess.yaml \
  --image datasets/teleop/<session-id>/Images/<image-file>
```

warp parameter를 저장한 뒤에는 학습 중 바꾸지 않는다. checkpoint에는 parameter
전체와 SHA-256이 들어가며 YAML이 달라진 상태에서는 resume을 거부한다. warp
checkpoint를 export하면 같은 parameter가 artifact manifest에 들어가고 차량
runtime도 동일한 perspective warp를 적용한다. 기존 full-frame checkpoint와
artifact 계약은 그대로 지원한다.

warp를 사용하는 ViT-small 설정은
`config/front_cam_policy_train_small_warp.yaml`이고 run name은
`vit_small_warp_hflip_p05_seed20260811`이다. GUI 저장 후 아래 명령으로 dataset과
warp YAML을 함께 검증하고 학습한다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw/ai
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_small_warp.yaml \
  --validate-only
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_small_warp.yaml
```

동일 warp run을 재개할 때는 저장된 YAML을 변경하지 않은 상태에서 실행한다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw/ai
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_small_warp.yaml \
  --resume artifacts/runs/front_cam_policy/vit_small_warp_hflip_p05_seed20260811/last.pt
```

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

ViT-small은 WSL dataset mirror가 최신인지 확인한 뒤 다음 두 명령을 순서대로
실행한다.

```bash
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_small.yaml \
  --validate-only

/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_small.yaml
```

중단된 run은 해당 directory의 `last.pt`에서 이어 간다. model/data/labelling
계약과 split이 현재 config와 다르면 resume를 거부한다.

```bash
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train.yaml \
  --resume artifacts/runs/front_cam_policy/<run-id>/last.pt
```

ViT-small resume은 config와 checkpoint를 함께 명시한다.

```bash
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_small.yaml \
  --resume artifacts/runs/front_cam_policy/vit_small_hflip_p05_seed20260810/last.pt
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
