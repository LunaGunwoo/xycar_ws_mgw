# xycar-ai

현재 `4090-LAPTOP`에서 source 작성, RTX 4090 Laptop GPU 학습·평가와 artifact
생성을 함께 수행하는 독립 `uv` project다.

## 환경 계약

- Python: 3.12
- uv: 0.11.24
- 새 dataset: `datasets/stateless_manual/`, `datasets/stateless_guided/`
  (`stateless_guided/generation_<N>/<collection-id>/`별 raw cohort)
- rollback dataset: `datasets/teleop/`
- training output: `artifacts/`
- 배포 후보: `artifacts/models/<artifact-id>/`

`.venv`, dataset과 artifact는 Git에 commit하지 않는다. `.venv`는 lockfile에서
재생성한다. 새 학습은 내부 ext4의 두 `datasets/stateless_*` directory만 사용한다.
기존 `datasets/teleop`은 rollback 실험에만 사용한다.
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

### Stateless manual 직결 LAN 미러

`stateless_manual` 대용량 전송은 Windows와 활성 Jetson을 LAN cable로 직접 연결한
전용 경로를 쓴다. 이 경로는 일반 Tailscale SSH와 섞거나 자동 fallback하지 않는다.
고정 계약은 Windows `Ethernet`의 `192.168.50.1/24`, Jetson `enP8p1s0`의
`192.168.50.2/24`이며 Jetson 유선 profile에는 gateway와 DNS를 넣지 않아 Wi-Fi
기본 경로를 유지한다.

최초 한 번 Windows 관리자 PowerShell에서 `Ethernet`의 기존 IPv4 주소를 정리하고
고정 주소를 만든다. 이 작업은 해당 Ethernet adapter의 기존 IPv4 설정을 바꾸므로
직결 cable에 사용하는 adapter인지 먼저 확인한다.

```powershell
$InterfaceAlias = 'Ethernet'
Get-NetIPAddress -InterfaceAlias $InterfaceAlias -AddressFamily IPv4 |
  Remove-NetIPAddress -Confirm:$false
Set-NetIPInterface -InterfaceAlias $InterfaceAlias -AddressFamily IPv4 -Dhcp Disabled
New-NetIPAddress -InterfaceAlias $InterfaceAlias `
  -IPAddress 192.168.50.1 -PrefixLength 24
Set-DnsClientServerAddress -InterfaceAlias $InterfaceAlias -ResetServerAddresses
```

Jetson에서는 다음 profile을 한 번 만든다. 기존 profile이 있으면 `add` 대신
`modify` branch가 실행된다. `sudo`는 이 최초 설정과 profile 활성화에만 필요하다.

```bash
ssh xytron@xycar-gpu
if nmcli -t -f NAME connection show | grep -Fxq xycar-direct-lan; then
  sudo nmcli connection modify xycar-direct-lan \
    connection.interface-name enP8p1s0 \
    ipv4.method manual ipv4.addresses 192.168.50.2/24 \
    ipv4.gateway '' ipv4.dns '' ipv4.never-default yes \
    ipv6.method disabled connection.autoconnect yes
else
  sudo nmcli connection add type ethernet ifname enP8p1s0 \
    con-name xycar-direct-lan \
    ipv4.method manual ipv4.addresses 192.168.50.2/24 \
    ipv4.never-default yes ipv6.method disabled \
    connection.autoconnect yes
fi
sudo nmcli connection up xycar-direct-lan
```

설정 뒤 WSL의 MGW root에서 다음 명령을 실행한다. 인자 없는 기본 명령이 곧 실제
미러 적용이다. 차량 source가 authoritative하므로 새 파일과 변경 파일을 복사하고,
차량에서 없어진 파일·session 및 WSL에만 있는 파일은 복사가 성공한 뒤
`--delete-delay`로 삭제한다. 차량 원본은 읽기만 한다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw
./scripts/ai/sync_stateless_manual_lan.sh
```

source는 `xytron@192.168.50.2:/home/xytron/xycar_data/stateless_manual`, destination은
내부 ext4 `ai/datasets/stateless_manual`로 고정한다. 실행마다 Windows link,
직결 route와 `192.168.50.2:22`, 검증된 `xycar-gpu` SSH host key, 원격 hostname과
source 절대 경로를 검사한다. 처음 대상이 없거나 비어 있으면
`.xycar-ai-local-dataset` marker를 만들며, 비어 있지 않은 unmarked 대상,
symlink와 ext4가 아닌 대상은 거부한다. 중복 실행도 lock으로 거부한다.

정상·`*_incomplete*`를 포함한 닫힌 session은 그대로 보존한다. 아직 쓰는 중인
`_recording_*`, `.rsync-partial`, `*.partial`, `*.part`, `*.tmp`는 전송하지 않는다.
빈 차량 source는 WSL 전체 삭제 사고를 막기 위해 거부하며 의도적인 빈 미러에만
`--allow-empty-source`를 쓴다. 비교만 하거나 내용 checksum까지 확인하는 명령은
다음과 같다.

```bash
./scripts/ai/sync_stateless_manual_lan.sh --dry-run
./scripts/ai/sync_stateless_manual_lan.sh --checksum
./scripts/ai/sync_stateless_manual_lan.sh --dry-run --checksum
./scripts/ai/sync_stateless_manual_lan.sh --allow-empty-source
./scripts/ai/sync_stateless_manual_lan.sh --help
```

JPEG에는 SSH/rsync 압축을 쓰지 않으며 진행률, 변경 항목, 통계와 Windows가 보고한
link speed를 출력한다. 현재 협상 속도 `100 Mbps full-duplex`에서는 1.2 GB 최초
복사가 대략 2분 안팎이다. split manifest는 자동 수정하지 않으므로 복사 후 완료
session을 검토하고 `manual/<session-id>` train/validation 배치를 수동 승인한다.

### 기존 Windows C: snapshot 미러 (선택 사항)

아래 T7·Windows 절은 기존 `datasets/teleop` rollback dataset을 유지할 때만
사용한다. 새 stateless 기본은 뒤의 차량 두-root 동기화 절차다. 기존 C: snapshot을
원본으로 계속 사용해야 할 때만 MGW root에서 Windows dataset을 WSL 작업본으로 가져온다. 기본
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
`D:\teleop`은 WSL의 `/mnt/d/teleop`으로 접근하고, 전체 또는 선별한 session을
내부 ext4 `datasets/teleop`으로 증분 복사한다. 실제 학습은 `/mnt/d`가 아니라
ext4 작업본에서 수행한다.

T7 exFAT volume은 사용 전에 `Healthy / OK`인지 확인한다. 다른 상태라면 중요한
데이터를 먼저 다른 곳에 백업하고 Windows의 관리자 PowerShell에서 검사·복구한다.
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
`XYCAR_AI_SSD_TELEOP_ROOT=/mnt/e/teleop`으로 덮어쓴다. SSD의 모든 session을
metadata 조건 없이 내부 ext4에 보관하려면 `--direct --all`을 사용한다. 기본은
dry-run이며 `--apply`만 새 파일과 변경 파일을 복사한다. `_recording_*`과 rsync
partial만 제외하고 WSL-only 파일은 보존한다. `--all`은 삭제를 수행하는
`--mirror`와 함께 사용할 수 없다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw
./scripts/ai/pull_dataset_ssd.sh --direct --all
./scripts/ai/pull_dataset_ssd.sh --direct --all --apply
./scripts/ai/pull_dataset_ssd.sh --direct --all --checksum
./scripts/ai/pull_dataset_ssd.sh --direct --all --checksum --apply
```

학습 조건에 맞는 session만 작업본에 넣고 싶을 때는 `--all` 없이 실행한다.
인자 없는 direct 명령은 dry-run이고 `--apply`만 복사한다. 완료된 gamepad session
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
MagicDNS 이름은 `xycar-gpu`다. sync와 model deploy script의 기본 SSH 대상은
`xytron@xycar-gpu`이고 직접 IP로 자동 fallback하지 않는다. 먼저
`getent hosts xycar-gpu`와 `ssh xytron@xycar-gpu`로 연결을 확인한다. WSL은 Windows
Tailscale을 통해 MagicDNS를 사용할 수 있어 Linux CLI가 없어도 된다.
기존 CPU rollback dataset을 명시적으로 가져올 때만 한 명령에
`XYCAR_AI_VEHICLE_SSH=xytron@xycar`를 지정한다.

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

일반 Tailscale 차량 sync와 marker 기반 SSD publish/pull은 `_recording_*`, rsync partial과 partial 파일을
제외하고 새 파일과 변경 파일만 복사한다. 이 두 경로는 destination-only 파일을
보존하며 `--delete`를 사용하지 않는다. SSD marker가 없거나 경로가 mount된 별도
filesystem이 아니면 작업을 거부한다. Windows→WSL 명령만 앞 절의 완전 미러
계약을 사용한다. `D:\teleop` direct 모드는 marker 기반 공유 동작을 변경하지
않으며 삭제는 `--direct --mirror --apply`에서만 수행한다. 별도
`sync_stateless_manual_lan.sh`만 차량 manual source의 exact mirror이므로 기본
실행에서 WSL destination-only 데이터를 삭제하며 이 일반 no-delete 계약을 바꾸지
않는다.

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
오른쪽의 **Image number (1-based)**에 전체 dataset 기준 사진 번호를 입력하고
**Go** 또는 Enter를 누르면 멀리 떨어진 다른 session의 image로 바로 이동한다.
`bev_width`와 `bev_height`는 slider 옆 입력칸에 `224`처럼 정확한 정수를
입력하고 Enter를 눌러 적용할 수도 있다. warp 출력은 이 크기로 생성된 뒤 ViT
입력 직전에 `224×224`로 resize된다.

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
`vit_small_warp_angle_mean5_hflip_p05_seed20260811`이다. train split의 angle
target은 같은 session 안에서 현재 frame 중심 5개 angle을 평균하며, session
경계에서는 첫·마지막 angle을 반복한다. 평균은 class 양자화 전에 적용한다.
validation/test angle과 모든 speed target은 원본 frame 값을 유지한다. GUI 저장 후
아래 명령으로 dataset과 warp YAML을 함께 검증하고 학습한다.

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
  --resume artifacts/runs/front_cam_policy/vit_small_warp_angle_mean5_hflip_p05_seed20260811/last.pt
```

### 기존 AR control token 재현 (rollback 전용)

이 절은 과거 결과 재현과 rollback 검증에만 사용한다. 새 stateless dataset과
curriculum에는 적용하지 않는다. AR 설정은 현재 warped image token 뒤에 과거 4 frame의 angle/speed token과 현재
angle/speed query를 다음 순서로 붙인다.

```text
[CLS, image patches,
 angle/speed(t-4), angle/speed(t-3), angle/speed(t-2), angle/speed(t-1),
 angle query, speed query]
```

train history는 같은 session의 정답만 사용한다. angle은 각 과거 frame에 계산된
centered 5-frame 평균 target이고 speed는 원본 target이다. 부족한 과거 위치는
session의 첫 target을 반복하며 horizontal flip은 현재 angle과 history angle을
모두 `200 - class_id`로 바꾼다. 현재 두 query에만 loss를 계산한다. 선택한 계약상
과거 smoothed angle에 현재·미래 raw angle이 포함될 수 있고 t=0 history에도 첫
target이 들어간다. validation/test와 runtime은 정답을 사용하지 않고 session마다
`(angle=0, speed=25)` 네 쌍으로 시작해 직전 argmax 예측을 순차적으로 넣는다.

두 출력은 201개 control value embedding weight를 공용 output projection으로
사용하고 task별 bias만 분리한다. `shared_type`은 여기에 angle/speed type embedding을
추가한다. loss는 기존 예선 및 stateless 모델과 같은 다음 식이다.

```text
CE_angle + 0.5 * CE_speed + 0.2 * (EMD_angle + 0.5 * EMD_speed)
```

EMD는 예측 softmax와 정답 one-hot의 누적분포 간 평균 절대 차이이므로 정답 class에서
멀리 떨어진 확률에 더 큰 손실을 준다. 두 설정을 같은 20-epoch cosine schedule로
만들고 5 epoch에서 probe를 멈춘 뒤 rollout validation score가 낮은 run만 `last.pt`로
재개한다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw/ai
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_small_warp_ar_shared.yaml \
  --validate-only
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_small_warp_ar_shared_type.yaml \
  --validate-only

/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_small_warp_ar_shared.yaml \
  --stop-after-epoch 5
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_small_warp_ar_shared_type.yaml \
  --stop-after-epoch 5

# <winner-config>과 <winner-run>은 5-epoch rollout validation 비교로 선택한다.
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/<winner-config>.yaml \
  --resume artifacts/runs/front_cam_policy/<winner-run>/last.pt
```

probe run에는 최종 test 대신 `probe_summary.json`을 기록한다. 승자를 총 20 epoch까지
resume한 뒤에만 best checkpoint를 prediction rollout 방식으로 test 평가한다.

### Stateless ViT-Small 증분 학습과 EMA sampling

새 기본 workflow는 기존 `datasets/teleop`과 checkpoint를 사용하지 않는다.
수동·사람 보정 session은 각각 다음 root에 물리적으로 분리한다. Base G0는 Manual
root만 읽고, G1부터 두 source를 함께 읽는다.

```text
datasets/stateless_manual  # control_mode=gamepad, 항상 generation 0
datasets/stateless_guided  # control_mode=guided_policy, curriculum.generation 필수
├── <legacy-direct-session>                         # 기존 G1 호환
└── generation_<N>/<collection-id>/<session-id>    # 새 분리 저장 구조
```

Guided loader는 기존 root 직속 session과 새
`generation_<N>/<collection-id>/<session-id>`를 함께 읽는다. 새 nested session의
source-qualified ID는
`guided/generation_<N>/<collection-id>/<session-id>`이며 directory generation과
`metadata.yaml`의 `curriculum.generation`이 다르면 학습을 거부한다.

새 steering cutover는
`config/front_cam_policy_train_stateless_normalized_v1.yaml`을 사용한다. 이
설정은 `required_steering_contract: normalized_percent_v1`로 metadata의 전체
`-100..100 -> -40..40` 계약과 topic을 검사하므로 계약이 없는 기존 session을
자동 제외한다. 원본 instantaneous angle을 그대로 쓰며
`train_angle_mean_window: 1`이라 시간축 조향 평균은 적용하지 않는다.

기존 `config/front_cam_policy_train_stateless_ema.yaml`은 pretrained
`vit_small_patch16_224`, `task_tokens`, history 0, 224x224 road warp와 raw instantaneous
angle을 사용하는 legacy/reference 설정이다. AR 학습 설정과 schema v2/v3 runtime도
rollback용으로 남아 있지만 새 Base에는 사용하지 않는다.
legacy/reference 설정은 기존 generation-only decay와 patience 5를 유지한다. 새
normalized Guided G1 이상은 source anchor와 별도 fine-tuning 설정을 사용한다.

새 split은 `config/front_cam_policy_split_stateless_normalized_v1.yaml`을 사람이
검토해 갱신한다. 처음에는 빈 fail-closed template이며 normalized Manual complete
session이 최소 11개/10,000 frames가 된 뒤 train 7 / validation 2 / test 2 이상으로
채운다. G0 split ID는 plain `<session-id>`이며 config의 minimum frame/session gate가
기준 미달 학습을 거부한다.

새 Base 검증 뒤 Guided G1에는
`front_cam_policy_train_stateless_normalized_v1_g1.yaml`과 별도의 빈
`front_cam_policy_split_stateless_normalized_v1_g1.yaml`을 사용한다. 기존 G1 split
ID는 `manual/<session-id>` 또는 `guided/<session-id>`이고, 분리 저장한 Guided
cohort는 `guided/generation_<N>/<collection-id>/<session-id>`로 적는다. 서로 다른
root에 같은 directory 이름이 있어도 source-qualified ID가 다르므로 충돌하지
않는다.
guided session에 generation이 없거나 train split에 generation 1 sample이 없거나
더 미래 generation이 있으면 검증을 거부한다.
각 Guided 세대는 완료 session 약 5개를 수집해 train/validation/test `3/1/1`로
배치한다. 다음 세대 split은 이전 세대까지 누적하되 한번 정한 session 소속은
바꾸지 않는다. Manual G0 split도 새로 수집하거나 이동하지 않고 replay anchor로
고정한다.

기존 Base snapshot은 2026-08-14와 2026-08-17에 수집한 manual generation 0의
32 sessions/30,889 samples를 사용한다. guided sample은 split에 포함하지 않는다.
시간상 인접한 frame이 split 사이에 직접 섞이지 않도록 session timestamp 순서로
train 25개/22,958, validation 3개/3,886, test 4개/4,045 samples로 배치했다. speed
label은 30,859개가 15이고 나머지 30개만 trigger 전환 구간의 6.703~14.628이므로 이
Base의 speed head는 사실상 15 유지 모델이다. 이 32개 session은 legacy steering
label이므로 새 normalized split에는 하나도 포함하지 않는다. 새 generation 0은
normalized Manual만 EMA mass 1.0으로 sampling하며 ImageNet pretrained weight에서
새로 시작한다.

새 G1 이상은 source별 총 sampling mass를 먼저 고정하고 같은 source 안에서만
세대 decay를 정규화한다. raw session은 삭제하지 않으며 같은 source/generation
안에서는 frame을 균등하게 뽑는다.

```text
source mass:
  Manual = 0.5
  Guided = 0.5

within Guided:
  raw_mass(g) = 0.8 ** (current_generation - g)
  mass(g) = 0.5 * raw_mass(g) / sum(Guided raw masses)

G2 example:
  Manual G0 = 0.5000
  Guided G2 = 0.2778
  Guided G1 = 0.2222
```

`source_sampling_masses` key가 없는 legacy 설정은 기존 generation-only 계산을
그대로 유지한다. 새 설정은 source key가 정확히 `manual/guided`, 각 값이 finite
양수이고 합이 1일 때만 시작한다. epoch sample 수는 current generation이 기대값
기준 한 번 노출되도록 current sample 수를 current sampling mass로 나눈 값이다.
class frequency weight와 validation checkpoint 선택식
`angle_mae + 0.25 * speed_mae`에도 같은 source/generation mass를 적용한다. stats와
checkpoint에는 source/generation별 sample 수와 mass를 기록한다.
`manual_anchor_split_manifest`는 G0 split의 Manual session 소속을 split별로 정확히
고정하고 `current_generation_session_counts: {train: 3, val: 1, test: 1}`은 매
Guided generation의 5-session 배치를 fail-closed로 검사한다.

generation 0은 ImageNet pretrained weight에서 시작한다. generation 1은 전용 G1
config와 직전 stateless Base의 `best.pt`를 `--initialize-from`으로 반드시 지정한다. 이 옵션은
model state만 strict load하고 optimizer, scheduler, AMP scaler, epoch와
early-stopping state는 새로 시작한다. 같은 generation run의 중단 재개만
`--resume`을 사용한다. G1 이상은 전체 model을 learning rate `1e-4`, 최대 10
epochs, patience 3으로 fine-tune한다. `train_angle_mean_window: 1`과 angle/speed
label smoothing `0.02`는 유지하며 시간축 조향 평균이나 model-weight EMA는 없다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw/ai
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_stateless_normalized_v1.yaml \
  --validate-only

# generation 0
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_stateless_normalized_v1.yaml

# generation 1: normalized Manual + Guided G1 split
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_stateless_normalized_v1_g1.yaml \
  --initialize-from artifacts/runs/front_cam_policy/<previous-run>/best.pt
```

학습 뒤 parent와 candidate를 동일한 누적 validation split에서 비교한다. 비교
명령은 50:50 전체 score와 Guided angle MAE 개선, Manual angle MAE 회귀 `+2.0`
이하, Manual within-10 하락 `3%p` 이하, Manual/Guided speed MAE 회귀 각각 `+1.0`
이하와 parent checkpoint hash lineage를 모두 검사한다. 실패하면 non-zero로
종료하고 같은 generation에서 데이터나 학습을 보완한다.

```bash
/home/xytron/.local/bin/uv run --locked xycar-compare-policy \
  --config config/front_cam_policy_train_stateless_normalized_v1_g1.yaml \
  --parent artifacts/runs/front_cam_policy/<previous-run>/best.pt \
  --candidate artifacts/runs/front_cam_policy/<current-run>/best.pt
```

성공하면 candidate run에 `promotion_gate.json`을 atomic하게 기록한다. G1 이상
export는 이 report가 모든 gate를 통과했고 candidate SHA-256과 일치할 때만
허용한다. manifest에는 report, parent와 candidate hash가 들어간다. G0 export는
promotion report가 필요 없다.

```bash
/home/xytron/.local/bin/uv run --locked xycar-export-policy \
  --checkpoint artifacts/runs/front_cam_policy/<current-run>/best.pt \
  --promotion-report artifacts/runs/front_cam_policy/<current-run>/promotion_gate.json \
  --artifact-id <schema-v1-stateless-artifact-id> \
  --require-schema-version 1
```

offline gate를 통과한 artifact도 별도 실행 승인을 받은 제한 실차 검증 전에는 다음
Guided 세대 parent로 사용하지 않는다. G2 이상은 G1 config와 split을 새 이름으로
복사하고 `current_generation`, `split_manifest`, `output.run_name`을 함께 올린다.

차량 두 root를 Laptop으로 가져올 때는 하나의 directory로 합치지 않는다. manual은
직결 LAN exact mirror를 쓴다. Guided는 generation과 collection ID의 정확한 cohort
root를 같은 상대 구조의 Laptop root로 지정하고, Tailscale no-delete script를 먼저
dry-run한 뒤 `--apply`한다. 상위 `stateless_guided` 전체를 source로 지정하지 않는다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw
./scripts/ai/sync_stateless_manual_lan.sh

GENERATION=2
COLLECTION_ID=g2-20260817-a
XYCAR_AI_VEHICLE_DATASET_ROOT=/home/xytron/xycar_data/stateless_guided/generation_${GENERATION}/${COLLECTION_ID} \
XYCAR_AI_LOCAL_DATASET_ROOT="$PWD/ai/datasets/stateless_guided/generation_${GENERATION}/${COLLECTION_ID}" \
  ./scripts/ai/sync_dataset.sh
XYCAR_AI_VEHICLE_DATASET_ROOT=/home/xytron/xycar_data/stateless_guided/generation_${GENERATION}/${COLLECTION_ID} \
XYCAR_AI_LOCAL_DATASET_ROOT="$PWD/ai/datasets/stateless_guided/generation_${GENERATION}/${COLLECTION_ID}" \
  ./scripts/ai/sync_dataset.sh --apply
```

### 기존 stateless/AR 감속 비교 재현 (rollback 전용)

기존 min-speed-20 split은 speed를 크게 낮춘 세 session이 모두 train에 있어
validation 685장은 전부 speed 25였고 test도 702장 중 699장이 speed 25였다.
감속 능력을 별도로 검증할 때는 기존 실험의 재현성을 보존하면서
`config/front_cam_policy_split_min_speed20_speed_balanced.yaml`을 사용한다. 감속
session 하나씩을 train/validation/test에 session-disjoint하게 배치하며 sample 수와
`speed < 25` 비율은 다음과 같다.

```text
train: 2,999 samples, 224 slowdown samples (7.47%)
val:     849 samples, 161 slowdown samples (18.96%)
test:    766 samples, 186 slowdown samples (24.28%)
```

stateless와 AR을 공정하게 비교하려면 기존 checkpoint를 새 validation/test에
재평가하지 않는다. 새 validation/test session이 기존 checkpoint의 train에 들어간
적이 있기 때문에 두 모델 모두 아래 전용 config로 처음부터 다시 학습한다. 두
config는 같은 road warp, centered-5 angle, optimizer, loss와 seed를 사용한다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw/ai
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_small_warp_speed_balanced.yaml \
  --validate-only
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_small_warp_ar_shared_speed_balanced.yaml \
  --validate-only

/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_small_warp_speed_balanced.yaml
/home/xytron/.local/bin/uv run --locked xycar-train \
  --config config/front_cam_policy_train_small_warp_ar_shared_speed_balanced.yaml
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
  probe_summary.json  # --stop-after-epoch probe에서만 생성
  test_metrics.json
  summary.json
  promotion_gate.json  # G1+ parent/candidate offline 비교가 통과하면 생성
```

best checkpoint 선택식은 `val_angle_mae + 0.25 * val_speed_mae`다. A/B winner도
두 `summary.json`의 `best_score`가 더 낮은 run으로 정하고 동률이면 baseline을
선택한다. test 결과는 winner를 validation으로 고른 뒤 최종 확인에만 사용한다.
tiny A/B 설정은 validation best score가 5 epochs 연속 개선되지 않으면 조기
종료한다. `last.pt`에 연속 미개선 횟수를 저장하므로 resume에서도 patience가
이어지며, `summary.json`에 최대/완료 epoch와 조기 종료 여부를 기록한다.
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
artifact ID는 덮어쓰지 않는다. stateless schema v1은 image `[1,3,224,224]` 하나를
받고 기존 AR schema v2는 image와 predicted history `[1,4,2]`를 함께 받는다.
현재 AR exporter는 같은 input shape에 실제 실행 history를 요구하는 schema v3를
생성한다. v1/v2 runtime 호환은 유지한다.

G1 이상 run은 아래 명령에 같은 run의
`--promotion-report artifacts/runs/front_cam_policy/<stateless-run>/promotion_gate.json`
을 추가한다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw/ai
/home/xytron/.local/bin/uv run --locked xycar-export-policy \
  --checkpoint artifacts/runs/front_cam_policy/<stateless-run>/best.pt \
  --artifact-id <schema-v1-stateless-artifact-id> \
  --require-schema-version 1
```

exporter는 checkpoint model state를 strict load하고 eager/trace/reload 결과와 두
`[1,201]` 출력을 확인한 뒤 `model.ts`, `manifest.yaml`, `SHA256SUMS`를 atomic하게
생성한다. manifest에는 checkpoint/source/dataset, RGB input, timm preprocessing,
label decode, `normalized_percent_v1`, CPU thread와 warm-up 계약이 포함된다. 차량 배포와 실행 방법은
`src/xycar_ai_drive/README.md`를 따른다.

## 대회 신호등·지름길 temporal pipeline

대회 pipeline은 기존 schema v1 Base Lap artifact를 변경하지 않고 signal과
shortcut policy를 별도로 학습한다. signal은 상단 2/3 RGB의
MobileNetV3-small+GRU, shortcut은 unwarped full-frame과 직전 실제 명령을 받는
ViT-tiny+GRU다. 전체 결정은 ROS runtime의 deterministic FSM이 수행한다.

### 데이터 동기화와 승인

Jetson의 `/home/xytron/xycar_data/competition_manual`을 로컬
`datasets/competition_manual`로 증분 복사한다. 기본은 dry-run이고 `--apply`도
vehicle source를 삭제하지 않는다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw
./scripts/ai/sync_competition_dataset.sh
./scripts/ai/sync_competition_dataset.sh --apply
```

각 완료 session은 다음 순서로 라벨링한다. `init`이 생성한
`mission_annotations.yaml`에서 `annotator`와 null 값을 채운다. signal bbox는 원본
frame 정규화 좌표고 shortcut phase는 `APPROACH`, `ENTRY_STRAIGHT`, `FIRST_LEFT`,
`CONNECTOR`, `SECOND_LEFT`, `REACQUIRE` 순서다. 초안 작성자와 다른 사람이
연속 frame을 검수한 뒤에만 approve한다. approve는 annotation을 원본
`samples.csv` SHA-256과 함께 고정한다.

```bash
cd /home/xytron/xycar_ws/apps/xycar_ws_mgw/ai
/home/xytron/.local/bin/uv run --locked xycar-competition-data init \
  --session datasets/competition_manual/<session-id>
/home/xytron/.local/bin/uv run --locked xycar-competition-data validate \
  --session datasets/competition_manual/<session-id>
/home/xytron/.local/bin/uv run --locked xycar-competition-data approve \
  --session datasets/competition_manual/<session-id> --reviewer <reviewer>
/home/xytron/.local/bin/uv run --locked xycar-competition-data validate \
  --dataset-root datasets/competition_manual --require-approved
```

학습 최소치는 red/yellow/left/straight-green 각각 20 session, 신호가 없는
negative 10 session과 완전한 shortcut 30 session이다. 세부 현장 체크는 상위
하네스의 `docs/competition_ai_data_checklist.md`를 따른다.

### Session-disjoint split과 학습

signal split에는 signal과 shortcut capture를 모두 넣어 shortcut 접근 중의 신호도
학습한다. shortcut split은 `capture_kind=shortcut`만 사용한다. 생성된 split
파일은 실제 dataset snapshot의 session ID를 담는 실험 입력이다. 기존 파일을
덮어쓰지 않으므로 새 snapshot이면 ID 또는 output 경로를 바꾼다.

```bash
/home/xytron/.local/bin/uv run --locked xycar-competition-data make-split \
  --dataset-root datasets/competition_manual \
  --output config/competition_signal_split.yaml --seed 20260815
/home/xytron/.local/bin/uv run --locked xycar-competition-data make-split \
  --dataset-root datasets/competition_manual \
  --output config/competition_shortcut_split.yaml --seed 20260815 \
  --capture-kind shortcut

/home/xytron/.local/bin/uv run --locked xycar-train-signal \
  --config config/traffic_signal_policy.yaml --validate-only
/home/xytron/.local/bin/uv run --locked xycar-train-shortcut \
  --config config/shortcut_policy.yaml --validate-only
/home/xytron/.local/bin/uv run --locked xycar-train-signal \
  --config config/traffic_signal_policy.yaml
/home/xytron/.local/bin/uv run --locked xycar-train-shortcut \
  --config config/shortcut_policy.yaml
```

signal train loader는 red, yellow, left, green, approach-unknown, background window가
epoch마다 같은 총 sampling mass를 갖도록 inverse-frequency sampler를 사용한다.
validation과 test는 실제 분포를 유지한다. split별로 모든 lamp와 negative가 한
session 이상 없으면 학습을 거부하므로, 그 경우 새 seed로 split을 생성한다.
horizontal flip은 신호 의미와 지름길 geometry를 바꾸므로 사용하지 않는다.
shortcut 학습은 normalized v1 mission session만 허용하고 checkpoint provenance에
steering contract를 남긴다. bundle builder는 조향을 출력하는 Base와 shortcut의
계약이 정확히 일치하지 않으면 bundle을 만들지 않는다. signal model은 조향을
출력하지 않으므로 이 비교 대상이 아니다.

run은 `config.yaml`, session/annotation checksum provenance, `metrics.csv`,
`best.pt`, `last.pt`, `test_metrics.json`을 남긴다. 같은 run directory는
덮어쓰지 않는다. best 선택은 단순 평균 loss가 아니라 signal의 stop
false-negative/false-left 또는 shortcut의 early-handoff/angle MAE를 크게 벌점으로
주는 validation safety score를 사용한다.

### Export와 competition bundle

signal test의 stop false-negative와 false-left가 모두 0, shortcut test의 early
handoff가 0이고 first-step angle MAE가 10 이하일 때만 기본 export가
race-qualified가 된다. gate 미통과 model은
`--development-artifact`로 개별 offline 분석용 export만 가능하고 bundle에는
들어갈 수 없다.

```bash
/home/xytron/.local/bin/uv run --locked xycar-export-signal \
  --checkpoint artifacts/runs/traffic_signal_policy/<run-id>/best.pt \
  --artifact-id <signal-artifact-id>
/home/xytron/.local/bin/uv run --locked xycar-export-shortcut \
  --checkpoint artifacts/runs/shortcut_policy/<run-id>/best.pt \
  --artifact-id <shortcut-artifact-id>
/home/xytron/.local/bin/uv run --locked xycar-build-competition-bundle \
  --base-artifact artifacts/models/<schema-v1-base-artifact-id> \
  --signal-artifact artifacts/models/<signal-artifact-id> \
  --shortcut-artifact artifacts/models/<shortcut-artifact-id> \
  --artifact-id <competition-bundle-id>
```

bundle은 `base_model.ts`, `signal_model.ts`, `shortcut_model.ts`, 세 source
manifest, FSM threshold와 전체 `SHA256SUMS`를 포함한다. runtime은 세 model을
시작 시 CUDA memory에 모두 올리고 warm-up하므로 주행 중 model load는 없다.
