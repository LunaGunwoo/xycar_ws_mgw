# Xycar MGW

실차 source의 원본은 개발 PC
`/home/xytron/xycar_ws/apps/xycar_ws_mgw`다. 차량
`xytron@xycar:/home/xytron/xycar_ws_mgw`는 Tailscale 배포 checkout이며 tracked
source를 현장에서 수정하지 않는다.

차량 반영 순서:

```bash
getent hosts xycar
ssh xytron@xycar
cd /home/xytron/xycar_ws_mgw
git status --short --branch
git pull --ff-only
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select <changed-package>
```

개발 Laptop과 차량은 Tailscale 설치 후 같은 tailnet에 로그인돼 있어야 하며
`getent hosts xycar`에서 차량 MagicDNS 이름이 해석돼야 한다. WSL은 Windows
Tailscale을 사용할 수 있어 Linux `tailscale` CLI 자체는 필수가 아니다.

모터 또는 실제 센서 장치를 시작하는 명령은 `AGENTS.md`의 매 실행 승인 규칙을
따른다. package별 build·run 방법은 해당 `src/<package>/README.md`가 기준이다.

독립 AI 학습 workflow는 `ai/README.md`를 따른다. 학습 source는 `ai/`에
작성하고 현재 RTX 4090 Laptop에서 학습·평가·artifact 생성까지 수행한다.
dataset은 차량 또는 marker가 있는 외장 SSD에서 증분 동기화하며 model과 함께
Git에 넣지 않는다. Front-camera TorchScript 주행은
`src/xycar_ai_drive/README.md`를 따른다.
