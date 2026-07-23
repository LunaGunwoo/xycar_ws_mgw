# Xycar MGW

실차 source의 원본은 개발 PC
`/home/xytron/xycar_ws/apps/xycar_ws_mgw`다. 차량
`xytron@10.42.0.1:/home/xytron/xycar_ws_mgw`는 배포 checkout이며 tracked
source를 현장에서 수정하지 않는다.

차량 반영 순서:

```bash
ssh xytron@10.42.0.1
cd /home/xytron/xycar_ws_mgw
git status --short --branch
git pull --ff-only
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select <changed-package>
```

모터 또는 실제 센서 장치를 시작하는 명령은 `AGENTS.md`의 매 실행 승인 규칙을
따른다. package별 build·run 방법은 해당 `src/<package>/README.md`가 기준이다.

독립 AI 학습 workflow는 `ai/README.md`를 따른다. 학습 source는 `ai/`에
작성하고 5090의 `/home/gunwoo/Documents/xycar-ai` 루트로 flatten한다.
dataset과 model은 Git에 넣지 않는다.
