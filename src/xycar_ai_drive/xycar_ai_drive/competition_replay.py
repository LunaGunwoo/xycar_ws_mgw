"""Offline replay and latency benchmark for a competition bundle."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import cv2

from xycar_ai_drive.competition_fsm import CompetitionMode, MissionStateMachine
from xycar_ai_drive.competition_gpu_runtime import CompetitionGpuRuntime
from xycar_ai_drive.control import STOP_COMMAND


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--run-mode",
        choices=("signal_only", "shortcut_only", "combined"),
        required=True,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup-count", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output")
    options = parser.parse_args(args)
    runtime = CompetitionGpuRuntime(
        artifact_dir=options.artifact_dir,
        device=options.device,
        torch_num_threads=4,
        warmup_count=options.warmup_count,
    )
    session = Path(options.session).expanduser().resolve()
    frames = _session_frames(session)
    if options.max_frames > 0:
        frames = frames[: options.max_frames]
    previous = STOP_COMMAND
    latencies: list[float] = []
    mode_counts: dict[str, int] = {}
    transitions: list[dict[str, object]] = []
    faults: list[str] = []
    if options.run_mode == "signal_only":
        fsm = None
    else:
        fsm = MissionStateMachine(
            runtime.artifact.mission,
            shortcut_only=options.run_mode == "shortcut_only",
        )
        fsm.enable(0.0)
    for replay_index, image_path in enumerate(frames):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not read replay image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if fsm is None:
            inference_mode = "signal_only"
        else:
            if fsm.mode in {CompetitionMode.DISABLED, CompetitionMode.FAULT}:
                break
            inference_mode = fsm.inference_mode
        inference = runtime.infer(
            rgb,
            mode=inference_mode,
            previous_command=previous,
        )
        latencies.append(inference.inference_ms)
        mode_counts[inference_mode] = mode_counts.get(inference_mode, 0) + 1
        if fsm is not None:
            before = fsm.mode
            decision = fsm.update(
                inference,
                now_monotonic=replay_index / 20.0,
            )
            previous = decision.command
            if decision.reset_shortcut:
                runtime.reset_shortcut()
            if decision.mode != before:
                transitions.append(
                    {
                        "frame": replay_index + 1,
                        "from": before.value,
                        "to": decision.mode.value,
                        "reason": decision.reason,
                    }
                )
            if decision.mode == CompetitionMode.FAULT:
                faults.append(decision.reason)
        else:
            previous = STOP_COMMAND
    if not latencies:
        raise RuntimeError("replay produced no inference")
    ordered = sorted(latencies)
    report = {
        "artifact_id": runtime.artifact.artifact_id,
        "run_mode": options.run_mode,
        "frames": len(latencies),
        "inference_ms": {
            "mean": statistics.fmean(latencies),
            "p50": _percentile(ordered, 0.50),
            "p95": _percentile(ordered, 0.95),
            "p99": _percentile(ordered, 0.99),
            "maximum": max(latencies),
        },
        "mode_counts": mode_counts,
        "transitions": transitions,
        "faults": faults,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if options.output:
        output = Path(options.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def _session_frames(session: Path) -> list[Path]:
    samples_path = session / "samples.csv"
    if not samples_path.is_file():
        raise RuntimeError(f"samples.csv is missing: {session}")
    result: list[Path] = []
    with samples_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            relative = Path(row["image"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("unsafe replay image path")
            path = session / relative
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"replay image is missing: {path}")
            result.append(path)
    return result


def _percentile(ordered: list[float], fraction: float) -> float:
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


if __name__ == "__main__":
    main()
