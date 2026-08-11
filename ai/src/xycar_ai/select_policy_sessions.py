from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import yaml

from xycar_ai.front_cam_policy_data import (
    SESSION_NAME_RE,
    metadata_matches_policy_filter,
)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List completed policy sessions matching a metadata filter."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--min-forward-speed", required=True, type=float)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset source does not exist: {root}")
    if not -100.0 <= args.min_forward_speed <= 100.0:
        raise ValueError("min-forward-speed must be in [-100, 100]")

    selected: list[str] = []
    invalid_metadata: list[Path] = []
    for path in sorted(root.iterdir()):
        if (
            not path.is_dir()
            or path.is_symlink()
            or not SESSION_NAME_RE.fullmatch(path.name)
        ):
            continue
        metadata_path = path / "metadata.yaml"
        try:
            metadata = _load_metadata(metadata_path)
        except (OSError, TypeError, UnicodeError, ValueError):
            invalid_metadata.append(metadata_path)
            continue
        if metadata_matches_policy_filter(
            metadata,
            control_mode="gamepad",
            max_forward_speed=None,
            min_forward_speed=args.min_forward_speed,
        ):
            selected.append(path.name)

    if invalid_metadata:
        print(
            "warning: skipped session(s) with missing or invalid metadata: "
            + ", ".join(str(path) for path in invalid_metadata),
            file=sys.stderr,
        )

    if not selected and not args.allow_empty:
        raise ValueError(
            "no completed gamepad sessions match "
            f"min_forward_speed>={args.min_forward_speed} under {root}"
        )
    if selected:
        print("\n".join(selected))
    return 0


def _load_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing metadata.yaml: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"metadata root must be a mapping: {path}")
    return dict(payload)


if __name__ == "__main__":
    raise SystemExit(main())
