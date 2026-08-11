from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path, PurePosixPath
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from xycar_ai.front_cam_policy_warp import (
    RoadWarpConfig,
    draw_warp_overlay,
    load_road_warp_config,
    save_road_warp_config,
    warp_image_array,
)

DEFAULT_CONFIG = "config/front_cam_policy_preprocess.yaml"
DEFAULT_DATASET_ROOT = "datasets/teleop"

FLOAT_PARAMETERS = {
    "top_y": (0.0, 1.0, 0.001),
    "bottom_y": (0.0, 1.0, 0.001),
    "top_left_x": (0.0, 1.0, 0.001),
    "top_right_x": (0.0, 1.0, 0.001),
    "bottom_left_x": (0.0, 1.0, 0.001),
    "bottom_right_x": (0.0, 1.0, 0.001),
    "dst_left_x": (0.0, 0.49, 0.001),
    "dst_right_x": (0.51, 1.0, 0.001),
}
INTEGER_PARAMETERS = {
    "bev_width": (80, 1920, 1),
    "bev_height": (60, 1080, 1),
}


class WarpTunerState:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.saved = load_road_warp_config(self.config_path)
        self.pending_values = self.saved.serializable()

    def set_value(self, field_name: str, value: float | int) -> None:
        if field_name not in self.pending_values:
            raise KeyError(f"unknown warp parameter: {field_name}")
        self.pending_values[field_name] = value

    def pending_config(self) -> RoadWarpConfig:
        return RoadWarpConfig(**self.pending_values)

    def has_pending_changes(self) -> bool:
        try:
            return self.pending_config() != self.saved
        except (TypeError, ValueError):
            return True

    def reset(self) -> None:
        self.pending_values = self.saved.serializable()

    def save(self) -> RoadWarpConfig:
        config = self.pending_config()
        save_road_warp_config(self.config_path, config)
        self.saved = config
        self.pending_values = config.serializable()
        return config


class WarpTunerApplication:
    def __init__(
        self,
        *,
        config_path: Path,
        image_paths: list[Path],
        start_index: int,
    ) -> None:
        if not image_paths:
            raise ValueError("at least one source image is required")
        self.state = WarpTunerState(config_path)
        self.image_paths = image_paths
        self.image_index = min(max(start_index, 0), len(image_paths) - 1)
        self.root = tk.Tk()
        self.root.title("Xycar front-camera road warp tuner")
        self.root.geometry("1420x860")
        self.root.minsize(1100, 720)
        self.variables: dict[str, tk.DoubleVar | tk.IntVar] = {}
        self.dimension_inputs: dict[str, tk.StringVar] = {}
        self.image_number = tk.StringVar(value=str(self.image_index + 1))
        self._original_photo: ImageTk.PhotoImage | None = None
        self._warped_photo: ImageTk.PhotoImage | None = None
        self._build_layout()
        self._bind_keys()
        self._sync_controls_from_state()
        self.refresh_preview()

    def run(self) -> None:
        self.root.mainloop()

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        preview = ttk.Frame(outer)
        preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        controls = ttk.Frame(outer, padding=(16, 0, 0, 0), width=420)
        controls.pack(side=tk.RIGHT, fill=tk.Y)

        self.original_title = ttk.Label(preview, text="Original + road ROI")
        self.original_title.pack(anchor=tk.W)
        self.original_label = ttk.Label(preview, anchor=tk.CENTER)
        self.original_label.pack(fill=tk.BOTH, expand=True, pady=(4, 12))
        ttk.Label(
            preview,
            text="Warped road output (training then resizes this to 224×224)",
        ).pack(anchor=tk.W)
        self.warped_label = ttk.Label(preview, anchor=tk.CENTER)
        self.warped_label.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        ttk.Label(
            controls,
            text="Perspective warp parameters",
            font=("TkDefaultFont", 14, "bold"),
        ).pack(anchor=tk.W, pady=(0, 4))
        ttk.Label(
            controls,
            text="Preview changes are not written until Save is pressed.",
            wraplength=390,
        ).pack(anchor=tk.W, pady=(0, 8))

        parameter_frame = ttk.Frame(controls)
        parameter_frame.pack(fill=tk.X)
        for field in fields(RoadWarpConfig):
            name = field.name
            row = ttk.Frame(parameter_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=name, width=18).pack(side=tk.LEFT)
            if name in INTEGER_PARAMETERS:
                minimum, maximum, resolution = INTEGER_PARAMETERS[name]
                variable: tk.DoubleVar | tk.IntVar = tk.IntVar()
            else:
                minimum, maximum, resolution = FLOAT_PARAMETERS[name]
                variable = tk.DoubleVar()
            self.variables[name] = variable
            scale = tk.Scale(
                row,
                from_=minimum,
                to=maximum,
                resolution=resolution,
                orient=tk.HORIZONTAL,
                showvalue=True,
                variable=variable,
                command=lambda _value, field_name=name: self._parameter_changed(
                    field_name
                ),
                length=185 if name in INTEGER_PARAMETERS else 235,
            )
            scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            if name in INTEGER_PARAMETERS:
                dimension_input = tk.StringVar()
                self.dimension_inputs[name] = dimension_input
                entry = ttk.Entry(row, textvariable=dimension_input, width=6)
                entry.pack(side=tk.RIGHT, padx=(4, 0))
                entry.bind(
                    "<Return>",
                    lambda _event, field_name=name: self._dimension_changed(
                        field_name
                    ),
                )

        action_row = ttk.Frame(controls)
        action_row.pack(fill=tk.X, pady=(12, 4))
        ttk.Button(action_row, text="Save YAML", command=self.save).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(action_row, text="Reset", command=self.reset).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(8, 0)
        )
        image_row = ttk.Frame(controls)
        image_row.pack(fill=tk.X, pady=4)
        ttk.Button(image_row, text="◀ Previous", command=self.previous_image).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(image_row, text="Next ▶", command=self.next_image).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(8, 0)
        )
        jump_row = ttk.Frame(controls)
        jump_row.pack(fill=tk.X, pady=4)
        ttk.Label(jump_row, text="Image number (1-based)").pack(side=tk.LEFT)
        self.image_number_entry = ttk.Entry(
            jump_row,
            textvariable=self.image_number,
            width=9,
        )
        self.image_number_entry.pack(side=tk.LEFT, padx=(8, 4))
        self.image_number_entry.bind("<Return>", lambda _event: self.go_to_image())
        ttk.Button(jump_row, text="Go", command=self.go_to_image).pack(side=tk.LEFT)
        ttk.Button(controls, text="Quit", command=self.root.destroy).pack(
            fill=tk.X, pady=(4, 8)
        )
        self.status = ttk.Label(controls, wraplength=390, justify=tk.LEFT)
        self.status.pack(anchor=tk.W, fill=tk.X)
        ttk.Label(
            controls,
            text="Keys: S save · R reset · ←/P previous · →/N next · Q/Esc quit",
            wraplength=390,
        ).pack(anchor=tk.W, side=tk.BOTTOM, pady=(12, 0))

    def _bind_keys(self) -> None:
        self.root.bind("<Escape>", lambda _event: self.root.destroy())
        self.root.bind("q", lambda _event: self.root.destroy())
        self.root.bind("s", lambda _event: self.save())
        self.root.bind("r", lambda _event: self.reset())
        self.root.bind("<Left>", lambda _event: self.previous_image())
        self.root.bind("p", lambda _event: self.previous_image())
        self.root.bind("<Right>", lambda _event: self.next_image())
        self.root.bind("n", lambda _event: self.next_image())

    def _sync_controls_from_state(self) -> None:
        for name, value in self.state.pending_values.items():
            self.variables[name].set(value)
            if name in self.dimension_inputs:
                self.dimension_inputs[name].set(str(value))

    def _parameter_changed(self, field_name: str) -> None:
        variable = self.variables[field_name]
        value: float | int
        if field_name in INTEGER_PARAMETERS:
            value = int(variable.get())
        else:
            value = float(variable.get())
        self.state.set_value(field_name, value)
        if field_name in self.dimension_inputs:
            self.dimension_inputs[field_name].set(str(value))
        self.refresh_preview()

    def _dimension_changed(self, field_name: str) -> None:
        input_variable = self.dimension_inputs[field_name]
        minimum, maximum, _resolution = INTEGER_PARAMETERS[field_name]
        try:
            value = int(input_variable.get().strip())
        except ValueError:
            value = -1
        if not minimum <= value <= maximum:
            messagebox.showerror(
                "Invalid warp dimension",
                f"{field_name} must be a whole number from {minimum} to {maximum}.",
            )
            input_variable.set(str(self.state.pending_values[field_name]))
            return
        self.variables[field_name].set(value)
        self.state.set_value(field_name, value)
        self.refresh_preview()

    def refresh_preview(self) -> None:
        path = self.image_paths[self.image_index]
        self.image_number.set(str(self.image_index + 1))
        try:
            config = self.state.pending_config()
            with Image.open(path) as source:
                rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
            overlay = draw_warp_overlay(rgb, config)
            warped = warp_image_array(rgb, config)
            self._original_photo = _photo_for_panel(overlay, (760, 390))
            self._warped_photo = _photo_for_panel(warped, (760, 330))
            self.original_label.configure(image=self._original_photo)
            self.warped_label.configure(image=self._warped_photo)
            state = "UNSAVED preview" if self.state.has_pending_changes() else "saved"
            self.status.configure(
                text=(
                    f"{state}\nImage {self.image_index + 1}/{len(self.image_paths)}\n"
                    f"{path}\nConfig: {self.state.config_path}"
                )
            )
        except (OSError, TypeError, ValueError) as exc:
            self.status.configure(text=f"Invalid preview: {exc}")

    def save(self) -> None:
        try:
            self.state.save()
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("Could not save warp YAML", str(exc))
            return
        self.refresh_preview()

    def reset(self) -> None:
        self.state.reset()
        self._sync_controls_from_state()
        self.refresh_preview()

    def previous_image(self) -> None:
        self.image_index = (self.image_index - 1) % len(self.image_paths)
        self.refresh_preview()

    def next_image(self) -> None:
        self.image_index = (self.image_index + 1) % len(self.image_paths)
        self.refresh_preview()

    def go_to_image(self) -> None:
        raw_number = self.image_number.get().strip()
        try:
            image_number = int(raw_number)
        except ValueError:
            messagebox.showerror(
                "Invalid image number",
                f"Enter a whole number from 1 to {len(self.image_paths)}.",
            )
            self.image_number.set(str(self.image_index + 1))
            return
        if not 1 <= image_number <= len(self.image_paths):
            messagebox.showerror(
                "Invalid image number",
                f"Image number must be from 1 to {len(self.image_paths)}.",
            )
            self.image_number.set(str(self.image_index + 1))
            return
        self.image_index = image_number - 1
        self.refresh_preview()


def discover_dataset_images(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    images: list[Path] = []
    for samples_path in sorted(root.glob("*_session*/samples.csv")):
        session_root = samples_path.parent.resolve()
        with samples_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "image" not in (reader.fieldnames or []):
                continue
            for row in reader:
                relative = row.get("image", "")
                posix = PurePosixPath(relative)
                if not relative or posix.is_absolute() or ".." in posix.parts:
                    continue
                candidate = session_root.joinpath(*posix.parts).resolve()
                if session_root not in candidate.parents or not candidate.is_file():
                    continue
                images.append(candidate)
    if not images:
        raise ValueError(f"no valid dataset images found under {root}")
    return images


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune the offline front-camera road perspective warp."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--image",
        default="",
        help="use one image instead of discovering samples from the dataset",
    )
    parser.add_argument("--start-index", type=int, default=0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    if args.image:
        image_path = Path(args.image).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"source image does not exist: {image_path}")
        image_paths = [image_path]
    else:
        image_paths = discover_dataset_images(args.dataset_root)
    application = WarpTunerApplication(
        config_path=config_path,
        image_paths=image_paths,
        start_index=args.start_index,
    )
    application.run()
    return 0


def _photo_for_panel(image: np.ndarray, bounds: tuple[int, int]) -> ImageTk.PhotoImage:
    pil_image = Image.fromarray(image)
    pil_image.thumbnail(bounds, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(pil_image)


if __name__ == "__main__":
    raise SystemExit(main())
