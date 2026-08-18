"""Tune road-warp parameters from the live ROS front-camera topic."""

from __future__ import annotations

import time
import tkinter as tk
from dataclasses import fields
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PilImage
from PIL import ImageTk
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from xycar_ai_drive.artifact import RoadWarpParameters
from xycar_ai_drive.road_warp import (
    draw_road_warp_overlay,
    load_road_warp_config,
    road_warp_from_mapping,
    road_warp_values,
    save_road_warp_config,
    warp_road_image,
)

DEFAULT_INITIAL_CONFIG = (
    '/home/xytron/xycar_ws_mgw/ai/config/'
    'front_cam_policy_preprocess.yaml'
)
DEFAULT_OUTPUT_CONFIG = (
    '/home/xytron/.config/xycar/front_cam_policy_preprocess.yaml'
)
FLOAT_PARAMETERS = {
    'top_y': (0.0, 1.0, 0.001),
    'bottom_y': (0.0, 1.0, 0.001),
    'top_left_x': (0.0, 1.0, 0.001),
    'top_right_x': (0.0, 1.0, 0.001),
    'bottom_left_x': (0.0, 1.0, 0.001),
    'bottom_right_x': (0.0, 1.0, 0.001),
    'dst_left_x': (0.0, 0.49, 0.001),
    'dst_right_x': (0.51, 1.0, 0.001),
}
INTEGER_PARAMETERS = {
    'bev_width': (80, 1920, 1),
    'bev_height': (60, 1080, 1),
}


class LiveWarpTunerNode(Node):
    """Subscribe to the camera without creating Joy or motor endpoints."""

    def __init__(self) -> None:
        super().__init__('live_warp_tuner')
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('initial_config_path', DEFAULT_INITIAL_CONFIG)
        self.declare_parameter('output_config_path', DEFAULT_OUTPUT_CONFIG)

        self.camera_topic = str(self.get_parameter('camera_topic').value)
        initial_path_text = str(
            self.get_parameter('initial_config_path').value
        ).strip()
        output_path_text = str(
            self.get_parameter('output_config_path').value
        ).strip()
        if not self.camera_topic:
            raise ValueError('camera_topic must not be empty')
        if not initial_path_text:
            raise ValueError('initial_config_path must not be empty')
        if not output_path_text:
            raise ValueError('output_config_path must not be empty')

        initial_path = Path(initial_path_text).expanduser()
        self.output_path = Path(output_path_text).expanduser()
        load_path = self.output_path if self.output_path.is_file() else initial_path
        self.initial_config = load_road_warp_config(load_path)
        self.loaded_config_path = load_path
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_monotonic: float | None = None
        self.latest_sequence = 0
        self._bridge = CvBridge()
        self.camera_subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self._on_camera,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            'Live warp tuner started without a motor publisher. '
            f'camera={self.camera_topic}, initial={load_path}, '
            f'output={self.output_path}'
        )

    def _on_camera(self, message: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(
                message,
                desired_encoding='bgr8',
            )
            if (
                not isinstance(frame, np.ndarray)
                or frame.dtype != np.uint8
                or frame.ndim != 3
                or frame.shape[2] != 3
            ):
                raise ValueError('converted frame is not uint8 BGR')
        except Exception as exc:
            self.get_logger().warning(f'camera conversion failed: {exc}')
            return
        self.latest_frame = frame
        self.latest_frame_monotonic = time.monotonic()
        self.latest_sequence += 1


class LiveWarpTunerApplication:
    """Existing offline tuner layout backed by the latest ROS camera frame."""

    def __init__(self, node: LiveWarpTunerNode) -> None:
        self.node = node
        self.saved = node.initial_config
        self.pending_values = road_warp_values(self.saved)
        self.frozen_frame: np.ndarray | None = None
        self.closed = False
        self.root = tk.Tk()
        self.root.title('Xycar front-camera road warp tuner (LIVE)')
        self.root.geometry('1420x860')
        self.root.minsize(1100, 720)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.variables: dict[str, tk.DoubleVar | tk.IntVar] = {}
        self.dimension_inputs: dict[str, tk.StringVar] = {}
        self._original_photo: ImageTk.PhotoImage | None = None
        self._warped_photo: ImageTk.PhotoImage | None = None
        self._last_preview_signature: tuple[object, ...] | None = None
        self._build_layout()
        self._bind_keys()
        self._sync_controls_from_state()
        self._schedule_update()

    def run(self) -> None:
        self.root.mainloop()

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        preview = ttk.Frame(outer)
        preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        controls = ttk.Frame(outer, padding=(16, 0, 0, 0), width=420)
        controls.pack(side=tk.RIGHT, fill=tk.Y)

        self.original_title = ttk.Label(preview, text='LIVE original + road ROI')
        self.original_title.pack(anchor=tk.W)
        self.original_label = ttk.Label(preview, anchor=tk.CENTER)
        self.original_label.pack(fill=tk.BOTH, expand=True, pady=(4, 12))
        ttk.Label(
            preview,
            text='Warped road output (training then resizes this to 224x224)',
        ).pack(anchor=tk.W)
        self.warped_label = ttk.Label(preview, anchor=tk.CENTER)
        self.warped_label.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        ttk.Label(
            controls,
            text='Perspective warp parameters',
            font=('TkDefaultFont', 14, 'bold'),
        ).pack(anchor=tk.W, pady=(0, 4))
        ttk.Label(
            controls,
            text=(
                'Live preview changes are not written until Save YAML '
                'is pressed.'
            ),
            wraplength=390,
        ).pack(anchor=tk.W, pady=(0, 8))

        parameter_frame = ttk.Frame(controls)
        parameter_frame.pack(fill=tk.X)
        for field in fields(RoadWarpParameters):
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
                command=lambda _value, field_name=name: (
                    self._parameter_changed(field_name)
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
                    '<Return>',
                    lambda _event, field_name=name: (
                        self._dimension_changed(field_name)
                    ),
                )

        action_row = ttk.Frame(controls)
        action_row.pack(fill=tk.X, pady=(12, 4))
        ttk.Button(action_row, text='Save YAML', command=self.save).pack(
            side=tk.LEFT,
            expand=True,
            fill=tk.X,
        )
        ttk.Button(action_row, text='Reset', command=self.reset).pack(
            side=tk.LEFT,
            expand=True,
            fill=tk.X,
            padx=(8, 0),
        )
        ttk.Button(
            controls,
            text='Pause current frame',
            command=self.toggle_pause,
        ).pack(fill=tk.X, pady=4)
        ttk.Button(controls, text='Quit', command=self.close).pack(
            fill=tk.X,
            pady=(0, 8),
        )
        self.status = ttk.Label(controls, wraplength=390, justify=tk.LEFT)
        self.status.pack(anchor=tk.W, fill=tk.X)
        ttk.Label(
            controls,
            text='Keys: S save | R reset | Space pause/live | Q/Esc quit',
            wraplength=390,
        ).pack(anchor=tk.W, side=tk.BOTTOM, pady=(12, 0))

    def _bind_keys(self) -> None:
        self.root.bind('<Escape>', lambda _event: self.close())
        self.root.bind('q', lambda _event: self.close())
        self.root.bind('s', lambda _event: self.save())
        self.root.bind('r', lambda _event: self.reset())
        self.root.bind('<space>', lambda _event: self.toggle_pause())

    def _sync_controls_from_state(self) -> None:
        for name, value in self.pending_values.items():
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
        self.pending_values[field_name] = value
        if field_name in self.dimension_inputs:
            self.dimension_inputs[field_name].set(str(value))
        self._last_preview_signature = None
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
                'Invalid warp dimension',
                f'{field_name} must be a whole number from {minimum} to '
                f'{maximum}.',
            )
            input_variable.set(str(self.pending_values[field_name]))
            return
        self.variables[field_name].set(value)
        self.pending_values[field_name] = value
        self._last_preview_signature = None
        self.refresh_preview()

    def _schedule_update(self) -> None:
        if self.closed:
            return
        rclpy.spin_once(self.node, timeout_sec=0.0)
        self.refresh_preview()
        self.root.after(15, self._schedule_update)

    def refresh_preview(self) -> None:
        live_frame = self.node.latest_frame
        frame = self.frozen_frame if self.frozen_frame is not None else live_frame
        if frame is None:
            self.status.configure(
                text=f'Waiting for camera frames on {self.node.camera_topic}'
            )
            return
        signature = (
            self.node.latest_sequence if self.frozen_frame is None else 'paused',
            tuple(self.pending_values.items()),
            int(self._frame_age() * 10),
        )
        if signature == self._last_preview_signature:
            return
        try:
            config = road_warp_from_mapping(self.pending_values)
            overlay = draw_road_warp_overlay(frame, config)
            warped = warp_road_image(frame, config)
            self._original_photo = _photo_for_panel(overlay, (760, 390))
            self._warped_photo = _photo_for_panel(warped, (760, 330))
            self.original_label.configure(image=self._original_photo)
            self.warped_label.configure(image=self._warped_photo)
            mode = 'PAUSED' if self.frozen_frame is not None else 'LIVE'
            state = 'UNSAVED preview' if config != self.saved else 'saved'
            age = self._frame_age()
            freshness = f'{age:.2f}s old'
            if age > 1.0:
                freshness = f'STALE ({freshness})'
            self.status.configure(
                text=(
                    f'{mode} | {state} | frame {freshness}\n'
                    f'Camera: {self.node.camera_topic}\n'
                    f'Output: {self.node.output_path}'
                )
            )
            self._last_preview_signature = signature
        except (TypeError, ValueError) as exc:
            self.status.configure(text=f'Invalid preview: {exc}')

    def _frame_age(self) -> float:
        received = self.node.latest_frame_monotonic
        if received is None:
            return 0.0
        return max(0.0, time.monotonic() - received)

    def save(self) -> None:
        try:
            config = road_warp_from_mapping(self.pending_values)
            saved_path = save_road_warp_config(self.node.output_path, config)
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror('Could not save warp YAML', str(exc))
            return
        self.saved = config
        self.pending_values = road_warp_values(config)
        self._last_preview_signature = None
        self.node.get_logger().info(f'Saved road warp config: {saved_path}')
        self.refresh_preview()

    def reset(self) -> None:
        self.pending_values = road_warp_values(self.saved)
        self._sync_controls_from_state()
        self._last_preview_signature = None
        self.refresh_preview()
        self.node.get_logger().info('Reset preview to the last saved values')

    def toggle_pause(self) -> None:
        if self.frozen_frame is not None:
            self.frozen_frame = None
            self.original_title.configure(text='LIVE original + road ROI')
            self.node.get_logger().info('Live preview resumed')
        elif self.node.latest_frame is not None:
            self.frozen_frame = self.node.latest_frame.copy()
            self.original_title.configure(text='PAUSED original + road ROI')
            self.node.get_logger().info('Preview paused on the current frame')
        self._last_preview_signature = None
        self.refresh_preview()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.root.destroy()


def _photo_for_panel(
    bgr_image: np.ndarray,
    bounds: tuple[int, int],
) -> ImageTk.PhotoImage:
    rgb_image = np.ascontiguousarray(bgr_image[:, :, ::-1])
    image = PilImage.fromarray(rgb_image)
    resampling = getattr(PilImage, 'Resampling', PilImage)
    image.thumbnail(bounds, resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node: LiveWarpTunerNode | None = None
    try:
        node = LiveWarpTunerNode()
        application = LiveWarpTunerApplication(node)
        application.run()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
