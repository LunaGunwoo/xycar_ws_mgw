"""Tune road-warp parameters from the live ROS front-camera topic."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from xycar_ai_drive.artifact import RoadWarpParameters
from xycar_ai_drive.road_warp import (
    draw_road_warp_overlay,
    load_road_warp_config,
    road_warp_from_mapping,
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
PREVIEW_WINDOW = 'Xycar live road-warp preview'
CONTROL_WINDOW = 'Xycar road-warp controls'
PANEL_WIDTH = 640
PANEL_HEIGHT = 480
HEADER_HEIGHT = 92

# Trackbar value = position / scale. Integer dimensions use scale 1.
TRACKBAR_SPECS: dict[str, tuple[float, float, int]] = {
    'top_y': (0.0, 1.0, 1000),
    'bottom_y': (0.0, 1.0, 1000),
    'top_left_x': (0.0, 1.0, 1000),
    'top_right_x': (0.0, 1.0, 1000),
    'bottom_left_x': (0.0, 1.0, 1000),
    'bottom_right_x': (0.0, 1.0, 1000),
    'bev_width': (80.0, 1920.0, 1),
    'bev_height': (60.0, 1080.0, 1),
    'dst_left_x': (0.0, 0.49, 1000),
    'dst_right_x': (0.51, 1.0, 1000),
}


class LiveWarpTunerNode(Node):
    """ROS subscriber with an OpenCV GUI and explicit config saving."""

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
        self._saved = load_road_warp_config(load_path)
        self._bridge = CvBridge()
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_monotonic: float | None = None
        self._frozen_frame: np.ndarray | None = None
        self._last_error = ''
        self._running = True

        self._create_windows(self._saved)
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

    @property
    def running(self) -> bool:
        return self._running

    def _create_windows(self, config: RoadWarpParameters) -> None:
        cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            PREVIEW_WINDOW,
            PANEL_WIDTH * 2,
            PANEL_HEIGHT + HEADER_HEIGHT,
        )
        cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CONTROL_WINDOW, 520, 680)
        for name, (minimum, maximum, scale) in TRACKBAR_SPECS.items():
            label = _trackbar_label(name, scale)
            value = float(getattr(config, name))
            position = int(round(value * scale))
            limit = int(round(maximum * scale))
            cv2.createTrackbar(
                label,
                CONTROL_WINDOW,
                position,
                limit,
                lambda _value: None,
            )
            cv2.setTrackbarMin(
                label,
                CONTROL_WINDOW,
                int(round(minimum * scale)),
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
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_frame_monotonic = time.monotonic()

    def render_once(self) -> None:
        frame, frame_age = self._display_frame()
        if frame is None:
            canvas = np.zeros(
                (PANEL_HEIGHT + HEADER_HEIGHT, PANEL_WIDTH * 2, 3),
                dtype=np.uint8,
            )
            _put_line(
                canvas,
                f'Waiting for camera frames on {self.camera_topic}',
                34,
                (0, 200, 255),
            )
            _put_line(
                canvas,
                'S save | R reset | Space pause/live | Q/Esc quit',
                68,
                (220, 220, 220),
            )
            cv2.imshow(PREVIEW_WINDOW, canvas)
            return

        try:
            config = self._pending_config()
            overlay = draw_road_warp_overlay(frame, config)
            warped = warp_road_image(frame, config)
            self._last_error = ''
        except ValueError as exc:
            config = None
            overlay = frame.copy()
            warped = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
            self._last_error = str(exc)

        left = _fit_panel(overlay)
        right = _fit_panel(warped)
        canvas = np.zeros(
            (PANEL_HEIGHT + HEADER_HEIGHT, PANEL_WIDTH * 2, 3),
            dtype=np.uint8,
        )
        canvas[HEADER_HEIGHT:, :PANEL_WIDTH] = left
        canvas[HEADER_HEIGHT:, PANEL_WIDTH:] = right
        cv2.line(
            canvas,
            (PANEL_WIDTH, HEADER_HEIGHT),
            (PANEL_WIDTH, canvas.shape[0]),
            (90, 90, 90),
            1,
        )
        mode = 'PAUSED' if self._frozen_frame is not None else 'LIVE'
        changed = config is None or config != self._saved
        save_state = 'UNSAVED' if changed else 'saved'
        freshness = f'frame age {frame_age:.2f}s'
        if frame_age > 1.0:
            freshness = f'STALE {freshness}'
        _put_line(
            canvas,
            f'{mode} | {save_state} | {freshness}',
            28,
            (0, 220, 255) if changed else (120, 230, 120),
        )
        _put_line(
            canvas,
            'S save | R reset | Space pause/live | Q/Esc quit',
            58,
            (220, 220, 220),
        )
        detail = self._last_error or f'Output: {self.output_path}'
        _put_line(
            canvas,
            detail,
            84,
            (80, 80, 255) if self._last_error else (170, 170, 170),
            scale=0.48,
        )
        _put_line(canvas, 'Original + source ROI', HEADER_HEIGHT + 25)
        _put_line(
            canvas,
            'Warped BEV preview',
            HEADER_HEIGHT + 25,
            x=PANEL_WIDTH + 14,
        )
        cv2.imshow(PREVIEW_WINDOW, canvas)

    def handle_gui_events(self) -> None:
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            self._running = False
        elif key in (ord('s'), ord('S')):
            self._save()
        elif key in (ord('r'), ord('R')):
            self._reset()
        elif key == ord(' '):
            self._toggle_pause()
        try:
            visible = cv2.getWindowProperty(
                PREVIEW_WINDOW,
                cv2.WND_PROP_VISIBLE,
            )
        except cv2.error:
            visible = 0.0
        if visible < 1.0:
            self._running = False

    def _display_frame(self) -> tuple[np.ndarray | None, float]:
        with self._frame_lock:
            latest = self._latest_frame
            received = self._latest_frame_monotonic
        selected = self._frozen_frame if self._frozen_frame is not None else latest
        if selected is None or received is None:
            return None, 0.0
        return selected, max(0.0, time.monotonic() - received)

    def _pending_config(self) -> RoadWarpParameters:
        values: dict[str, float | int] = {}
        for name, (_minimum, _maximum, scale) in TRACKBAR_SPECS.items():
            position = cv2.getTrackbarPos(
                _trackbar_label(name, scale),
                CONTROL_WINDOW,
            )
            value = position / scale
            values[name] = int(round(value)) if scale == 1 else float(value)
        return road_warp_from_mapping(values)

    def _save(self) -> None:
        try:
            config = self._pending_config()
            saved_path = save_road_warp_config(self.output_path, config)
        except (OSError, ValueError) as exc:
            self._last_error = f'Save failed: {exc}'
            self.get_logger().error(self._last_error)
            return
        self._saved = config
        self._last_error = ''
        self.get_logger().info(f'Saved road warp config: {saved_path}')

    def _reset(self) -> None:
        for name, (_minimum, _maximum, scale) in TRACKBAR_SPECS.items():
            value = float(getattr(self._saved, name))
            position = int(round(value * scale))
            cv2.setTrackbarPos(
                _trackbar_label(name, scale),
                CONTROL_WINDOW,
                position,
            )
        self._last_error = ''
        self.get_logger().info('Reset preview to the last saved values')

    def _toggle_pause(self) -> None:
        if self._frozen_frame is not None:
            self._frozen_frame = None
            self.get_logger().info('Live preview resumed')
            return
        with self._frame_lock:
            if self._latest_frame is not None:
                self._frozen_frame = self._latest_frame.copy()
        if self._frozen_frame is not None:
            self.get_logger().info('Preview paused on the current frame')


def _fit_panel(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(PANEL_WIDTH / width, PANEL_HEIGHT / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    x = (PANEL_WIDTH - resized_width) // 2
    y = (PANEL_HEIGHT - resized_height) // 2
    panel[y:y + resized_height, x:x + resized_width] = resized
    return panel


def _trackbar_label(name: str, scale: int) -> str:
    return f'{name} x{scale}' if scale != 1 else name


def _put_line(
    image: np.ndarray,
    text: str,
    y: int,
    color: tuple[int, int, int] = (255, 255, 255),
    *,
    x: int = 14,
    scale: float = 0.62,
) -> None:
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node: LiveWarpTunerNode | None = None
    try:
        node = LiveWarpTunerNode()
        while rclpy.ok() and node.running:
            rclpy.spin_once(node, timeout_sec=0.01)
            node.render_once()
            node.handle_gui_events()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
