#!/usr/bin/env python3

import time

import rospy
from std_msgs.msg import Float32MultiArray

from xycar_motor_safety.watchdog import MotorCommandWatchdog


class MotorWatchdogNode:
    def __init__(self) -> None:
        timeout_sec = float(rospy.get_param('~timeout_sec', 0.25))
        publish_rate_hz = float(rospy.get_param('~publish_rate_hz', 20.0))
        self._stop_publish_count = int(
            rospy.get_param('~stop_publish_count', 5)
        )
        if publish_rate_hz <= 0.0:
            raise ValueError('publish_rate_hz must be positive')
        if self._stop_publish_count < 1:
            raise ValueError('stop_publish_count must be positive')
        self._watchdog = MotorCommandWatchdog(
            timeout_sec,
            input_angle_min=float(
                rospy.get_param('~input_angle_min', -100.0)
            ),
            input_angle_max=float(
                rospy.get_param('~input_angle_max', 100.0)
            ),
            driver_angle_min=float(
                rospy.get_param('~driver_angle_min', -50.0)
            ),
            driver_angle_max=float(
                rospy.get_param('~driver_angle_max', 50.0)
            ),
        )
        self._publisher = rospy.Publisher(
            '/xycar_motor_safe',
            Float32MultiArray,
            queue_size=1,
        )
        self._subscriber = rospy.Subscriber(
            '/xycar_motor',
            Float32MultiArray,
            self._on_command,
            queue_size=1,
        )
        self._timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / publish_rate_hz),
            self._on_timer,
        )
        rospy.on_shutdown(self._publish_stop_burst)

    def _on_command(self, message: Float32MultiArray) -> None:
        if not self._watchdog.observe(message.data, time.monotonic()):
            rospy.logwarn_throttle(
                1.0,
                'Rejected invalid normalized /xycar_motor command; '
                'publishing stop.',
            )

    def _on_timer(self, _event) -> None:
        self._publish(self._watchdog.command(time.monotonic()))

    def _publish(self, command) -> None:
        message = Float32MultiArray()
        message.data = [float(command[0]), float(command[1])]
        self._publisher.publish(message)

    def _publish_stop_burst(self) -> None:
        for _ in range(self._stop_publish_count):
            self._publish((0.0, 0.0))
            rospy.sleep(0.05)


def main() -> None:
    rospy.init_node('xycar_motor_watchdog')
    MotorWatchdogNode()
    rospy.spin()


if __name__ == '__main__':
    main()
