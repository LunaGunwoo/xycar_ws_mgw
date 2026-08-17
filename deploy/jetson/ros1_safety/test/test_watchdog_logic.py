import math
import unittest

from xycar_motor_safety.watchdog import MotorCommandWatchdog, STOP_COMMAND


class MotorCommandWatchdogTest(unittest.TestCase):
    def test_normalized_angle_mapping_and_speed_passthrough(self):
        watchdog = MotorCommandWatchdog(0.25)
        for normalized, driver in (
            (-100.0, -40.0),
            (-50.0, -20.0),
            (0.0, 0.0),
            (50.0, 20.0),
            (100.0, 40.0),
        ):
            self.assertTrue(watchdog.observe([normalized, 17.25], 1.0))
            self.assertEqual(watchdog.command(1.24), (driver, 17.25))

    def test_out_of_range_angle_invalidates_the_command(self):
        watchdog = MotorCommandWatchdog(0.25)
        for invalid_angle in (-100.0001, 100.0001):
            self.assertFalse(watchdog.observe([invalid_angle, 7.0], 1.0))
            self.assertEqual(watchdog.command(1.0), STOP_COMMAND)

    def test_stops_on_stale_invalid_or_reversed_time(self):
        watchdog = MotorCommandWatchdog(0.25)
        self.assertEqual(watchdog.command(1.0), STOP_COMMAND)
        self.assertTrue(watchdog.observe([1.0, 2.0], 1.0))
        self.assertEqual(watchdog.command(1.25), STOP_COMMAND)
        self.assertTrue(watchdog.observe([1.0, 2.0], 2.0))
        self.assertEqual(watchdog.command(1.9), STOP_COMMAND)
        self.assertEqual(watchdog.command(2.1), STOP_COMMAND)
        for invalid in (
            [1.0],
            [1.0, 2.0, 3.0],
            [math.nan, 2.0],
            [1.0, math.inf],
        ):
            self.assertFalse(watchdog.observe(invalid, 3.0))
            self.assertEqual(watchdog.command(3.0), STOP_COMMAND)

    def test_rejects_a_different_steering_contract(self):
        with self.assertRaisesRegex(ValueError, 'normalized_percent_v1'):
            MotorCommandWatchdog(0.25, driver_angle_max=50.0)


if __name__ == '__main__':
    unittest.main()
