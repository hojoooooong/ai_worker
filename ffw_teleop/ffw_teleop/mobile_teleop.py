#!/usr/bin/env python3
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import select
import sys
import termios
import tty

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node


# Movement key bindings: (linear.x, linear.y)
# Layout:
#   Q W E      ↖ ↑ ↗
#   A   D  ->  ←   →
#   Z X C      ↙ ↓ ↘
MOVE_BINDINGS = {
    'w': (1.0, 0.0),     # Forward
    'x': (-1.0, 0.0),    # Backward
    'a': (0.0, 1.0),     # Left
    'd': (0.0, -1.0),    # Right
    'q': (1.0, 1.0),     # Forward-Left
    'e': (1.0, -1.0),    # Forward-Right
    'z': (-1.0, 1.0),    # Backward-Left
    'c': (-1.0, -1.0),   # Backward-Right
}

# Rotation key bindings: angular.z only
#   H / J  ->  ↺ / ↻
ROTATE_BINDINGS = {
    'h': 1.0,   # Rotate Left
    'j': -1.0,  # Rotate Right
}


class KeyboardTeleop(Node):

    def __init__(self):
        super().__init__('keyboard_teleop')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 5)

        self.linear_speed = 0.4    # m/s
        self.lateral_speed = 0.4   # m/s
        self.angular_speed = 0.8   # rad/s

        # Current velocity state
        self.linear_x = 0.0
        self.linear_y = 0.0
        self.angular_z = 0.0

        # Terminal settings
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)

        # Timer for publishing at 20Hz
        self.timer = self.create_timer(0.05, self.publish_twist)

        self.get_logger().info(
            '\nKeyboard Teleop (Joystick Mode)\n'
            '--------------------------------\n'
            '   Q W E      ↖ ↑ ↗\n'
            '   A   D  ->  ←   →\n'
            '   Z X C      ↙ ↓ ↘\n'
            '--------------------------------\n'
            '   H / J  ->  ↺ / ↻ (rotate)\n'
            '--------------------------------\n'
            'Hold key for continuous movement\n'
            'Space: Stop\n'
            'Ctrl+C: Quit\n'
        )

    def get_key_nonblocking(self):
        """Read key if available (non-blocking)."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def publish_twist(self):
        """Publish twist based on current velocity state."""
        twist = Twist()
        twist.linear.x = self.linear_x
        twist.linear.y = self.linear_y
        twist.angular.z = self.angular_z
        self.publisher.publish(twist)

    def run(self):
        try:
            tty.setraw(self.fd)

            while rclpy.ok():
                key = self.get_key_nonblocking()

                if key:
                    key_lower = key.lower()

                    if key_lower in MOVE_BINDINGS:
                        # Movement keys: control linear.x and linear.y only
                        direction = MOVE_BINDINGS[key_lower]
                        self.linear_x = direction[0] * self.linear_speed
                        self.linear_y = direction[1] * self.lateral_speed
                    elif key_lower in ROTATE_BINDINGS:
                        # Rotation keys: control angular.z only
                        self.angular_z = ROTATE_BINDINGS[key_lower] * self.angular_speed
                    elif key == ' ':
                        # Space: stop all
                        self.linear_x = 0.0
                        self.linear_y = 0.0
                        self.angular_z = 0.0
                    elif key == '\x03':  # Ctrl+C
                        break
                else:
                    # No key pressed: stop (release behavior)
                    self.linear_x = 0.0
                    self.linear_y = 0.0
                    self.angular_z = 0.0

                rclpy.spin_once(self, timeout_sec=0.05)

        finally:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

        # Stop robot on exit
        self.publisher.publish(Twist())
        self.get_logger().info('Teleop stopped.')


def main():
    rclpy.init()
    node = KeyboardTeleop()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
