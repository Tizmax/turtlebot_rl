import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from turtlebot_rl.TDMPC2Controller import TDMPC2GoToController


class GoToNode(Node):
    def __init__(self):
        super().__init__("goto_node")

        self.cmd_pub = self.create_publisher(Twist, "/commands/velocity", 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

        self.dt = 0.01
        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v_actual = 0.0
        self.w_actual = 0.0
        self.odom_fresh = False

        self.x_goal = 0.5
        self.y_goal = 0.5

        self.controller = TDMPC2GoToController(dt=self.dt)
        self.get_logger().info("GoTo node ready, waiting for odometry...")

    def quaternion_to_yaw(self, x, y, z, w):
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.theta = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.v_actual = msg.twist.twist.linear.x
        self.w_actual = msg.twist.twist.angular.z
        self.odom_fresh = True

    def timer_callback(self):
        if not self.odom_fresh:
            return
        self.odom_fresh = False  # only act on fresh odometry

        # Body-frame goal
        dx_w = self.x_goal - self.x
        dy_w = self.y_goal - self.y
        c, s = math.cos(self.theta), math.sin(self.theta)
        dx_b = c * dx_w + s * dy_w
        dy_b = -s * dx_w + c * dy_w

        dist = math.hypot(dx_b, dy_b)
        bearing = math.atan2(dy_b, dx_b)

        obs = [self.v_actual, self.w_actual, math.cos(bearing), math.sin(bearing), dist]

        v, w = self.controller.get_action(obs)

        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)

        if v == 0.0 and w == 0.0:
            self.get_logger().info("Goal reached!", once=True)


def main(args=None):
    rclpy.init(args=args)
    node = GoToNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        stop_twist = Twist()
        node.cmd_pub.publish(stop_twist)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
