import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import numpy as np
from turtlebot_rl.PIDController import PIDGoToController
from turtlebot_rl.TDMPC2Controller import TDMPC2GoToController

class GoToNode(Node):
    def __init__(self):
        super().__init__('goto_node')
        
        # Publisher pour envoyer les commandes aux moteurs
        self.cmd_pub = self.create_publisher(Twist, '/commands/velocity', 10)
        
        # Subscriber pour lire la position du robot
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Timer (10 Hz = dt de 0.1s) pour cadencer le calcul du PID
        self.dt = 0.1
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        # self.controller = PIDGoToController(dt=self.dt)
        self.controller = TDMPC2GoToController(dt=self.dt)
        
        # Variables d'état du robot
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # Cible (Vous pourrez la modifier dynamiquement plus tard)
        self.x_goal = 2.0
        self.y_goal = 2.0
        
        self.get_logger().info("Nœud GoTo initialisé. En attente de l'odométrie...")

    def quaternion_to_yaw(self, x, y, z, w):
        """Convertit un quaternion en angle de lacet (yaw/theta) autour de l'axe Z."""
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def odom_callback(self, msg):
        """Mise à jour de la position à chaque message de l'odométrie."""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.theta = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

    def timer_callback(self):
        """Boucle de contrôle principale."""
        # Calcul de la vitesse
        v, w = self.controller.get_action(self.x, self.y, self.theta, self.x_goal, self.y_goal)
        
        # Création et envoi du message
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)
        
        if v == 0.0 and w == 0.0:
            self.get_logger().info('Cible atteinte !', once=True)

def main(args=None):
    rclpy.init(args=args)
    node = GoToNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Arrêt demandé par l'utilisateur.")
    finally:
        # SÉCURITÉ : On arrête le robot avant de fermer le programme
        stop_twist = Twist()
        node.cmd_pub.publish(stop_twist)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()