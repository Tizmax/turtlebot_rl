import math
import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.integral = 0.0; self.prev_error = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

class PIDGoToController:
    def __init__(self, dt):
        self.dt = dt
        self.pid_w = PIDController(kp=2.0, ki=0.0, kd=0.1)
        self.pid_v = PIDController(kp=0.5, ki=0.0, kd=0.0)
        self.max_v = 0.22  
        self.max_w = 2.84  

    def get_action(self, x, y, theta, x_g, y_g):
        distance_error = math.hypot(x_g - x, y_g - y)
        angle_to_goal = math.atan2(y_g - y, x_g - x)
        angle_error = math.atan2(math.sin(angle_to_goal - theta), math.cos(angle_to_goal - theta))

        if distance_error < 0.05: 
            return 0.0, 0.0

        w = self.pid_w.compute(angle_error, self.dt)
        v = self.pid_v.compute(distance_error, self.dt)

        # Logique stricte : on s'aligne d'abord (à 5° près)
        if abs(angle_error) > 0.1:
            v = 0.0 
        else:
            v = v * math.cos(angle_error)

        return np.clip(v, -self.max_v, self.max_v), np.clip(w, -self.max_w, self.max_w)
