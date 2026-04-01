import math
import numpy as np


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class PIDGoToController:
    """
    PID controller for the GoTo task.

    Same interface as TDMPC2GoToController:
        get_action(obs) -> (v_cmd, w_cmd)
    where obs = [surge, yaw_rate, cos(bearing), sin(bearing), dist].
    """

    _SUCCESS_THRESH = 0.15  # metres — consistent with TDMPC2
    _V_MAX = 0.22           # m/s
    _W_MAX = 2.84           # rad/s

    def __init__(self, dt):
        self.dt = dt
        self.pid_w = PIDController(kp=2.0, ki=0.0, kd=0.1)
        self.pid_v = PIDController(kp=0.5, ki=0.0, kd=0.0)

    def get_action(self, obs):
        """
        Args:
            obs: [surge, yaw_rate, cos(bearing), sin(bearing), dist]  (5,)

        Returns:
            (v_cmd, w_cmd) in physical units (m/s, rad/s)
        """
        cos_b = obs[2]
        sin_b = obs[3]
        dist = obs[4]

        bearing = math.atan2(sin_b, cos_b)

        if dist < self._SUCCESS_THRESH:
            self.pid_v.reset()
            self.pid_w.reset()
            return 0.0, 0.0

        w = self.pid_w.compute(bearing, self.dt)
        v = self.pid_v.compute(dist, self.dt)

        if abs(bearing) > 0.1:
            v = 0.0
        else:
            v = v * math.cos(bearing)

        return np.clip(v, -self._V_MAX, self._V_MAX), np.clip(w, -self._W_MAX, self._W_MAX)