import numpy as np
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "tdmpc_utils"))

try:
    from tdmpc2 import TDMPC2
    from envs import make_env
except ImportError:
    from turtlebot_rl.tdmpc_utils.tdmpc2 import TDMPC2
    from turtlebot_rl.tdmpc_utils.envs import make_env

from omegaconf import OmegaConf
from common.parser import parse_cfg
import hydra
from hydra import compose, initialize_config_dir


class TDMPC2GoToController:
    """
    Deploys a TD-MPC2 policy trained in the TB2KobukiGoToEnv on the real robot.

    The agent outputs actions in [-1, 1] (tanh-bounded). We scale them by the
    same constants the training env uses so the physical twist matches what the
    agent experienced during training. No additional clipping is needed.
    """

    # Must match TB2KobukiGoToEnv exactly
    _WHEEL_RADIUS = 0.035  # metres
    _WHEELBASE = 0.230  # metres
    _MAX_WHEEL_VEL = 20.0  # rad/s
    # _V_LINEAR_MAX = _WHEEL_RADIUS * _MAX_WHEEL_VEL  # 0.7 m/s
    _V_LINEAR_MAX = 0.22
    # _OMEGA_MAX = 2.0 * _WHEEL_RADIUS * _MAX_WHEEL_VEL / _WHEELBASE  # ≈6.09 rad/s
    _OMEGA_MAX = 2.84 
    _SUCCESS_THRESH = 0.15  # metres

    def __init__(self, dt, model_path=None, config_path=None):
        self.dt = dt

        base = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(base, "tdmpc_utils", "final.pt")
            print(model_path)
        if config_path is None:
            config_path = os.path.join(base, "tdmpc_utils", "config.yaml")
            print(config_path)

        with initialize_config_dir(
            config_dir=os.path.dirname(os.path.abspath(config_path)),
            version_base=None,
        ):
            cfg = compose(
                config_name=os.path.basename(config_path).replace(".yaml", ""),
                overrides=[
                    "hydra/launcher=basic",
                    "task=tb2-kobuki-goto",
                    f"checkpoint={model_path}",
                ],
            )

        OmegaConf.set_struct(cfg, False)
        hydra.utils.get_original_cwd = lambda: os.getcwd()
        cfg = parse_cfg(cfg)

        # make_env populates cfg fields (obs/action dims, etc.) that TDMPC2 needs
        env = make_env(cfg)
        del env

        cfg.compile = False
        print("Initialising TD-MPC2 agent...")
        self.agent = TDMPC2(cfg)
        self.agent.load(model_path)
        self.agent.model.eval()
        print("TD-MPC2 model ready.")

        self.is_first_step = True

    def get_action(self, obs):
        """
        Args:
            obs: [surge, yaw_rate, cos(bearing), sin(bearing), dist]  (5,)

        Returns:
            (v_cmd, w_cmd) in physical units (m/s, rad/s)
        """
        dist = obs[4]

        if dist < self._SUCCESS_THRESH:
            self.is_first_step = True
            return 0.0, 0.0

        t_obs = torch.tensor(obs, dtype=torch.float32)
        action = self.agent.act(t_obs, t0=self.is_first_step)
        self.is_first_step = False

        v = float(action[0])# * self._V_LINEAR_MAX
        w = float(action[1])# * self._OMEGA_MAX

        return np.clip(v, -self._V_LINEAR_MAX, self._V_LINEAR_MAX), np.clip(w, -self._OMEGA_MAX, self._OMEGA_MAX)

