import numpy as np
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "ppo"))

try:
    from ppo_agent import PPOAgent
    from envs import make_env
except ImportError:
    from turtlebot_rl.ppo.ppo_agent import PPOAgent
    from turtlebot_rl.tdmpc_utils.envs import make_env

from omegaconf import OmegaConf
from common.parser import parse_cfg
import hydra
from hydra import compose, initialize_config_dir



class PPOGoToController:
    """
    Deploys a PPO policy trained in the TB2KobukiGoToEnv on the real robot.

    The agent outputs actions in [-1, 1] (tanh-bounded). We scale them by the
    same constants the training env uses so the physical twist matches what the
    agent experienced during training.

    Best checkpoint: ppo-v2, step 550k
        /home/GTL/asave/ppo_logs/tb2-kobuki-goto/1/ppo-v2/models/550000.pt
    """

    # Must match TB2KobukiGoToEnv exactly
    _WHEEL_RADIUS   = 0.035   # metres
    _WHEELBASE      = 0.230   # metres
    _MAX_WHEEL_VEL  = 20.0    # rad/s
    _V_LINEAR_MAX   = 0.22    # m/s  (real robot limit)
    _OMEGA_MAX      = 2.84    # rad/s (real robot limit)
    _SUCCESS_THRESH = 0.15    # metres — hard stop inside this radius

    def __init__(self, dt, model_path=None, config_path=None):
        self.dt = dt
        self.is_first_step = True

        base = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(base, "ppo", "ppo.pt")
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

        # make_env populates cfg fields (obs/action dims, etc.) that PPOAgent needs
        env = make_env(cfg)
        del env

        print("Initialising PPO agent...")
        self.agent = PPOAgent(cfg).to("cuda")
        self.agent.load(model_path, device="cuda")
        self.agent.eval()
        print("PPO model ready.")

    def reset(self):
        """Call at the start of each new navigation goal."""
        self.is_first_step = True

    def get_action(self, obs):
        """
        Args:
            obs: [surge, yaw_rate, cos(bearing), sin(bearing), dist]  (5,)

        Returns:
            (v_cmd, w_cmd) in physical units (m/s, rad/s)
        """
        dist = float(obs[4])

        if dist < self._SUCCESS_THRESH:
            self.is_first_step = True
            return 0.0, 0.0

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to("cuda")
        with torch.no_grad():
            action = self.agent.act(obs_t, eval_mode=True)  # CPU tensor in [-1, 1]
        self.is_first_step = False

        v = float(action[0]) * self._V_LINEAR_MAX
        w = float(action[1]) * self._OMEGA_MAX

        return (
            float(np.clip(v, -self._V_LINEAR_MAX, self._V_LINEAR_MAX)),
            float(np.clip(w, -self._OMEGA_MAX,    self._OMEGA_MAX)),
        )