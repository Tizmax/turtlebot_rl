import math
import numpy as np
import torch
import os

# Assurez-vous que le dossier tdmpc2 est dans votre PYTHONPATH
# ou importez-le correctement selon votre arborescence.
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'tdmpc_utils'))
try:
    from tdmpc2 import TDMPC2
except ImportError:
    from turtlebot_rl.tdmpc_utils.tdmpc2 import TDMPC2
from omegaconf import OmegaConf # Probablement utilisé par tdmpc2 pour la config
from common.parser import parse_cfg
import hydra
class BaseGoToController:
    def __init__(self, dt):
        self.dt = dt
        self.max_v = 0.22
        self.max_w = 2.84

    def get_action(self, x, y, theta, x_g, y_g):
        raise NotImplementedError()

class TDMPC2GoToController(BaseGoToController):
    def __init__(self, dt, model_path=None, config_path=None):
        super().__init__(dt)
        
        # Default paths to tdmpc_utils files
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "tdmpc_utils",
                "150034.pt"
            )
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "tdmpc_utils",
                "config.yaml"
            )
        
        # 1. Charger la configuration
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")
        self.cfg = OmegaConf.load(config_path)

        #===============================================================================
        # On déverrouille l'objet pour avoir le droit de modifier
        OmegaConf.set_struct(self.cfg, False)

        hydra.utils.get_original_cwd = lambda: os.getcwd()

        self.cfg = parse_cfg(self.cfg)

        # 1. On remplace les ??? par les dimensions de notre Turtlebot
        self.cfg.obs_shape = {"state": [5]}  # dist, cos, sin, v, w
        self.cfg.action_dim = 2              # v et w
        
        self.cfg.num_q = 2  # On s'aligne avec le fichier .pt !

        # 2. ALIGNEMENT DE L'ARCHITECTURE AVEC LE FICHIER .PT (La correction est ici)
        self.cfg.multitask = False 
        self.cfg.episodic = False
        
        # 3. Sécurités pour le Sim-to-Real
        self.cfg.compile = False
        
        #===============================================================================

        # 2. Instancier l'agent
        print("Initialisation de l'agent TD-MPC2...")
        self.agent = TDMPC2(self.cfg)
        
        # 3. Charger les poids
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Fichier de poids introuvable : {model_path}")
        print(f"Chargement des poids depuis {model_path}...")
        self.agent.load(model_path)
        
        #   L'agent.load() met probablement deja le modele en eval, mais on assure :
        self.agent.model.eval()
        
        # Indicateur pour la méthode act() (t0 = True au premier step)
        self.is_first_step = True
        
        # Variables pour stocker l'état précédent afin de calculer la vitesse
        self.prev_x = None
        self.prev_y = None
        self.prev_theta = None
        print("Modèle TD-MPC2 prêt !")

    def get_action(self, x, y, theta, x_g, y_g):
        # 1. Calcul des vitesses réelles (Estimées depuis l'odométrie)
        if self.prev_x is None:
            # Au tout premier step, on assume que le robot est à l'arrêt
            v_actuel = 0.0
            w_actuel = 0.0
        else:
            dx = x - self.prev_x
            dy = y - self.prev_y
            
            # Projection pour obtenir la vitesse linéaire signée
            v_actuel = (dx * math.cos(theta) + dy * math.sin(theta)) / self.dt
            
            # Différence d'angle normalisée
            dtheta = theta - self.prev_theta
            dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta))
            w_actuel = dtheta / self.dt

        # Sauvegarde pour la prochaine itération
        self.prev_x = x
        self.prev_y = y
        self.prev_theta = theta

        # 2. Calcul des erreurs par rapport à la cible
        distance_error = math.hypot(x_g - x, y_g - y)
        angle_to_goal = math.atan2(y_g - y, x_g - x)
        angle_error = math.atan2(math.sin(angle_to_goal - theta), math.cos(angle_to_goal - theta))

        if distance_error < 0.05:
            self.is_first_step = True
            self.prev_x = None # Reset pour le prochain trajet
            return 0.0, 0.0

        # 3. L'OBSERVATION COMPLÈTE
        # /!\ L'ordre doit être strictement identique à celui de l'entraînement
        obs_numpy = np.array([
            v_actuel,
            w_actuel,
            math.cos(angle_error),
            math.sin(angle_error),
            distance_error
        ], dtype=np.float32)

        obs_tensor = torch.from_numpy(obs_numpy)

        # 4. Inférence TD-MPC2
        with torch.no_grad():
            action_tensor = self.agent.act(obs_tensor, t0=self.is_first_step, eval_mode=True)
            
        self.is_first_step = False

        # 5. Extraction et dénormalisation
        action = action_tensor.numpy()
        v_cmd = action[0] * self.max_v
        w_cmd = action[1] * self.max_w

        return np.clip(v_cmd, -self.max_v, self.max_v), np.clip(w_cmd, -self.max_w, self.max_w)