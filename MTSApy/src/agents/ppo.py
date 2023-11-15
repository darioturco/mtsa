from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from src.agents.agent import Agent
import numpy as np
class PPO(Agent):
    def __init__(self, env, args):
        def mask_fn(envi):
            # Do whatever you'd like in this function to return the action mask
            # for the current env. In this example, we assume the env has a
            # helpful method we can rely on.
            return envi.valid_action_mask()

        self.args = args
        self.batch_size = args["batch_size"]
        self.original_env = env
        self.env = ActionMasker(env, mask_fn)  # Wrap to enable masking
        self.model = MaskablePPO(MaskableActorCriticPolicy,
                                self.env,
                                learning_rate=args["learning_rate"],
                                gamma=args["gamma"],
                                 batch_size=args["batch_size"],
                                verbose=args["verbose"])
        self.callback = PPOCallback(self.batch_size)

        #policy: Union[str, Type[MaskableActorCriticPolicy]],
        #env: Union[GymEnv, str],
        #learning_rate: Union[float, Schedule] = 3e-4,
        #n_steps: int = 2048,
        #batch_size: Optional[int] = 64,
        #n_epochs: int = 10,
        #gamma: float = 0.99,
        #gae_lambda: float = 0.95,
        #clip_range: Union[float, Schedule] = 0.2,
        #clip_range_vf: Union[None, float, Schedule] = None,
        #normalize_advantage: bool = True,
        #ent_coef: float = 0.0,
        #vf_coef: float = 0.5,
        #max_grad_norm: float = 0.5,
        #target_kl: Optional[float] = None,
        #stats_window_size: int = 100,
        #tensorboard_log: Optional[str] = None,
        #policy_kwargs: Optional[Dict[str, Any]] = None,
        #verbose: int = 0,
        #seed: Optional[int] = None,
        #device: Union[th.device, str] = "auto",
        #_init_setup_model: bool = True,



    def fit_callback(self, obj):
        print(obj)

    def train(self, total_timesteps=1000):
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback)

    def get_action(self, state, *args, **kwargs):
        state = np.array(state)
        mask = self.original_env.valid_action_mask()
        return self.model.predict(state, action_masks=mask)[0]

    @classmethod
    def load(cls, env, path, args):
        agent = cls(env, args)
        agent.model = MaskablePPO.load(path)
        return agent

class PPOCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, batch_size, verbose=0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.count = 0
        self.epoachs = 0
        self.batch_size = batch_size

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.count += 1
        self.epoachs += self.batch_size

        # Escribir el csv

        if self.count % 5 == 0:

            # Guardar el modelo
            print("Save Model")
            instance, n, k = self.locals["env"].envs[0].env.env.get_instance_info()
            self.model.save(f"./results/models/PPO/{instance}/{instance}-{n}-{k}-{self.epoachs}-partial")



        print("Rollout Ends")

    def _on_training_end(self) -> None:
        pass