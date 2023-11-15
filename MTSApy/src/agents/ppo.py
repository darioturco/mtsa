import numpy as np
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.agents.agent import Agent



class PPO(Agent):
    def __init__(self, env, args):
        def mask_fn(envi):
            return envi.valid_action_mask()

        self.args = args
        self.batch_size = args["batch_size"]
        self.original_env = env
        self.env = ActionMasker(env, mask_fn)  # Wrap to enable masking
        self.model = self.init_model()
        self.callback = PPOCallback(self.batch_size)

    def init_model(self):
        return MaskablePPO(MaskableActorCriticPolicy,
                            self.env,
                            learning_rate=self.args["learning_rate"],
                            gamma=self.args["gamma"],
                            batch_size=self.args["batch_size"],
                            verbose=self.args["verbose"])

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



class PPOCNN(PPO):
    def init_model(self):
        return MaskablePPO(MaskableActorCriticPolicy,
                            self.env,
                            learning_rate=self.args["learning_rate"],
                            gamma=self.args["gamma"],
                            batch_size=self.args["batch_size"],
                            verbose=self.args["verbose"],
                            policy_kwargs={"features_extractor_class": CCNExtractor})



class PPOCallback(BaseCallback):
    def __init__(self, batch_size, verbose=0):
        super().__init__(verbose)
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

        # Write the csv file

        if self.count % 5 == 0:
            # Save the model
            print("Saved Model")
            instance, n, k = self.locals["env"].envs[0].env.env.get_instance_info()
            self.model.save(f"./results/models/PPO/{instance}/{instance}-{n}-{k}-{self.epoachs}-partial")

    def _on_training_end(self) -> None:
        pass



class CCNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, observation_space.shape[0])
        self.cnn = nn.Conv2d(1, 1, (1, observation_space.shape[1]))
        self.flatten = nn.Flatten()

    def forward(self, observations):
        observations = observations.reshape((-1, 1, 16, 28))
        x = self.cnn(observations)
        return self.flatten(x)