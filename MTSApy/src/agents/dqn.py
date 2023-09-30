"""
from src.agents import Agent

class DQN(Agent):

    def __init__(self, env):
        super().__init__(env)
        # Guardar el resto de parametros
        # Initialize all

    def train(self, epochs):


    def predict(self, state):
        pass
"""

import torch
import torch.nn as nn
import json
import time
import numpy as np
import os

from src.agents.replay_buffer import ReplayBuffer
from src.agents.agent import Agent

class Model:
    def __init__(self):
        pass

    def predict(self, s):
        raise NotImplementedError()

    def eval_batch(self, obss):
        raise NotImplementedError()

    def eval(self, s):
        raise NotImplementedError()

    def best(self, s):
        raise NotImplementedError()

    def current_loss(self):
        raise NotImplementedError()


class OnnxModel(Model):
    def __init__(self, model):
        super().__init__()
        assert model.has_learned_something

        self.onnx_model, self.session = model.to_onnx()

    def save(self, path):
        onnx.save(self.onnx_model, path + ".onnx")

    def predict(self, s):
        if s is None:
            return 0
        return self.session.run(None, {'X': s})[0]

    def eval_batch(self, ss):
        return np.array([self.eval(s) for s in ss])

    def eval(self, s):
        return np.max(self.predict(s))

    def current_loss(self):
        raise NotImplementedError()



class TorchModel(Model):
    def __init__(self, nfeatures, network, args):
        super().__init__()
        self.nfeatures = nfeatures
        self.n, self.k = None, None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = network
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=args["learning_rate"],
                                         momentum=args["momentum"],
                                         nesterov=args["nesterov"],
                                         weight_decay=args["weight_decay"])

        self.has_learned_something = False
        self.losses = []

        # print("Using", self.device, "device")
        # print(self.model)
        # print("Learning rate:", args["learning_rate"])

    def eval_batch(self, ss):
        return np.array([self.eval(s) for s in ss])

    def eval(self, s):
        if not self.has_learned_something or s is None:
            return 0
        return float(self.predict(s).max())

    def best(self, s):
        if not self.has_learned_something or s is None:
            return 0
        return int(self.predict(s).argmax())

    def predict(self, s):
        if not self.has_learned_something or s is None:
            return 0
        return self.model(torch.tensor(s).to(self.device))

    def single_update(self, s, value):
        return self.batch_update(np.array([s]), np.array([value]))

    def batch_update(self, ss, values):

        ss = torch.tensor(ss).to(self.device)
        values = torch.tensor(values, dtype=torch.float, device=self.device).reshape(len(ss), 1)

        self.optimizer.zero_grad()
        pred = self.model(ss)

        loss = self.loss_fn(pred, values)
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        self.has_learned_something = True



    def nfeatures(self):
        return self.nfeatures

    # should be called only at the end of each episode
    def current_loss(self):
        avg_loss = np.mean(self.losses)
        self.losses = []
        return avg_loss


class NeuralNetwork(nn.Module):
    def __init__(self, nfeatures, nnsize):
        super(NeuralNetwork, self).__init__()
        nnsize = list(nnsize) + [1]
        layers = [nn.Linear(nfeatures, nnsize[0])]
        for i in range(len(nnsize)-1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(nnsize[i], nnsize[i+1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x.to(torch.float))
        return x

    def reuse_onnx_model(self, onnx_path):
        raise NotImplementedError

class DQN(Agent):
    def __init__(self, env, nn_model, args, save_file=None, verbose=False):
        assert nn_model is not None

        self.env = env
        self.args = args
        self.model = nn_model

        self.target = None
        self.buffer = None

        self.save_file = save_file
        self.save_idx = 0

        self.training_start = None
        self.training_steps = 0
        self.epsilon = args["first_epsilon"]

        self.verbose = verbose

        self.training_data = []

        self.best_training_perf = {}
        self.last_best = None
        self.converged = False
        self.initializeBuffer()

    def initializeBuffer(self):
        """ Initialize replay buffer uniformly with experiences """
        exp_per_instance = self.args["buffer_size"]

        print(f"Initializing buffer with {exp_per_instance} observations...")

        self.buffer = ReplayBuffer( self.env, self.args["buffer_size"])
        random_experience = self.buffer.get_experience_from_random_policy(total_steps=exp_per_instance, nstep=self.args["n_step"])
        for action_features, reward, obs2 in random_experience:
            self.buffer.add(action_features, reward, obs2)



        print("Done.")

    def train(self, seconds=None, max_steps=None, max_eps=None, save_freq=200000, last_obs=None,
              early_stopping=False, save_at_end=False, results_path=None, top=1000):

        if self.training_start is None:
            self.training_start = time.time()
            self.last_best = 0

        steps, eps = 0, 0
        epsilon_step = (self.args["first_epsilon"] - self.args["last_epsilon"])
        epsilon_step /= self.args["epsilon_decay_steps"]

        obs = self.env.reset() if (last_obs is None) else last_obs

        last_steps = []
        while top:  # What is top used for?
            a = self.get_action(obs, self.epsilon)
            last_steps.append(obs[a])

            obs2, reward, done, info = self.env.step(a)

            if self.args["exp_replay"]:
                if done:
                    for j in range(len(last_steps)):
                        self.buffer.add(self.env.context.compute_features(last_steps[j]), -len(last_steps) + j, None)
                    last_steps = []
                else:
                    if len(last_steps) >= self.args["n_step"]:
                        self.buffer.add(self.env.context.compute_features(last_steps[0]), -self.args["n_step"], obs2)
                    last_steps = last_steps[len(last_steps) - self.args["n_step"] + 1:]
                self.batch_update()
            else:
                self.update(obs, a, reward, obs2)

            if done:
                instance = (self.env.info["problem"], self.env.info["n"], self.env.info["k"])
                if instance not in self.best_training_perf.keys() or \
                        info["expanded transitions"] < self.best_training_perf[instance]:
                    self.best_training_perf[instance] = info["expanded transitions"]
                    print(f"New best at instance {str(instance)}! {self.best_training_perf[instance]} Steps: {self.training_steps}")
                    self.last_best = self.training_steps
                info.update({
                    "training time": time.time() - self.training_start,
                    "training steps": self.training_steps,
                    "instance": instance,
                    "loss": self.model.current_loss(),
                    })
                self.training_data.append(info)
                obs = self.env.reset()
            else:
                obs = obs2

            if self.training_steps % save_freq == 0 and results_path is not None:
                self.save(self.env.info, path=results_path)


            if self.args["target_q"] and self.training_steps % self.args["reset_target_freq"] == 0:
                if self.verbose:normalize_reward=False
            steps += 1
            self.training_steps += 1
            if done:
                top -= 1
                eps += 1

            if seconds is not None and time.time() - self.training_start > seconds:
                break

            if max_steps is not None and not early_stopping and steps >= max_steps:
                break

            if max_eps is not None and eps >= max_eps:
                break

            if max_steps is not None and self.training_steps > max_steps and (self.training_steps - self.last_best) / self.training_steps > 0.33:
                print("Converged since steps are", self.training_steps, "and max_steps is", max_steps, "and last best was", self.last_best)
                self.converged = True

            if early_stopping and self.converged:
                print("Converged!")
                break

            if self.epsilon > self.args["last_epsilon"] + 1e-10:
                self.epsilon -= epsilon_step

        if results_path is not None and save_at_end:
            self.save(self.env.info, results_path)

        return obs

    def get_action(self, s, epsilon, env=None):
        """ Gets epsilon-greedy action using self.model """
        if env is None:
            env = self.env
        if np.random.rand() <= epsilon:
            return np.random.randint(len(s))
        else:
            features = env.actions_to_features(s)
            return self.model.best(features)

    def update(self, obs, action, reward, obs2):
        """ Gets epsilon-greedy action using self.model """
        if self.target is not None:
            value = self.target.eval(obs2)
        else:
            value = self.model.eval(obs2)

        self.model.single_update(obs[action], value+reward)

        if self.verbose:
            print("Single update. Value:", value+reward)

    def batch_update(self):
        action_featuress, rewards, obss2 = self.buffer.sample(self.args["batch_size"])
        if self.target is not None:
            values = self.target.eval_batch(obss2)
        else:
            values = self.model.eval_batch(obss2)

        if self.verbose:
            print("Batch update. Values:", rewards+values)

        self.model.batch_update(np.array(action_featuress), rewards + values)
