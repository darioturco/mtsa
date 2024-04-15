import os
import io
import sys
import contextlib

import torch
import torch.nn as nn
import time
import numpy as np
import csv
import matplotlib.pyplot as plt

from onnx2torch import convert
from src.agents.replay_buffer import ReplayBuffer
from src.agents.agent import Agent
import onnx
from onnxruntime import InferenceSession
from torch.optim.lr_scheduler import LambdaLR

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

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



class TorchModel(Model):
    def __init__(self, nfeatures, network, args):
        super().__init__()
        self.nfeatures = nfeatures

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = network
        self.loss_fn = nn.MSELoss()
        if args["Adam"]:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=args["learning_rate"],
                                              weight_decay=args["weight_decay"])

        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=args["learning_rate"],
                                             momentum=args["momentum"],
                                             nesterov=args["nesterov"],
                                             weight_decay=args["weight_decay"])

        if args["lambda_warm_up"] is None:
            args["lambda_warm_up"] = self.constant_one_function
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=[args["lambda_warm_up"]], verbose=False)

        self.args = args
        self.has_learned_something = False
        self.losses = []
        self.path = ""
        self.tmp_path = f"./tmp.onnx"

    def constant_one_function(self, epoach):
        return 1.0

    def eval_batch(self, ss):
        return np.array([self.eval(s) for s in ss])

    def eval(self, s):
        if not self.has_learned_something or s is None:
            return 0
        return float(self.predict(s).max())

    def best(self, s):
        if not self.has_learned_something or s is None:
            return 0
        res = self.predict(s)
        #for fv in
        return int(res.argmax())

    def predict(self, s):
        if not self.has_learned_something or s is None:
            return 0
        return self.model(torch.tensor(s).to(self.device))

    def single_update(self, s, value):
        return self.batch_update(np.array([s]), np.array([value]))

    def batch_update(self, ss, values, eps):
        ss = torch.tensor(ss).to(self.device)
        values = torch.tensor(values, dtype=torch.float, device=self.device).reshape(len(ss), 1)

        self.optimizer.zero_grad()
        pred = self.model(ss)

        loss = self.loss_fn(pred, values)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        self.losses.append(loss.item())
        self.has_learned_something = True

    def set_lr(self, new_lr):
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

    def nfeatures(self):
        return self.nfeatures

    # should be called only at the end of each episode
    def current_loss(self):
        avg_loss = np.mean(self.losses)
        self.losses = []
        return avg_loss

    def to_onnx(self):
        x = torch.randn(1, self.nfeatures, device=self.device)
        torch.onnx.export(self.model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          self.tmp_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          verbose=False,
                          input_names=['X'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'X': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})

        return onnx.load(self.tmp_path), InferenceSession(self.tmp_path)

    def copy_network(self):
        net = NeuralNetwork(self.nfeatures, self.model.nnsize)
        weights = self.model.state_dict()
        net.load_state_dict(weights)
        return net

    def set_tmp_path(self, new_path):
        self.tmp_path = new_path

    def copy(self):
        network_copy = self.copy_network()
        res = TorchModel(self.nfeatures, network_copy, self.args)
        res.has_learned_something = self.has_learned_something
        res.set_tmp_path(self.tmp_path)
        return res

    def save(self, path):
        OnnxModel(self).save(path)

    def remove_temp_files(self):
        os.system(f"rm {self.tmp_path}")

    @classmethod
    def load(cls, nfeatures, path, args):
        network = convert(onnx.load(path))
        new_model = cls(nfeatures, network, args)
        new_model.has_learned_something = True
        new_model.path = path
        return new_model

class LSTMNeuralNetwork(nn.Module):
    def __init__(self, nfeatures, nnsize):
        super(LSTMNeuralNetwork, self).__init__()
        self.nnsize = nnsize
        self.layers = nn.ModuleList([nn.LSTM(nfeatures, nnsize[0]), nn.Linear(nnsize[0], 1)])

    def forward(self, x):
        for layer in self.layers:
            if isinstance(x, tuple):
                x = x[0]
            x = layer(x.to(torch.float))
        return x

class NeuralNetwork(nn.Module):
    def __init__(self, nfeatures, nnsize):
        super(NeuralNetwork, self).__init__()
        self.nnsize = nnsize

        nnsize = list(nnsize) + [1]
        layers = [nn.Linear(nfeatures, nnsize[0])]
        for i in range(len(nnsize)-1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(nnsize[i], nnsize[i+1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(x, tuple):
                x = x[0]
            x = layer(x.to(torch.float))
        return x

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

class DQN(Agent):
    def __init__(self, env, nn_model, args, verbose=False):
        assert nn_model is not None

        self.env = env
        self.args = args
        self.model = nn_model

        self.target = None
        self.buffer = None

        self.trained = False

        self.training_start = None
        self.epsilon = args["first_epsilon"]

        self.verbose = verbose

        self.training_data = []
        self.expanded_transitions = set()

        self.best_training_perf = {}
        self.last_best = None
        self.converged = False
        self.freq_save = args['freq_save']

    def get_name(self):
        return self.model.path

    def initializeBuffer(self):
        """ Initialize replay buffer uniformly with experiences """
        buffer_size = self.args["buffer_size"]

        print(f"Initializing buffer with {buffer_size} observations...")

        self.buffer = ReplayBuffer( self.env, buffer_size)
        random_experience = self.buffer.get_experience_from_random_policy(total_steps=buffer_size, nstep=self.args["n_step"])
        for action_features, reward, obs2 in random_experience:
            self.buffer.add(action_features, reward, obs2)

        print("Done.")

    def reset_train_for(self, env):
        self.env = env
        self.epsilon = self.args["first_epsilon"]
        self.converged = False
        self.trained = False

    def train(self, seconds=None, max_steps=None, max_eps=10000, pth_path=None, transitions_path=None, freq_save=None):
        last_obs = None
        if freq_save is not None:
            self.freq_save = freq_save

        instance, n, k = self.env.get_instance_info()
        experiment_name = self.env.get_experiment_name()
        csv_path = f"./results/training/{instance}-{experiment_name}.csv"
        os.makedirs(csv_path.rsplit('/', 1)[0], exist_ok=True)

        acumulated_reward = 0
        expansion_steps = 0
        all_rewards = []
        all_expansions = []
        losses = []
        self.initializeBuffer()

        if self.training_start is None:
            self.training_start = time.time()
            self.last_best = 0

        self.steps = 1
        self.eps = 1

        epsilon_step = (self.args["first_epsilon"] - self.args["last_epsilon"])
        epsilon_step /= self.args["epsilon_decay_steps"]

        obs = self.env.reset() if (last_obs is None) else last_obs

        last_steps = []
        while self.eps < max_eps:
            a = self.get_action(obs, self.epsilon)
            last_steps.append(obs[a])
            self.expanded_transitions.add(self.feature_vector_to_number(obs[a]))

            obs2, reward, done, info = self.env.step(a)

            acumulated_reward += reward
            expansion_steps += 1
            if self.args["exp_replay"]:
                if done:
                    for j in range(len(last_steps)):
                        self.buffer.add(last_steps[j], -len(last_steps) + j, None)
                    last_steps = []
                else:
                    if len(last_steps) >= self.args["n_step"]:
                        self.buffer.add(last_steps[0], -self.args["n_step"], obs2)
                    last_steps = last_steps[len(last_steps) - self.args["n_step"] + 1:]
                self.batch_update()
            else:
                self.update(obs, a, reward, obs2)

            if done:
                loss = self.model.current_loss()
                losses.append(loss)

                all_rewards.append(acumulated_reward)
                all_expansions.append(expansion_steps)
                print(f"Step: {self.steps} - Epsode: {self.eps} - Expansions: {expansion_steps} - Reward: {acumulated_reward} - Acumulated: {np.mean(all_rewards[-32:])} - Epsilon: {self.epsilon}")
                if self.freq_save is not None and self.eps % self.freq_save == 0:
                    if pth_path is not None:
                        if len(all_rewards) > 1000:
                            all_rewards = all_rewards[100:]

                        DQN.save(self, f"{pth_path[:-4]}-{self.eps}", partial=True)
                        if not os.path.exists(csv_path):
                            with open(csv_path, 'w') as _:  # Create the file if it doesn't exist
                                pass

                        with open(csv_path, 'a') as f:
                            writer = csv.writer(f)
                            for i in range(self.freq_save - 1):
                                idx = -self.freq_save + 1 + i
                                writer.writerow([self.steps, all_expansions[idx], all_rewards[idx], losses[idx]])



                    if transitions_path is not None:
                        with open(transitions_path, 'w') as f:
                            f.write(str(self.expanded_transitions))

                obs = self.env.reset()
                acumulated_reward = 0
                expansion_steps = 0
                self.eps += 1

                if max_steps is not None and self.steps >= max_steps:
                    break

            else:
                obs = obs2

            if self.args["target_q"] and self.steps % self.args["reset_target_freq"] == 0:
                print("Resetting target.")
                self.target = self.model.copy()

            self.steps += 1
            if self.epsilon > self.args["last_epsilon"] + 1e-10:
                self.epsilon -= epsilon_step

            if seconds is not None and time.time() - self.training_start > seconds:
                break

        self.model.remove_temp_files()
        return obs.copy()

    # TODO: Arreglar este desorden
    def get_action(self, s, epsilon, env=None):
        """ Gets epsilon-greedy action using self.model """
        if env is None:
            env = self.env

        #print(f"State FV: {s}\n - {len(s)}")
        #env.composite. printFrontier()
        #env.getJavaEnv().dcs.heuristic.printFrontier()

        res = 0
        if np.random.rand() <= epsilon:
            res = np.random.randint(len(s))
        else:
            res = self.model.best(s)

        #print(f"Expanded: {res}")
        return res

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

        self.model.batch_update(np.array(action_featuress), rewards + values, self.eps)


    @staticmethod
    def save(agent, path, partial=False):
        if partial:
            path = path + f"-partial"
        agent.model.save(path)
