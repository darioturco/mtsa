import torch
import torch.nn as nn
import time
import numpy as np
import csv
from torch.optim import Adam
from torch.distributions import Categorical
import matplotlib.pyplot as plt

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



class TorchModel(Model):
    def __init__(self, nfeatures, network, args):
        super().__init__()
        self.nfeatures = nfeatures

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

    def batch_update(self, ss, values, eps):
        ss = torch.tensor(ss).to(self.device)
        values = torch.tensor(values, dtype=torch.float, device=self.device).reshape(len(ss), 1)

        self.optimizer.zero_grad()
        pred = self.model(ss)

        loss = self.loss_fn(pred, values)
        #if loss.item() < 1.0:
        loss.backward()
        #if eps > 1000 and loss.item() > 1.0:
        #    print("Mucho loss")
        #    return
        #print(loss)
        self.optimizer.step()

        self.losses.append(loss.item())
        self.has_learned_something = True



    def nfeatures(self):
        return self.nfeatures

    # should be called only at the end of each episode
    def current_loss(self):
        avg_loss = np.mean(self.losses)
        #avg_loss = np.max(self.losses)
        self.losses = []
        return avg_loss

    def to_onnx(self):
        x = torch.randn(1, self.nfeatures, device=self.device)
        torch.onnx.export(self.model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          "tmp.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['X'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'X': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})
        return onnx.load("tmp.onnx"), InferenceSession("tmp.onnx")

    def save(self, path):
        torch.save(self.model, path)
    @classmethod
    def load(cls, nfeatures, path, args):
        network = torch.load(path)
        new_model = cls(nfeatures, network, args)
        new_model.has_learned_something = True
        return new_model


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

import onnx
from onnxruntime import InferenceSession
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
        self.training_steps = 0
        self.epsilon = args["first_epsilon"]

        self.verbose = verbose

        self.training_data = []

        self.best_training_perf = {}
        self.last_best = None
        self.converged = False
        self.freq_save = 100

    def initializeBuffer(self):
        """ Initialize replay buffer uniformly with experiences """
        buffer_size = self.args["buffer_size"]

        print(f"Initializing buffer with {buffer_size} observations...")

        self.buffer = ReplayBuffer( self.env, buffer_size)
        random_experience = self.buffer.get_experience_from_random_policy(total_steps=buffer_size, nstep=self.args["n_step"])
        for action_features, reward, obs2 in random_experience:
            self.buffer.add(action_features, reward, obs2)



        print("Done.")

    #def train(self, seconds=None, max_steps=None, max_eps=100000, early_stopping=False, pth_path=None):
    def train(self, seconds=None, max_steps=None, max_eps=10000,
              early_stopping=False, pth_path=None):

        last_obs = None
        top = 1000

        instance, n, k = self.env.get_instance_info()
        csv_path = f"./results/training/{instance}-{n}-{k}-partial.csv"
        #saved = False
        acumulated_reward = 0
        all_rewards = []
        losses = []
        self.initializeBuffer()

        if self.training_start is None:
            self.training_start = time.time()
            self.last_best = 0

        steps = 1
        self.eps = 1

        epsilon_step = (self.args["first_epsilon"] - self.args["last_epsilon"])
        epsilon_step /= self.args["epsilon_decay_steps"]

        obs = self.env.reset() if (last_obs is None) else last_obs

        last_steps = []
        while top:  # What is top used for?
            a = self.get_action(obs, self.epsilon)
            last_steps.append(obs[a])

            obs2, reward, done, info = self.env.step(a)

            acumulated_reward += reward
            if self.args["exp_replay"]:
                if done:
                    #if self.eps > 500 and -acumulated_reward >= 75:
                    #    print("Wooo que paso aqui")

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
                #if instance not in self.best_training_perf.keys() or \
                #        info["expanded transitions"] < self.best_training_perf[instance]:
                #    self.best_training_perf[instance] = info["expanded transitions"]
                #    print("New best at instance " + str(instance) + "!", self.best_training_perf[instance], "Steps:", self.training_steps)
                #    self.last_best = self.training_steps
                loss = self.model.current_loss()
                losses.append(loss)
                #info.update({
                #    "training time": time.time() - self.training_start,
                #    "training steps": self.training_steps,
                #    "instance": instance,
                #    "loss": loss
                #})

                #sw = 16
                #plt.plot([np.mean([losses[i:i+sw]]) for i in range(len(losses)-sw)])
                #plt.draw()
                #plt.pause(0.0001)
                #plt.clf()

                all_rewards.append(-acumulated_reward)
                print(f"Epsode: {self.eps} - Acumulated Reward: {-acumulated_reward} - Acumulated: {np.mean(all_rewards[-32:])} - Epsilon: {self.epsilon}")
                acumulated_reward = 0
                #self.training_data.append(info)
                obs = self.env.reset()

                if self.eps % self.freq_save == 0 and pth_path is not None:
                    if len(all_rewards) > 1000:
                        all_rewards = all_rewards[100:]

                    DQN.save(self, f"{pth_path[:-4]}-{self.eps}.pth", partial=True)

                    with open(csv_path, 'a') as f:
                        writer = csv.writer(f)
                        for i in range(self.freq_save - 1):
                            idx = -self.freq_save + 1 + i
                            writer.writerow([steps, all_rewards[idx], losses[idx]])

                    print("Partial Saved!")

                self.eps += 1

            else:
                obs = obs2



            if self.args["target_q"] and self.training_steps % self.args["reset_target_freq"] == 0:
                print("Resetting target.")
                self.target = OnnxModel(self.model)

            steps += 1
            self.training_steps += 1

            if seconds is not None and time.time() - self.training_start > seconds:
                break

            if max_steps is not None and not early_stopping and steps >= max_steps:
                break

            if max_eps is not None and self.eps >= max_eps:
                break

            if max_steps is not None and self.training_steps > max_steps and (
                    self.training_steps - self.last_best) / self.training_steps > 0.33:
                print("Converged since steps are", self.training_steps, "and max_steps is", max_steps,
                      "and last best was", self.last_best)
                self.converged = True

            if early_stopping and self.converged:
                print("Converged!")
                break

            if self.epsilon > self.args["last_epsilon"] + 1e-10:
                self.epsilon -= epsilon_step

        return obs.copy()

    def get_action(self, s, epsilon, env=None):
        """ Gets epsilon-greedy action using self.model """
        if env is None:
            env = self.env
        if np.random.rand() <= epsilon:
            return np.random.randint(len(s))
        else:
            #features = env.actions_to_features(s)
            #return self.model.best(features)
            return self.model.best(s)

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
            path = path[:-4] + f"-partial.pth"
        agent.model.save(path)
