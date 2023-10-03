import sys
import random
import numpy as np
from src.composition import CompositionGraph, CompositionAnalyzer
from src.environment import Environment
from src.agents.dqn import DQN, NeuralNetwork, TorchModel
from src.agents.random import RandomAgent
import cProfile
import time

class Experiment(object):
    def __init__(self, name="Test"):
        self.name = name
        self.platform = sys.platform
        self.seed = 14
        random.seed(self.seed)
        np.random.seed(self.seed)


    #def run(self):
    #    raise NotImplementedError

    def run_instance(self, env, agent):
        state = env.reset()
        finish = False
        rewards = []
        i = 0

        while not finish:
            idx = agent.get_action(state, 0, env)
            state, reward, finish, info = env.step(idx)
            rewards.append(reward)
            i = i + 1

        res = env.get_info()

        env.close()
        return res

    def print_res(self, title, res):
        print(title)
        print(f' Syntesis Time: {res["synthesis time(ms)"]}ms')  # Time used in java synthesis
        print(f' Expanded Transactions: {res["expanded transitions"]}')  # Amount of time step was called
        print(f' Expanded states: {res["expanded states"]}')  # Amount of states in the graph

    # Original parameters:
    # problems='AT-BW-CM-DP-TA-TL', exp_path='my experiment', step_2_results='step_2_results.csv', step_3_results='step_3_results.csv', desc='main',
    # learning_rate=1e-05, first_epsilon=1.0, last_epsilon=0.01, epsilon_decay_steps=250000, target_q=True, reset_target_freq=10000, exp_replay=True,
    # buffer_size=10000, batch_size=10, n_step=1, nesterov=True, momentum=0.9, weight_decay=0.0001, training_steps=500000, save_freq=5000, early_stopping=True,
    # step_2_n=100, step_2_budget=5000, step_3_budget=15000, ra=False, labels=True, context=True, state_labels=1, je=True, nk=False, prop=False, visits=False,
    # ltr=False, boolean=True, cbs=False, overwrite=False, max_instance_size=15, nn_size=(20,))

    def default_args(self):
        return {"nn_size": [20],
                "learning_rate": 1e-5,
                "momentum": 0.9,
                "nesterov": True,
                "weight_decay": 1e-4,
                "first_epsilon": 1.0,
                "buffer_size": 10000,
                "n_step": 5,           # 1
                "last_epsilon": 0.01,
                "epsilon_decay_steps": 250000,
                "exp_replay": True,
                "target_q": False,
                "reset_target_freq": 10000,
                "batch_size": 10
                }

class TestTrainInstance(Experiment):
    def __init__(self, name="Test"):
        super().__init__(name)

    def run(self, path, instance, n, k):
        d = CompositionGraph(instance, n, k, path)
        d.start_composition()
        context = CompositionAnalyzer(d)
        env = Environment(context, False)
        nfeatures = env.get_nfeatures()
        args = self.default_args()

        neural_network = NeuralNetwork(nfeatures, args["nn_size"]).to("cpu")
        nn_model = TorchModel(nfeatures, network=neural_network, args=args)
        agent = DQN(env, nn_model, args, save_file=None, verbose=False)
        agent.train(seconds=None, max_steps=None, max_eps=None, last_obs=None, early_stopping=False, top=10)
        print("Trained :)\n")

        res = self.run_instance(env, agent)
        self.print_res(f"Results For Intance: {instance} {n}-{k}", res)



class TrainSmallInstanceCheckBigInstance(Experiment):
    def __init__(self, name="Test"):
        super().__init__(name)

    def run(self, instance, n_train, k_train, n_test, k_test, use_saved_agent=False):
        if "linux" in self.platform:
            path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp"  # For Linux
        else:
            path = "F:\\UBA\\Tesis\\MTSApy\\fsp"  # For Windows

        d = CompositionGraph(instance, n_train, k_train, path).start_composition()
        context = CompositionAnalyzer(d)
        env = Environment(context, False)
        nfeatures = env.get_nfeatures()
        args = self.default_args()
        onnx_path = f"results/models/{instance}/{instance}-{n_train}-{k_train}.onnx"

        if use_saved_agent:
            nn_model = TorchModel.load(nfeatures, onnx_path, args=args)
            dqn_agent = DQN(env, nn_model, args, verbose=False)

        else:
            neural_network = NeuralNetwork(nfeatures, args["nn_size"]).to("cpu")
            nn_model = TorchModel(nfeatures, network=neural_network, args=args)
            dqn_agent = DQN(env, nn_model, args, verbose=False)
            dqn_agent.train(seconds=None, max_steps=None, max_eps=None, last_obs=None, early_stopping=False, top=1000)
            print(f"Trained in instance: {instance} {n_train}-{k_train}")
            DQN.save(dqn_agent, onnx_path)

        env.reset(CompositionGraph(instance, n_test, k_test, path).start_composition())
        random_agent = RandomAgent()

        print(f"Runing Random Agent in instance: {instance} {n_test}-{k_test}")
        res = self.run_instance(env, random_agent)
        self.print_res("Random Agent: ", res)

        print(f"Runing DQN Agent in instance: {instance} {n_test}-{k_test}")
        res = self.run_instance(env, dqn_agent)
        self.print_res("DQN Agent: ", res)

class TrainSmallerInstanceCheckInAll(Experiment):
    def __init__(self, name="Test"):
        super().__init__(name)

    def run(self, instance, n_min, k_min, n_max, k_max, use_saved_agent=False):
        if "linux" in self.platform:
            path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp"  # For Linux
        else:
            path = "F:\\UBA\\Tesis\\MTSApy\\fsp"  # For Windows

        d = CompositionGraph(instance, n_min, k_min, path).start_composition()
        context = CompositionAnalyzer(d)
        env = Environment(context, False)
        nfeatures = env.get_nfeatures()
        args = self.default_args()
        onnx_path = f"results/models/{instance}/{instance}-{n_min}-{k_min}.onnx"

        if use_saved_agent:
            nn_model = TorchModel.load(nfeatures, onnx_path, args=args)
            dqn_agent = DQN(env, nn_model, args, verbose=False)

        else:
            neural_network = NeuralNetwork(nfeatures, args["nn_size"]).to("cpu")
            nn_model = TorchModel(nfeatures, network=neural_network, args=args)
            dqn_agent = DQN(env, nn_model, args, verbose=False)
            dqn_agent.train(seconds=None, max_steps=None, max_eps=None, last_obs=None, early_stopping=False, top=100000)
            print(f"Trained in instance: {instance} {n_min}-{k_min}")
            DQN.save(dqn_agent, onnx_path)

        random_agent = RandomAgent()
        instances = [(n, k) for n in range(n_min, n_max+1) for k in range(k_min, k_max+1)]
        for n, k in instances:
            env.reset(CompositionGraph(instance, n, k, path).start_composition())

            print(f"\n----------------------------------------\n")

            print(f"Runing Random Agent in instance: {instance} {n}-{k}")
            res = self.run_instance(env, random_agent)
            self.print_res("Random Agent: ", res)

            print(f"Runing DQN Agent in instance: {instance} {n}-{k}")
            res = self.run_instance(env, dqn_agent)
            self.print_res("DQN Agent: ", res)
