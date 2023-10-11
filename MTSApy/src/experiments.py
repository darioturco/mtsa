import sys
import os
import random
import datetime
import numpy as np
from src.composition import CompositionGraph, CompositionAnalyzer
from src.environment import Environment
from src.agents.dqn import DQN, NeuralNetwork, TorchModel
from src.agents.random import RandomAgent
from src.agents.ready_abstraction import ReadyAbstraction
from src.agents.debugging_abstraction import DebuggingAbstraction
import cProfile
import time
import csv

class Experiment(object):
    def __init__(self, name="Test"):
        self.name = name
        self.platform = sys.platform
        self.BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]
        #self.BENCHMARK_PROBLEMS = ["DP", "TA", "TL"]
        self.max_instance_size = 15
        self.seed = 1
        random.seed(self.seed)
        np.random.seed(self.seed)


    #def run(self):
    #    raise NotImplementedError

    def all_instances_iterator(self):
        for instance in self.BENCHMARK_PROBLEMS:
            for n in range(2, self.max_instance_size + 1):
                for k in range(2, self.max_instance_size + 1):
                    yield instance, n, k

    def run_instance(self, env, agent, budget=-1):
        if budget == -1:
            budget = float("inf")
        state = env.reset()
        finish = False
        rewards = []

        i = 0

        while not finish and i <= budget:
            idx = agent.get_action(state, 0, env)
            state, reward, finish, info = env.step(idx)
            rewards.append(reward)
            i = i + 1

        res = env.get_info()
        res["failed"] = not finish

        env.close()
        return res

    def print_res(self, title, res):
        print(title)
        print(f' Syntesis Time: {res["synthesis time(ms)"]}ms')  # Time used in java synthesis
        print(f' Expanded Transactions: {res["expanded transitions"]}')  # Amount of time step was called
        print(f' Expanded states: {res["expanded states"]}\n')  # Amount of states in the graph

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
                "buffer_size": 10,
                "n_step": 1,
                "last_epsilon": 0.001,       #0.01
                "epsilon_decay_steps": 250000,
                "exp_replay": True,
                "target_q": False,
                "reset_target_freq": 10000,  # 10000
                "batch_size": 10,
                "max_eps": 200000 # 100000
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
        agent.train(seconds=None, max_steps=None, max_eps=args["max_eps"], last_obs=None, early_stopping=False)
        print("Trained :)\n")

        res = self.run_instance(env, agent)
        self.print_res(f"Results For Intance: {instance} {n}-{k}", res)



class TrainSmallInstanceCheckBigInstance(Experiment):
    def __init__(self, name="Test"):
        super().__init__(name)

    def run(self, instance, n_train, k_train, n_test, k_test, use_saved_agent=False):
        if "linux" in self.platform:
            path = "/home/dario/Documents/Tesis/mtsa/MTSApy/fsp"  # For Linux
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
            dqn_agent.train(seconds=None, max_steps=None, max_eps=args["max_eps"], early_stopping=False, onnx_path=onnx_path)
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
            dqn_agent.train(seconds=None, max_steps=None, max_eps=args["max_eps"], early_stopping=False, onnx_path=onnx_path)
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

class RunRandomInAllInstances(Experiment):
    def __init__(self, name="Test"):
        super().__init__(name)

    def save_to_csv(self, path, info):
        header_list = list(info.keys())
        new_file = False
        if not os.path.isfile(path):
            new_file = True

        with open(path, 'a') as f:
            dictwriter = csv.DictWriter(f, fieldnames=header_list)
            if new_file:
                dictwriter.writerow(dict(zip(header_list, header_list)))

            dictwriter.writerow(info)
            f.close()

    def init_instance_res(self):
        return {"expanded transitions max": -1,
                "expanded states max": -1,
                "synthesis time(max)": -1,
                "expanded transitions min": 9999999,
                "expanded states min": 9999999,
                "synthesis time(min)": 9999999,
                "expanded transitions mean": 0,
                "expanded states mean": 0,
                "synthesis time(mean)": 0,
                "failed": 0}

    def update_instance_res(self, instance_res, res):
        instance_res["expanded transitions min"] = min(res["expanded transitions"],
                                                       instance_res["expanded transitions min"])
        instance_res["expanded states min"] = min(res["expanded states"], instance_res["expanded states min"])
        instance_res["synthesis time(min)"] = min(res["synthesis time(ms)"], instance_res["synthesis time(min)"])
        instance_res["expanded transitions max"] = max(res["expanded transitions"],
                                                       instance_res["expanded transitions max"])
        instance_res["expanded states max"] = max(res["expanded states"], instance_res["expanded states max"])
        instance_res["synthesis time(max)"] = max(res["synthesis time(ms)"], instance_res["synthesis time(max)"])
        instance_res["expanded transitions mean"] += res["expanded transitions"]
        instance_res["expanded states mean"] += res["expanded states"]
        instance_res["synthesis time(mean)"] += res["synthesis time(ms)"]
        instance_res["failed"] += int(res["failed"])

    def run(self, budget, repetitions):
        # La idea es correrlo con un budget de 5000 y con 100 repeticiones por instancia
        if "linux" in self.platform:
            path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp"  # For Linux
        else:
            path = "F:\\UBA\\Tesis\\MTSApy\\fsp"  # For Windows

        agent = RandomAgent(None)

        last_instance = ""
        failed = False
        for instance, n, k in self.all_instances_iterator():
            if last_instance != instance:
                failed = False

            instance_res = self.init_instance_res()
            if failed:
                instance_res = self.init_instance_res()
                instance_res["failed"] = repetitions

            else:
                for _ in range(repetitions):
                    d = CompositionGraph(instance, n, k, path).start_composition()
                    context = CompositionAnalyzer(d)
                    env = Environment(context, False)

                    print(f"Runing Random Agent in instance: {instance} {n}-{k}")
                    # print(f"Starting at: {datetime.datetime.now()}")
                    res = self.run_instance(env, agent, budget)
                    self.print_res("Random Agent: ", res)
                    self.update_instance_res(instance_res, res)

                if instance_res["failed"] >= repetitions - 1:
                    failed = True

            instance_res["expanded transitions mean"] /= repetitions
            instance_res["expanded states mean"] /= repetitions
            instance_res["synthesis time(mean)"] /= repetitions

            csv_path = f"./results/csv/random.csv"
            info = {"Instance": instance, "N": n, "K": k,
                    "Transitions (min)": instance_res["expanded transitions min"], "States (min)": instance_res["expanded states min"], "Time(min)": instance_res["synthesis time(min)"],
                    "Transitions (max)": instance_res["expanded transitions max"], "States (max)": instance_res["expanded states max"], "Time(max)": instance_res["synthesis time(max)"],
                    "Transitions (mean)": instance_res["expanded transitions mean"], "States (mean)": instance_res["expanded states mean"], "Time(mean)": instance_res["synthesis time(mean)"],
                    "Failed": instance_res["failed"]}
            last_instance = instance
            self.save_to_csv(csv_path, info)





