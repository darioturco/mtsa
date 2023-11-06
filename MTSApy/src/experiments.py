import sys
import os
import random
import datetime
import numpy as np
from src.composition import CompositionGraph, CompositionAnalyzer
from src.environment import Environment, FeatureEnvironment, EmbeddingEnvironment
from src.agents.dqn import DQN, NeuralNetwork, TorchModel
from src.agents.random import RandomAgent
from src.config import *
import time
import csv
import subprocess

class Experiment(object):
    def __init__(self, name="Test"):
        self.name = name
        self.platform = sys.platform
        self.BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]
        self.min_instance_size = 2
        self.max_instance_size = 15
        self.seed = 12
        random.seed(self.seed)
        np.random.seed(self.seed)


    #def run(self):
    #    raise NotImplementedError

    def all_instances_iterator(self):
        for instance in self.BENCHMARK_PROBLEMS:
            for n in range(self.min_instance_size, self.max_instance_size + 1):
                for k in range(self.min_instance_size, self.max_instance_size + 1):
                    yield instance, n, k

    def all_instances_of(self, instance):
        for n in range(self.min_instance_size, self.max_instance_size + 1):
            for k in range(self.min_instance_size, self.max_instance_size + 1):
                yield instance, n, k

    def run_instance(self, env, agent, budget=-1):
        if budget == -1:
            budget = float("inf")
        state = env.reset()
        finish = False
        rewards = []
        trace = []
        expanded_fv = set()

        i = 0

        while not finish and i <= budget:
            idx = agent.get_action(state, 0, env)
            expanded_fv.add(agent.feature_vector_to_number(state[idx]))
            state, reward, finish, info = env.step(idx)
            rewards.append(reward)
            trace.append(idx)
            i = i + 1

        res = env.get_info()
        res["failed"] = not finish
        res["trace"] = trace
        res["features vectores"] = expanded_fv

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
                "weight_decay": 0.0001,
                "first_epsilon": 1.0,
                "buffer_size": 10000,
                "n_step": 1,
                "last_epsilon": 0.01,
                "epsilon_decay_steps": 250000,   # 250000
                "exp_replay": True,
                "target_q": True,
                "reset_target_freq": 10000,      # 10000
                "batch_size": 10,


                ### Miscellaneous
                'freq_save': 100,
                'seconds': None,
                'max_steps': None,
                "max_eps": 15000

                }

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

    def get_fsp_path(self):
        if "linux" in self.platform:
            if NONBLOCKING:
                path = "/home/dario/Documents/Tesis/mtsa/MTSApy/fsp/NonBlocking"  # For Linux
            else:
                path = "/home/dario/Documents/Tesis/mtsa/MTSApy/fsp"  # For Linux

        else:
            path = "F:\\UBA\\Tesis\\mtsa\\MTSApy\\fsp"  # For Windows
        return path

    def get_environment(self, instance, n, k, path):
        d = CompositionGraph(instance, n, k, path).start_composition()
        context = CompositionAnalyzer(d)
        return EmbeddingEnvironment(context, False)



class RunRAInAllInstances(Experiment):
    def __init__(self, name="Test"):
        super().__init__(name)

    def read_lines(self, lines):
        results = {}
        results["expanded transitions"] = int(lines[0][len('ExpandedTransitions: '):])
        results["expanded states"] = 0
        results["synthesis time(ms)"] = int(lines[2][len('Elapsed in Synthesis: '):-3])
        results["OutOfMem"] = False
        results["Exception"] = False
        return results

    def read_results(self, lines, err_lines, command_run):

        if np.any(["OutOfMem" in line for line in err_lines]):
            print(f"Out of memory")
            self.debug_output = None
            results = {"expanded transitions": np.nan, "expanded states": np.nan, "synthesis time(ms)": np.nan, "OutOfMem": True}

        else:
            try:
                results = self.read_lines(lines)

            except BaseException as err:
                results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan, "OutOfMem": False,
                           "expanded states": np.nan, "Exception": True}

                print("Exception!", " ".join(command_run))

                if np.any([("Frontier" in line) for line in err_lines]):
                    print("Frontier did not fit in the buffer.")
                else:
                    for line in lines:
                        print(line)
                    for line in err_lines:
                        print(line)

        return results

    def run(self, budget):
        if "linux" in self.platform:
            path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp"  # For Linux
            mtsa_path = './mtsa.jar'
        else:
            path = "F:\\UBA\\Tesis\\mtsa\\MTSApy\\fsp"  # For Windows
            mtsa_path = 'F:\\UBA\\Tesis\\mtsa\\maven-root\\mtsa\\target\\mtsa.jar'

        for instance, n, k in self.all_instances_iterator():
            fsp_path = f"{path}/{instance}/{instance}-{n}-{k}.fsp"
            command = ["java", "-classpath", mtsa_path,
                       "MTSTools/ac/ic/doc/mtstools/model/operations/DCS/blocking/DirectedControllerSynthesisBlocking",
                       "-h", "Ready", "-i", fsp_path, "-e", str(budget)]

            try:
                proc = subprocess.run(command,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True)
                failed = False
            except Exception as e:
                failed = True

            if failed or proc.returncode == 124:
                print("Failed")
                results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan, "OutOfMem": False}
            else:
                lines = proc.stdout.split("\n")[2:]
                err_lines = proc.stderr.split("\n")
                results = self.read_results(lines, err_lines, command)

            csv_path = f"./results/csv/RA.csv"
            info = {"Instance": instance, "N": n, "K": k,
                    "Transitions": results["expanded transitions"],
                    "States": results["expanded states"],
                    "Time": results["synthesis time(ms)"],
                    "Failed": -1 < budget < results["expanded transitions"]}

            self.save_to_csv(csv_path, info)

class TrainSmallInstanceCheckBigInstance(Experiment):
    def __init__(self, name="Test"):
        super().__init__(name)

    def run(self, instance, n_train, k_train, n_test, k_test, use_saved_agent=False):
        path = self.get_fsp_path()
        env = self.get_environment(instance, n_train, k_train, path)

        nfeatures = env.get_nfeatures()
        args = self.default_args()
        pth_path = f"results/models/{instance}/{instance}-{n_train}-{k_train}.pth"
        transitions_path = f"results/transitions/{instance}/{instance}-{n_train}-{k_train}.txt"

        if use_saved_agent:
            nn_model = TorchModel.load(nfeatures, pth_path, args=args)
            dqn_agent = DQN(env, nn_model, args, verbose=False)

        else:
            neural_network = NeuralNetwork(nfeatures, args["nn_size"]).to("cpu")
            nn_model = TorchModel(nfeatures, network=neural_network, args=args)
            dqn_agent = DQN(env, nn_model, args, verbose=False)
            dqn_agent.train(seconds=args["seconds"], max_steps=args["max_steps"], max_eps=args["max_eps"], pth_path=pth_path, transitions_path=transitions_path)
            print(f"Trained in instance: {instance} {n_train}-{k_train}\n")
            DQN.save(dqn_agent, pth_path)

        env.reset(CompositionGraph(instance, n_test, k_test, path).start_composition())
        random_agent = RandomAgent()

        print(f"Runing Random Agent in instance: {instance} {n_test}-{k_test}")
        res = self.run_instance(env, random_agent)
        self.print_res("Random Agent: ", res)

        print(f"Runing DQN Agent in instance: {instance} {n_test}-{k_test}")
        res = self.run_instance(env, dqn_agent)
        self.print_res("DQN Agent: ", res)

class TestTrainedInAllInstances(Experiment):
    def __init__(self, name='Test'):
        super().__init__(name)

    def run(self, instance, budget, pth_path=None):
        path = self.get_fsp_path()
        env = self.get_environment(instance, self.min_instance_size, self.min_instance_size, path)

        nfeatures = env.get_nfeatures()
        args = self.default_args()
        if pth_path is None:
            pth_path = f"results/models/{instance}/{instance}-{self.min_instance_size}-{self.min_instance_size}.pth"

        nn_model = TorchModel.load(nfeatures, pth_path, args=args)
        dqn_agent = DQN(env, nn_model, args, verbose=False)
        last_failed = False
        last_n = self.min_instance_size

        for instance, n, k in self.all_instances_of(instance):
            if n != last_n:
                last_failed = False

            if last_failed:
                print(f"DQN Agent in instance: {instance} {n}-{k}: Failed")
                res = {"expanded transitions": budget+1,
                    "expanded states": budget+1,
                    "synthesis time(ms)": 9999,
                    "failed": True,
                    "features vectores": set()}
            else:
                env = self.get_environment(instance, n, k, path)

                print(f"Runing DQN Agent in instance: {instance} {n}-{k}")
                # print(f"Starting at: {datetime.datetime.now()}")
                res = self.run_instance(env, dqn_agent, budget)
                self.print_res("DQN Agent: ", res)

            csv_path = f"./results/csv/{instance}.csv"
            info = {"Instance": instance, "N": n, "K": k,
                    "Transitions": res["expanded transitions"],
                    "States": res["expanded states"],
                    "Time(ms)": res["synthesis time(ms)"],
                    "Failed": res["failed"],
                    "Features Vectors": res["features vectores"]}

            self.save_to_csv(csv_path, info)

            last_failed = res["failed"]
            last_n = n


class RunRandomInAllInstances(Experiment):
    def __init__(self, name="Test"):
        super().__init__(name)

    def run(self, budget, repetitions):
        path = self.get_fsp_path()

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

from src.agents.ppo import PPO
class TrainPPO(Experiment):
    def __init__(self, name="Test"):
        super().__init__(name)
    def run(self):
        path = self.get_fsp_path()

        d = CompositionGraph("DP", 2, 2, path).start_composition()
        context = CompositionAnalyzer(d)
        env = FeatureEnvironment(context, False)

        ppo = PPO(env)
        ppo.learn(1000000)

from src.agents.mcts import MCTS
class TrainMCST(Experiment):
    def __init__(self, name="Test"):
        super().__init__(name)
    def run(self):
        path = self.get_fsp_path()

        instance = "TL"
        d = CompositionGraph(instance, 2, 2, path).start_composition()
        context = CompositionAnalyzer(d)
        env = FeatureEnvironment(context, False)

        mcts = MCTS(env)
        mcts.train(10)

        print(f"Runing MCTS Agent in instance {instance}-2-2")
        res = self.run_instance(env, mcts, -1)
        self.print_res("MCTS Agent: ", res)

