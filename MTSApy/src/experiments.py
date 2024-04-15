import sys
import os
import random
import datetime
import subprocess
import numpy as np
import pandas as pd
from src.composition import CompositionGraph, CompositionAnalyzer
from src.environment import Environment, FeatureEnvironment, FeatureCompleteEnvironment
from src.agents.dqn import DQN, NeuralNetwork, TorchModel, OnnxModel
from src.agents.random import RandomAgent
import csv

class Experiment(object):
    def __init__(self, name="Test"):
        self.name = name
        self.platform = sys.platform
        self.BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]
        self.set_ranges(1, 15)
        self.seed = 12
        random.seed(self.seed)
        np.random.seed(self.seed)

    def set_ranges(self, min_instance_size, max_instance_size):
        self.min_instance_size = min_instance_size+1
        self.max_instance_size = max_instance_size

    def all_instances_iterator(self):
        for instance in self.BENCHMARK_PROBLEMS:
            for n in range(self.min_instance_size, self.max_instance_size + 1):
                for k in range(self.min_instance_size, self.max_instance_size + 1):
                    yield instance, n, k

    def all_instances_of(self, instance):
        #for n in range(5, self.max_instance_size + 1):
        for n in range(self.min_instance_size-1, self.max_instance_size + 1):
            for k in range(self.min_instance_size-1, self.max_instance_size + 1):
                yield instance, n, k

    def run_instance(self, env, agent, budget=-1):
        if budget == -1:
            budget = float("inf")
        state = env.reset()
        finish = False
        rewards = []
        trace = []
        info = {}

        i = 0

        while not finish and i <= budget:
            idx = agent.get_action(state, 0, env)
            state, reward, finish, info = env.step(idx)
            rewards.append(reward)
            trace.append(idx)
            i = i + 1

        res = env.get_info()
        res["failed"] = (not finish) or (info["error"])
        res["trace"] = trace

        env.close()
        return res

    def run_agent(self, budget, agent, instance, experiment_name, save=True, instance_list=None, total_budget=None):
        path = self.get_fsp_path()

        last_failed = False
        last_n = self.min_instance_size-1
        solved = 0
        expansions = 0
        all_fail = False

        if instance_list is None:
            instance_list = [(i, j) for i in range(self.min_instance_size-1, self.max_instance_size+1) for j in range(self.min_instance_size-1, self.max_instance_size+1)]

        if total_budget is None:
            total_budget = budget * len(instance_list) + 1

        for i, (n, k) in enumerate(instance_list):
            if n != last_n:
                last_failed = False

            if expansions > total_budget:
                all_fail = True

            if last_failed or all_fail:
                print(f"DQN Agent in instance: {instance} {n}-{k}: Failed")
                res = {"expanded transitions": budget + 1,
                       "expanded states": budget + 1,
                       "failed": True}

            else:
                env = self.get_environment(instance, experiment_name, n, k, path)

                print(f"Runing DQN Agent in instance: {instance} {n}-{k}")
                res = self.run_instance(env, agent, budget)
                self.print_res("DQN Agent: ", res)

            if save:
                info = {"Instance": instance, "N": n, "K": k,
                        "Name": agent.get_name(),
                        "Transitions": res["expanded transitions"],
                        "States": res["expanded states"],
                        "Failed": res["failed"]}

                csv_path = f"./results/csv/{experiment_name}-{instance}.csv"
                self.save_to_csv(csv_path, info)

            if res["failed"]:
                if k == 2:
                    all_fail = True

            else:
                solved += 1

            expansions += res["expanded transitions"]
            last_failed = res["failed"]
            last_n = n

        return solved, expansions

    def print_res(self, title, res):
        print(title)
        print(f' Expanded Transactions: {res["expanded transitions"]}')  # Amount of expansions
        print(f' Expanded states: {res["expanded states"]}\n')  # Amount of states in the graph

    # Original parameters:
    # problems='AT-BW-CM-DP-TA-TL', exp_path='my experiment', step_2_results='step_2_results.csv', step_3_results='step_3_results.csv', desc='main',
    # learning_rate=1e-05, first_epsilon=1.0, last_epsilon=0.01, epsilon_decay_steps=250000, target_q=True, reset_target_freq=10000, exp_replay=True,
    # buffer_size=10000, batch_size=10, n_step=1, nesterov=True, momentum=0.9, weight_decay=0.0001, training_steps=500000, save_freq=5000, early_stopping=True,
    # step_2_n=100, step_2_budget=5000, step_3_budget=15000, ra=False, labels=True, context=True, state_labels=1, je=True, nk=False, prop=False, visits=False,
    # ltr=False, boolean=True, cbs=False, overwrite=False, max_instance_size=15, nn_size=(20,))

    def default_args(self):
        return {"nn_size": [24],            # [20]
                "learning_rate": 1e-5,
                "momentum": 0.9,
                "nesterov": True,
                "weight_decay": 0.0001,
                "first_epsilon": 1.0,
                "buffer_size": 10000,   # 10000
                "n_step": 1,
                "last_epsilon": 0.01,          # 0.01
                "epsilon_decay_steps": 300000,   # 300000
                "exp_replay": True,
                "target_q": True,
                "reset_target_freq": 10000,      # 10000
                "batch_size": 10,
                "Adam": True,
                "reward_shaping": False,

                "lambda_warm_up": None,
                #"lambda_warm_up": lambda step: 1.0 if step > 5000 else step * 0.99,

                ### Miscellaneous
                'freq_save': 1,
                'seconds': None,
                'max_steps': 700000,    # 700000
                "max_eps": 1000000,
                "compute_python_features": False
                }

    def init_instance_res(self):
        return {"expanded transitions max": -1,
                "expanded states max": -1,
                "expanded transitions min": 9999999,
                "expanded states min": 9999999,
                "expanded transitions mean": 0,
                "expanded states mean": 0,
                "failed": 0}

    def update_instance_res(self, instance_res, res):
        instance_res["expanded transitions min"] = min(res["expanded transitions"],
                                                       instance_res["expanded transitions min"])
        instance_res["expanded states min"] = min(res["expanded states"], instance_res["expanded states min"])
        instance_res["expanded transitions max"] = max(res["expanded transitions"],
                                                       instance_res["expanded transitions max"])
        instance_res["expanded states max"] = max(res["expanded states"], instance_res["expanded states max"])
        instance_res["expanded transitions mean"] += res["expanded transitions"]
        instance_res["expanded states mean"] += res["expanded states"]
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
        path = os.path.join(os.getcwd(), "fsp")
        return path

    def get_environment(self, instance, experiment_name, n, k, path, reward_shaping=False):
        d = CompositionGraph(instance, experiment_name, n, k, path).start_composition()
        context = CompositionAnalyzer(d)
        return FeatureEnvironment(context, reward_shaping)

    def get_complete_environment(self, instance, n, k, path):
        d = CompositionGraph(instance, n, k, path).start_composition()
        context = CompositionAnalyzer(d)
        return FeatureCompleteEnvironment(context, False)



class TrainSmallInstance(Experiment):
    def __init__(self, name="Test"):
        super().__init__(name)
        self.instance_list = None

    def train(self, instance, n_train, k_train, experiment_name):
        args = self.default_args()
        path = self.get_fsp_path()
        env = self.get_environment(instance, experiment_name, n_train, k_train, path, args["reward_shaping"])

        nfeatures = env.get_nfeatures()
        pth_path = f"results/models/{instance}/{experiment_name}/{instance}-{n_train}-{k_train}.pth"
        os.makedirs(pth_path.rsplit('/', 1)[0], exist_ok=True)

        print(f"Starting training in instance {experiment_name}: {instance}-{n_train}-{k_train}...")
        neural_network = NeuralNetwork(nfeatures, args["nn_size"]).to("cpu")
        nn_model = TorchModel(nfeatures, network=neural_network, args=args)
        nn_model.set_tmp_path(f"./results/models/{instance}/{experiment_name}/tmp.onnx")
        dqn_agent = DQN(env, nn_model, args, verbose=False)
        dqn_agent.train(seconds=args["seconds"], max_steps=args["max_steps"], max_eps=args["max_eps"], pth_path=pth_path, transitions_path=None, freq_save=args["freq_save"])
        print(f"Trained in instance: {instance} {n_train}-{k_train}\n")



    def curriculum_train(self, instance, experiment_name, train_args):
        path = self.get_fsp_path()
        args = self.default_args()
        small_n, small_k = train_args[0]["n"], train_args[0]["k"]
        env = self.get_environment(instance, experiment_name, small_n, small_k, path)
        nfeatures = env.get_nfeatures()
        neural_network = NeuralNetwork(nfeatures, args["nn_size"]).to("cpu")
        nn_model = TorchModel(nfeatures, network=neural_network, args=args)
        nn_model.set_tmp_path(f"./results/models/{instance}/{experiment_name}/tmp.onnx")
        dqn_agent = DQN(env, nn_model, args, verbose=False)

        for t_args in train_args:
            n, k = t_args["n"], t_args["k"]
            env = self.get_environment(instance, experiment_name, n, k, path)
            pth_path = f"results/models/curriculum/{instance}/{instance}-{n}-{k}.pth"
            dqn_agent.reset_train_for(env)

            dqn_agent.train(seconds=t_args["seconds"], max_steps=t_args["max_steps"], max_eps=t_args["max_eps"],
                            pth_path=pth_path, transitions_path=None, freq_save=t_args["freq_save"])

    def train_and_select(self):
        pass



class TestTrainedInAllInstances(Experiment):
    def __init__(self, name='Test'):
        super().__init__(name)

    def get_previous_models(self, path):
        try:
            df = pd.read_csv(path)
            return set(df["Model"]), min(df["Expansions"])
        except:
            return set(), float('inf')

    #def pre_select_with_only_instance(self, instance, budget, path, amount_of_models=1000, csv_path=None, n, k):

    def pre_select(self, instance, experiment_name, budget, amount_of_models=1000, instance_list=None):
        path = f"./results/models/{instance}/{experiment_name}"
        all_models = {os.path.join(r, file) for r, d, f in os.walk(path) for file in f}
        csv_path = f"./results/selection/{experiment_name}-{instance}.csv"

        previous_models, best_expansions = self.get_previous_models(csv_path)
        all_models = list(all_models - previous_models)
        #print(all_models)

        models = np.random.choice(all_models, min(amount_of_models, len(all_models)), replace=False)
        best_expansions = min(len(models) * budget + 1, best_expansions)
        for model in models:
            print(f"Runing: {model}")
            solved, expansions = self.run(instance, experiment_name, budget, model, False, instance_list, best_expansions)
            best_expansions = min(best_expansions, expansions)

            # Save the info
            info = {"Instance": instance,
                    "Model": model,
                    "Solved": solved,
                    "Expansions": expansions}

            self.save_to_csv(csv_path, info)

    def run(self, instance, experiment_name, budget, pth_path=None, save=True, instance_list=None, total_budget=None):
        path = self.get_fsp_path()
        env = self.get_environment(instance, experiment_name, self.min_instance_size, self.min_instance_size, path)

        nfeatures = env.get_nfeatures()
        args = self.default_args()
        if pth_path is None:
            pth_path = f"results/models/{instance}/{instance}-{self.min_instance_size}-{self.min_instance_size}.pth"


        nn_model = TorchModel.load(nfeatures, pth_path, args=args)
        dqn_agent = DQN(env, nn_model, args, verbose=False)

        return self.run_agent(budget, dqn_agent, instance, experiment_name, save, instance_list, total_budget=total_budget)

    def get_best_model(self, csv_path):
        best_model = None
        max_solved = float('-inf')
        min_expansions = float('inf')

        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header

            for row in reader:
                solved = int(row[2])
                expansions = int(row[3])
                if solved > max_solved:
                    best_model = row[1]
                    max_solved = solved
                    min_expansions = expansions

                elif solved == max_solved and expansions < min_expansions:
                    best_model = row[1]
                    max_solved = solved
                    min_expansions = expansions

        return best_model

    def select_with_java(self, instance, experiment_name, budget):
        #command = f'java -classpath mtsa.jar MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.DCSForPython -s -i {instance} -e "{experiment_name}" -b {budget}'
        #command = f'java -classpath mtsa2.jar MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.DCSForPython -s -i {instance} -e "{experiment_name}" -b {budget}'
        command = f'java -classpath mtsaTA.jar MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.DCSForPython -s -i {instance} -e "{experiment_name}" -b {budget}'
        subprocess.call(command, shell=True)

    def test_with_java(self, instance, experiment_name, budget, onnx_path=None):
        if onnx_path is None:
            onnx_path = self.get_best_model(f"./results/selection/{experiment_name}-{instance}.csv")

        command = f'java -classpath mtsa.jar MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.DCSForPython -i {instance} -e "{experiment_name}" -b {budget} -m {onnx_path}'
        subprocess.call(command, shell=True)

