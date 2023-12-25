import numpy as np
import subprocess
from src.composition import CompositionGraph, CompositionAnalyzer
from src.environment import Environment, FeatureEnvironment, FeatureCompleteEnvironment
from src.agents.dqn import DQN, NeuralNetwork, TorchModel
from src.agents.random import RandomAgent
from src.experiments import Experiment
from src.agents.RA import RA
import time

class RunRAInAllInstances(Experiment):
    def __init__(self, name="Test"):
        super().__init__(name)

    def run(self, budget, instance, save=True, verbose=False):
        path = self.get_fsp_path()
        DCSForPython = CompositionGraph.getDCSForPython()

        last_failed = False
        last_n = self.min_instance_size
        solved = 0
        all_fail = False

        for instance, n, k in self.all_instances_of(instance):
            if n != last_n:
                last_failed = False

            if last_failed or all_fail:
                expanded = budget + 1
                print(f"RA Agent in instance: {instance} {n}-{k}: Failed")
                duration = 9999

            else:
                start = time.time()

                #expanded = DCSForPython.syntetizeWithHeuristic(f"{path}/{instance}/{instance}-{n}-{k}.fsp", "Complete", budget, verbose)
                expanded = DCSForPython.syntetizeWithHeuristic(f"{path}/{instance}/{instance}-{n}-{k}.fsp", "Complete", budget, verbose)
                end = time.time()
                duration = (end - start) * 1000
                print(f"RA Agent in instance: {instance} {n}-{k}:")
                print(f"   Expanded Transitions: {expanded}")
                print(f"   Syntesis Time: {duration:.3f} ms\n")


            failed = (expanded >= budget)
            if save:
                info = {"Instance": instance, "N": n, "K": k,
                        "Model": "RA",
                        "Transitions": expanded,
                        "States": -1,
                        "Time(ms)": round(duration),
                        "Failed": failed}

                csv_path = f"./results/csv/{instance}.csv"
                self.save_to_csv(csv_path, info)

            if failed:
                if k == 2:
                    all_fail = True
            else:
                solved += 1

            last_failed = failed
            last_n = n

        return solved



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
    def train(self, instances):
        path = self.get_fsp_path()

        for instance in instances:
            print(f"Training {instance}...")
            env = self.get_complete_environment(instance, 2, 2, path)

            ppo = PPO(env, self.default_args())
            ppo.train(10000)

            print(f"Runing PPO Agent in instance {instance}-2-2")
            res = self.run_instance(env, ppo, -1)
            self.print_res("PPO Agent: ", res)

    def default_args(self):
        return {"learning_rate": 3e-4, "gamma": 0.99, "batch_size": 50, "verbose": 1}

    def test(self, instance, budget, pth_path=None):
        path = self.get_fsp_path()

        args = self.default_args()
        if pth_path is None:
            pth_path = f"results/models/PPO/{instance}/{instance}-{self.min_instance_size}-{self.min_instance_size}"

        last_failed = False
        last_n = self.min_instance_size
        all_fall = False

        for instance, n, k in self.all_instances_of(instance):

            if n != last_n:
                last_failed = False

            if last_failed or all_fall:
                print(f"PPO Agent in instance: {instance} {n}-{k}: Failed")
                res = {"expanded transitions": budget + 1,
                       "expanded states": budget + 1,
                       "synthesis time(ms)": 9999,
                       "failed": True,
                       "features vectores": set()}
            else:
                env = self.get_complete_environment(instance, n, k, path)
                ppo_agent = PPO.load(env, pth_path, args)

                print(f"Runing PPO Agent in instance: {instance} {n}-{k}")
                res = self.run_instance(env, ppo_agent, budget)
                self.print_res("PPO Agent: ", res)

            csv_path = f"./results/csv/PPO-{instance}.csv"
            info = {"Instance": instance, "N": n, "K": k,
                    "Transitions": res["expanded transitions"],
                    "States": res["expanded states"],
                    "Time(ms)": res["synthesis time(ms)"],
                    "Failed": res["failed"],
                    "Features Vectors": res["features vectores"]}

            self.save_to_csv(csv_path, info)

            if res["failed"] and k == self.min_instance_size:
                all_fall = True

            last_failed = res["failed"]
            last_n = n