import numpy as np
import subprocess
from src.composition import CompositionGraph, CompositionAnalyzer
from src.environment import Environment, FeatureEnvironment, FeatureCompleteEnvironment
from src.agents.dqn import DQN, NeuralNetwork, TorchModel
from src.agents.random import RandomAgent
from src.experiments import Experiment
from src.agents.RA import RA

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

    def run2(self, budget, instance, save=True):
        return self.run_agent(budget, RA(), instance, save)


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