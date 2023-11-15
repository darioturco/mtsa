import sys
import os
import random
import datetime
import numpy as np
from src.composition import CompositionGraph, CompositionAnalyzer
from src.environment import Environment, FeatureEnvironment, FeatureCompleteEnvironment
from src.agents.dqn import DQN, NeuralNetwork, TorchModel
from src.agents.random import RandomAgent
import time
import csv
import subprocess

class Experiment(object):
    def __init__(self, name="Test"):
        self.name = name
        self.platform = sys.platform
        self.BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]
        #self.BENCHMARK_PROBLEMS = ["TL", "CM"]
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
        state, finish = env.reset()
        rewards = []
        trace = []
        expanded_fv = set()

        i = 0

        while not finish and i <= budget:
            idx = agent.get_action(state, 0, env)
            expanded_fv.add(agent.feature_vector_to_number(state[idx]))
            state, reward, finish, _, info = env.step(idx)
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
            path = "/home/dario/Documents/Tesis/mtsa/MTSApy/fsp"  # For Linux
        else:
            path = "F:\\UBA\\Tesis\\mtsa\\MTSApy\\fsp"  # For Windows
        return path

    def get_environment(self, instance, n, k, path):
        d = CompositionGraph(instance, n, k, path).start_composition()
        context = CompositionAnalyzer(d)
        return FeatureEnvironment(context, False)

    def get_complete_environment(self, instance, n, k, path):
        d = CompositionGraph(instance, n, k, path).start_composition()
        context = CompositionAnalyzer(d)
        return FeatureCompleteEnvironment(context, False)


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
#from stable_baselines3 import PPO

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

class TrainPPO(Experiment):
    def __init__(self, name="Test"):
        super().__init__(name)
    def run(self, instances):
        path = self.get_fsp_path()

        for instance in instances:
            print(f"Training {instance}...")
            env = self.get_complete_environment(instance, 2, 2, path)

            ppo = PPO(env, self.default_args())
            ppo.train(500000)

            print(f"Runing PPO Agent in instance {instance}-2-2")
            res = self.run_instance(env, ppo, -1)
            self.print_res("MCTS Agent: ", res)

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








class TrainGNN(Experiment):
    def __init__(self, name="Test"):
        super().__init__(name)
    def run(self):
        #ENABLED_PYTHON_FEATURES = {
        #    EventLabel: False,
        #    StateLabel: False,
        #    Controllable: False,
        #    MarkedSourceAndSinkStates: False,
        #    CurrentPhase: False,
        #    ChildNodeState: False,
        #    UncontrollableNeighborhood: False,
        #    ExploredStateChild: False,
        #    IsLastExpanded: False,
        #    RandomTransitionFeature: False,
        #    RandomOneHotTransitionFeature: False,
        #}

        instance = "TL"
        path = self.get_fsp_path()

        d = CompositionGraph(instance, 2, 2, path)
        d.start_composition()
        d.full_composition()


        #da = FeatureExtractor(d, None, feature_classes=ENABLED_PYTHON_FEATURES.keys())
        da = FeatureExtractor(d, None, None)
        data, device = da.composition_to_nx()

        sys.path.append("F:\\UBA\\Tesis\\dgl\\dgl\\examples\\pytorch\\vgae")
        import train_vgae
        from torch_geometric.utils import softmax

        x = softmax([1,2,3])
        dgl_data = to_dgl(data)

        best_model = train_vgae.dgl_main(dgl_data)



        #da.train_gae_on_full_graph(to_undirected=True, epochs=100000)




import torch
#from torch.utils.tensorboard import SummaryWriter
#from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, Sequential, GAE
from torch import nn
from torch_geometric.utils import from_networkx
from bidict import bidict


class LabelsOHE(object):
    @classmethod
    def compute(cls, state: CompositionGraph, node, dir="in"):
        composition = state.composition
        incoming = composition.in_edges(node, data=True) if dir == "in" else composition.out_edges(node, data=True)

        feature_vec_slice = [0.0 for _ in state._no_indices_alphabet]
        # arriving_to_s = transition.state.getParents()
        for edge in incoming:
            cls._set_transition_type_bit(feature_vec_slice, edge[2]["label"], state._fast_no_indices_alphabet_dict)
        return feature_vec_slice

    @classmethod
    def _set_transition_type_bit(cls, feature_vec_slice, transition, _fast_no_indices_alphabet_dict):
        no_idx_label = util_remove_indices(transition.toString() if type(transition) != str else transition)
        feature_vec_slice_pos = _fast_no_indices_alphabet_dict[no_idx_label]
        feature_vec_slice[feature_vec_slice_pos] = 1

class MarkedState(object):
    @classmethod
    def compute(cls, state: CompositionGraph, node):
        #return [float(node.marked)]
        return [float(len(node.markedByGuarantee) > 0)]

class NodePairSplitter:
    def __init__(self, data, split_labels=True, add_negative_train_samples=True, val_prop = 0.05, test_prop = 0.1, proportional=False):
        Warning("Sending split tensors to DEVICE may be convenient if using accelerators (TODO).")

        n_nodes = self.n_nodes = data.x.shape[0]
        n_edges = self.n_edges = data.edge_index.shape[1]
        n_neg_edges = (n_nodes ** 2) - n_edges if proportional else n_edges

        test_edge_index_idx = np.random.randint(0,n_edges,int(test_prop * n_edges))
        #val_edge_index_idx = np.random.randint(0, n_edges, int(val_prop * n_edges))
        train_edge_index_idx = [i for i in range(n_edges) if i not in test_edge_index_idx]

        #neg_edge_index = torch.tensor([(i,j) for i in range(n_nodes) for j in range(n_nodes) if torch.tensor((i,j)) not in data.edge_index.T])
        #assert(neg_edge_index.shape[0] == (n_edges ** 2)-n_edges)
        self.pos_training_edge_index = data.edge_index.T[train_edge_index_idx].tolist()
        self.pos_testing_edge_index = data.edge_index.T[test_edge_index_idx].tolist()

        self.neg_testing_edge_index = []
        self.neg_training_edge_index = []

        self.pos_training_edge_index = torch.tensor(self.pos_training_edge_index).T
        self.pos_testing_edge_index = torch.tensor(self.pos_testing_edge_index).T

    def get_split(self):
        return self.pos_training_edge_index, self.neg_training_edge_index, self.pos_testing_edge_index, self.neg_testing_edge_index

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        planes = 128
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_layer = Sequential('x, edge_index', [
            (GCNConv(in_channels, planes), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
        ])

        def conv_block():
            return Sequential('x, edge_index', [
                (GCNConv(planes, planes), 'x, edge_index -> x'),
                nn.ReLU(inplace=True)
            ])

        self.conv_blocks = Sequential('x, edge_index', [(conv_block(), 'x, edge_index -> x') for _ in range(7)])

        self.last_linear = nn.Linear(planes, out_channels)

    def forward(self, x, edge_index):
        x = self.first_layer(x, edge_index)
        x = self.conv_blocks(x, edge_index)
        x = self.last_linear(x)
        return x


class FeatureExtractor:
    """class used to get Composition information, usable as hand-crafted features
        Design:
        attrs: Features (list of classes)
        methods: .extract(featureClass, from = composition) .phi(composition)
    """

    def train_vgae_official(file_name="vgae.pt"):
        import pickle
        import sys
        sys.path.append("/home/marco/Desktop/dgl/dgl/examples/pytorch/vgae")
        import train_vgae
        import dgl
        from torch_geometric.utils import to_dgl
        for problem in ["AT", "DP", "TA", "TL", "BW", "CM"]:
            d = CompositionGraph(problem, 2, 2)
            d.start_composition()
            d.full_composition()

            da = FeatureExtractor(d, ENABLED_PYTHON_FEATURES, feature_classes=ENABLED_PYTHON_FEATURES.keys())

            data, device = da.composition_to_nx()

            dgl_data = to_dgl(data)
            best_model = train_vgae.dgl_main(dgl_data)  # TODO add parameters: graph, epochs, etc etc
            Warning(
                "I'm not so sure the parameters are correctly loaded or if the parameters are from the best model (watch out running mean and variance etc)")
            torch.save(best_model, problem + file_name)

    def __init__(self, composition, enabled_features_dict, feature_classes):
        #FIXME composition should be a parameter of phi, since FeatureExctractor works ...
        # for any context independently UNLESS there are trained features
        # in general, composition should be completely decoupled from extractor (passed always as a parameter)
        self.composition = composition
        self.context = CompositionAnalyzer(composition)
        assert (self.composition._started)

        self._no_indices_alphabet = list(set([self.remove_indices(str(e)) for e in composition._alphabet]))
        self._no_indices_alphabet.sort()
        self._fast_no_indices_alphabet_dict = dict()
        for i in range(len(self._no_indices_alphabet)): self._fast_no_indices_alphabet_dict[self._no_indices_alphabet[i]]=i
        self._fast_no_indices_alphabet_dict = bidict(self._fast_no_indices_alphabet_dict)

        self._feature_classes = self.context.get_features_methods()
        #self._feature_classes = feature_classes
        #self._enabled_feature_classes = enabled_features_dict if enabled_features_dict is not None else {feature : True for feature in self._feature_classes}
        #self._global_feature_classes = [feature_cls for feature_cls in self._feature_classes if feature_cls.__class__ == GlobalFeature]  #
        #self._node_feature_classes = [feature_cls for feature_cls in self._feature_classes if feature_cls.__class__ == NodeFeature]
    def phi(self):
        return self.frontier_feature_vectors()

    def extract(self, transition, state):
        res = []
        for feature in self._feature_classes:
            #if self.includes(feature):
            #    res += feature.compute(state=state, transition=transition)

            res += feature(transition)
        return res

    def _set_transition_type_bit(self, feature_vec_slice, transition):
        no_idx_label = self.remove_indices(transition.toString())
        feature_vec_slice_pos = self._fast_no_indices_alphabet_dict[no_idx_label]
        feature_vec_slice[feature_vec_slice_pos] = 1


    def remove_indices(self, transition_label : str):
        util_remove_indices(transition_label)
    def get_transition_features_size(self):
        if(len(self.composition.getFrontier())): return len(self.extract(self.composition.getFrontier()[0], self.composition))
        elif (len(self.composition.getNonFrontier())): return len(self.extract(self.composition.getNonFrontier()[0], self.composition))
        else: raise ValueError

    def non_frontier_feature_vectors(self) -> dict[tuple,list[float]]:
        d = dict()
        # TODO you can parallelize this (GPU etc)
        for trans in self.composition.getNonFrontier():
            d.update({(trans.state, trans.child): self.extract(trans, self.composition)})
        return d

    def frontier_feature_vectors(self) -> dict[tuple,list[float]]:
        #TODO you can parallelize this (GPU etc)
        #for trans in self.composition.getFrontier():

        return {(trans.state,trans.action) : self.extract(trans, self.composition) for trans in self.composition.getFrontier()}

    def set_static_node_features(self):
        #FIXME refactor this
        for node in self.composition.nodes:
            in_label_ohe = LabelsOHE.compute(self.context, node, dir="in")
            out_label_ohe = LabelsOHE.compute(self.context, node, dir="out")
            marked =  MarkedState.compute(self.composition, node)
            self.composition.nodes[node]["features"] = in_label_ohe + out_label_ohe + marked
            self.composition.nodes[node]["compostate"] = node.toString()

    def global_feature_vectors(self) -> dict:
        raise NotImplementedError

    def train_node2vec(self):
        raise NotImplementedError
    def train_watch_your_step(self):
        raise NotImplementedError
    def train_DGI(self):
        raise NotImplementedError
    def __str__(self):
        return "feature classes: " + str(self._enabled_feature_classes)

    def train_gae_on_full_graph(self, to_undirected=True, epochs=5000, debug_graph=None):
        Warning("This function will be replaced by the official VGAE implementation from DGL")
        # FIXME this should be converted into a Feature class in the future
        # FIXME FIXME the inference is being performed purely on edges!!!!!!!!!!!
        # from torch_geometric.transforms import RandomLinkSplit
        data, device = self.composition_to_nx(debug_graph, to_undirected)

        Warning("We should use RandomNodeSplit")
        Warning("How are negative edge features obtained?")
        splitter = NodePairSplitter(data)
        p_tr, n_tr, p_test, n_test = splitter.get_split()

        out_channels = 2

        num_features = data.x.shape[1]
        # TODO adapt for RandomLinkSplit, continue with tutorial structure
        gnc_ncoder = GCNEncoder(num_features, out_channels)
        model = GAE(gnc_ncoder)

        model = model.to(device)
        node_features = data.x.to(device)
        Warning("This features are only of connected nodes")
        # x_train = train_data.edge_attr.to(device)
        # x_test = test_data.edge_attr.to(device)

        # train_pos_edge_label_index = train_data.pos_edge_label_index.to(device)

        # FIXME how are neg edge features computed if inference is done on edge features and not node features?
        # train_neg_edge_label_index = train_data.neg_edge_label_index.to(device)  # TODO .encode and add to loss and EVAL
        # inizialize the optimizer

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # , , p_test, n_test
            # breakpoint()
            loss = self.train(model, optimizer, node_features, p_tr, n_tr)
            #auc, ap = test(model, p_test, n_test, node_features, data.edge_index, n_tr)
            #print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
            # writer.add_scalar("losses/loss", loss, epoch)
            # writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)
            # writer.add_scalar("metrics/AUC", auc, epoch)
            # writer.add_scalar("metrics/AP", ap, epoch)
        # writer.close()

    def train(self, model, optimizer, features, train_pos_edge_label_index, train_neg_edge_label_index):
        model.train()
        optimizer.zero_grad()

        z = model.encode(features, train_pos_edge_label_index)
        #   breakpoint()
        loss = model.recon_loss(z, train_pos_edge_label_index, train_neg_edge_label_index)
        # if args.variational:
        #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        return float(loss)

    def composition_to_nx(self, debug_graph=None, to_undirected=True, selected_transitions_to_inspect=[]):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(len(self.composition.nodes()), len(self.composition.edges()))
        edge_features = self.non_frontier_feature_vectors()
        self.set_static_node_features()
        CG = self.composition
        # fill attrs with features:
        selected_actions_to_inspect = []
        for ((s, t), features) in edge_features.items():
            # if CG[s][t]["label"] in selected_transitions_to_inspect:
            # selected_actions_to_inspect.append((s.toString(),t.toString(),CG[s][t]["label"]))

            # Codigo original de Marco
            for edge in CG[s][t].values():
                edge["features"] = features

            #CG[s][t]["features"] = features

        D = CG.to_pure_nx()
        G = CG.copy_with_nodes_as_ints(D)

        if to_undirected:
            G = G.to_undirected()  # FIXME what about double edges between nodes?

        data = from_networkx(G, group_node_attrs=["features"], group_edge_attrs=["label"]) if debug_graph is None else \
        debug_graph[0].to(device)
        data.feat = data.x
        return data, device

def util_remove_indices(transition_label):
    res = ""
    for c in transition_label:
        if not c.isdigit(): res += c
    return res

