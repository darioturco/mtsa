import networkx as nx
import jpype.imports
from bidict import bidict
import sys
import random

if not jpype.isJVMStarted():
    if "linux" in sys.platform:
        jpype.startJVM(classpath=['mtsa.jar'])  # For Linux
    else:
        jpype.startJVM(f"C:\\Program Files\\Java\\jdk-21\\bin\\server\\jvm.dll", '-ea', classpath=['F:/UBA/Tesis/mtsa/MTSApy/mtsa.jar'])  # For Windows

NONBLOCKING = False
if NONBLOCKING:
    print("WARNING: Runing NonBlocking environment")
    from MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking import DirectedControllerSynthesisNonBlocking, FeatureBasedExplorationHeuristic, DCSForPython
else:
    from MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking import DCSForPython

#FSP_PATH = "../fsp"
#BENCHMARK_PROBLEMS = ["AT", "BW", "CM", "DP", "TA", "TL"]

class CompositionGraph(nx.DiGraph):
    # The CompositionGraph requires the name of the problem, and its sizes
    # The name can be:
    #  - AT (Air Trafic)
    #  - DP (Dinner Philosophers)
    #  - BW (Bidding Workflow)
    #  - TL (Transfer Line)
    #  - TA (Travel Agency)
    #  - CM (Cat and Mouse)
    #  - Custom (n and k are ignored and you can pass a custom path in the start_composition function)

    @staticmethod
    def getDCSForPython():
        return DCSForPython

    def __init__(self, problem, feature_group, n, k, fsp_path):
        super().__init__()
        self.problem, self.n, self.k = problem, n, k
        self.feature_group = feature_group
        self._fsp_path = fsp_path
        self._initial_state = None
        self._frontier = []
        self._started, self._completed = False, False
        self._alphabet = []
        self._no_indices_alphabet = []
        self._number_of_goals = 0
        self._expansion_order = []
        self.javaEnv = None

    def reset_from_copy(self):
        return self.__class__(self.problem, self.feature_group, self.n, self.k, self._fsp_path).start_composition()

    def set_composition_parameters(self):
        # self.auxiliar_heuristic = "BFS"
        # self.auxiliar_heuristic = "Debugging"
        self.auxiliar_heuristic = "Ready"

    def start_composition(self):
        assert (self._initial_state is None)
        self._started = True
        if self.problem == "Custom":
            problem_path = self._fsp_path
        else:
            problem_path = f"{self._fsp_path}/{self.problem}/{self.problem}-{self.n}-{self.k}.fsp"

        self.set_composition_parameters()
        self.javaEnv = DCSForPython(self.auxiliar_heuristic)
        assert (self.javaEnv is not None)
        self.javaEnv.startSynthesis(problem_path)

        self._initial_state = self.javaEnv.dcs.initial
        self.add_node(str(self._initial_state.toString()))
        self._alphabet = [e for e in self.javaEnv.dcs.alphabet.actions]
        self._alphabet.sort()
        return self

    def expand(self, idx):
        assert (not self.javaEnv.isFinished()), "Invalid expansion, composition is already solved"
        assert (idx < len(self.getFrontier()) and idx >= 0), "Invalid index"
        self.javaEnv.expandAction(idx)
        new_state_action = self.getLastExpanded()
        controllability, label = self.getLastExpanded().action.isControllable(), self.getLastExpanded().action.toString()
        self.add_node(str(self.last_expansion_child_state().toString()))
        self.add_edge(new_state_action.state, self.last_expansion_child_state(), controllability=controllability,
                      label=label, action_with_features=new_state_action)
        self._expansion_order.append(self.getLastExpanded())

    def last_expansion_child_state(self):
        return self.javaEnv.heuristic.lastExpandedTo

    def last_expansion_source_state(self):
        return self.javaEnv.heuristic.lastExpandedFrom

    def getFrontier(self):
        return self.javaEnv.heuristic.actionsToExplore

    def getLastExpanded(self):
        return self.javaEnv.heuristic.lastExpandedStateAction

    def _check_no_repeated_states(self):
        raise NotImplementedError

    def explored(self, transition):
        """
        TODO
        Whether a transition from s or s ′ has
            already been explored.

        """
        raise NotImplementedError

    def last_expanded(self, transition):
        """
        TODO
        Whether s is the last expanded state in h
            (outgoing or incoming)."""
        raise NotImplementedError

    def finished(self):
        return self.javaEnv.isFinished()

    def get_info(self):
        return {"problem": self.problem, "n": self.n, "k": self.k}


class CompositionAnalyzer:
    def get_alphabet_without_indices(self, alphabet):
        res_set = set([self.remove_indices(str(e)) for e in self.composition._alphabet])
        if self.composition.problem == "AT":
            res_set.add('air.crash.')

        return list(res_set)

    def __init__(self, composition):
        assert composition._started

        self.composition = composition
        self.nfeatures = None
        self._no_indices_alphabet = self.get_alphabet_without_indices(composition._alphabet)
        self._no_indices_alphabet.sort()
        self._fast_no_indices_alphabet_dict = dict()
        for i in range(len(self._no_indices_alphabet)): self._fast_no_indices_alphabet_dict[
            self._no_indices_alphabet[i]] = i
        self._fast_no_indices_alphabet_dict = bidict(self._fast_no_indices_alphabet_dict)

        self._feature_methods = self.select_feature_group(self.composition.feature_group)

    def select_feature_group(self, feature_group_name):
        if feature_group_name == "GRL":
            # Nuevos features globales (GRL)
            return [self.event_label_feature, self.state_label_feature, self.controllable, self.marked_state,
                    self.current_phase, self.child_node_state, self.uncontrollable_neighborhood,
                    self.explored_state_child, self.isLastExpanded, self.child_dealdlock, self.missions_end]

        elif feature_group_name == "LRL":
            # Nuevos features locales mejorados (LRL)
            return [self.event_label_feature, self.state_label_feature, self.controllable, self.marked_state,
                    self.current_phase, self.child_node_state, self.uncontrollable_neighborhood,
                    self.explored_state_child, self.isLastExpanded, self.child_dealdlock, self.mission_feature,
                    self.has_index, self.entity_state_move]

        elif feature_group_name == "ERL": # No importa (No da buenos resultados)
            # Nuevos features locales (ERL)
            return [self.event_label_feature, self.state_label_feature, self.controllable, self.marked_state,
                    self.current_phase, self.child_node_state, self.uncontrollable_neighborhood,
                    self.explored_state_child, self.isLastExpanded, self.child_dealdlock, self.mission_feature]

        elif feature_group_name == "2-2":
            # Viejos features (2-2)
            return [self.event_label_feature, self.state_label_feature, self.controllable, self.marked_stateOld,
                    self.current_phase, self.child_node_state, self.uncontrollable_neighborhood,
                    self.explored_state_child, self.isLastExpanded]

        elif feature_group_name == "CRL":   # CRL: Custom RL (Es LRL pero agregando feature custom de para cada dominio)
            return [self.event_label_feature, self.state_label_feature, self.controllable, self.marked_state,
                    self.current_phase, self.child_node_state, self.uncontrollable_neighborhood,
                    self.explored_state_child, self.isLastExpanded, self.child_dealdlock, self.mission_feature,
                    self.has_index, self.entity_state_move, self.custom_feature]


        elif feature_group_name == "RRL":   # RRL: Random RL (Es lo mismo que LRL pero con un feature random)
            self.r_feature = 100
            return [self.event_label_feature, self.state_label_feature, self.controllable, self.marked_state,
                    self.current_phase, self.child_node_state, self.uncontrollable_neighborhood,
                    self.explored_state_child, self.isLastExpanded, self.child_dealdlock, self.mission_feature,
                    self.has_index, self.entity_state_move, self.random_feature]

        #elif feature_group_name == "BWFeatures":
            # Es igual a LRL pero con el feature de last expanded (la idea es que BW mejore con esto)
        #    return [self.event_label_feature, self.state_label_feature, self.controllable, self.marked_state,
        #            self.current_phase, self.child_node_state, self.uncontrollable_neighborhood,
        #            self.explored_state_child, self.isLastExpanded, self.child_dealdlock, self.mission_feature,
        #            self.has_index, self.bw_feature]

        else:
            assert False, "Incorrect feature group name"

    def random_feature(self, transiton):
        return [random.choice([1.0, 0.0]) for _ in range(self.composition.r_feature)]

    def last_entity(self, transition):
        return [float(transition.entity == self.composition.javaEnv.lastEntityExpanded), float(transition.entity == self.composition.javaEnv.lastEntityExpandedWithoutReset)]

    def has_index(self, transition):
        label = str(transition.action.toString())
        return [float(any(char.isdigit() for char in label))]

    def entity_state_move(self, transition):
        return [float(transition.upIndex), float(transition.downIndex)]

    def mission_feature(self, transition):
        # Es lo mismo que transiton.getMissionValue(0) TODO: mejorar y Unir
        return [float(transition.missionComplete)]

    def custom_feature(self, transition):
        return [float(transition.getMissionValue(i)) for i in range(1, int(transition.dcs.instanceDomain.f))]

    def missions_end(self, transition):
        return [float(transition.amountMissionComplete > 0), float(transition.amountMissionComplete > 1)]

    def event_label_feature(self, transition):
        """
        Determines the label of ℓ in A E p .
        """
        feature_vec_slice = [0.0 for _ in self._no_indices_alphabet]
        self._set_transition_type_bit(feature_vec_slice, transition.action)
        return feature_vec_slice

    def _set_transition_type_bit(self, feature_vec_slice, transition):
        no_idx_label = self.remove_indices(str(transition.toString()))
        feature_vec_slice_pos = self._fast_no_indices_alphabet_dict[no_idx_label]
        feature_vec_slice[feature_vec_slice_pos] = 1.0

    def state_label_feature(self, transition):
        """
        Determines the labels of the explored
            transitions that arrive at s.
        """
        feature_vec_slice = [0.0 for _ in self._no_indices_alphabet]
        arriving_to_s = transition.state.getParents()
        for trans in arriving_to_s:
            self._set_transition_type_bit(feature_vec_slice, trans.getFirst())
        return feature_vec_slice

    def controllable(self, transition):
        return [float(transition.action.isControllable())]

    def marked_state(self, transition):
        """Whether s and s′ ∈ M E p ."""
        return [float(transition.state.isMarked()), float(transition.childMarked)]

    def marked_stateOld(self, transition):
        """Whether s ∈ M E p ."""
        return [float(transition.state.isMarked())]

    def current_phase(self, transition):
        return [float(self.composition.javaEnv.dcs.heuristic.goals_found > 0),
                float(self.composition.javaEnv.dcs.heuristic.marked_states_found > 0),
                float(self.composition.javaEnv.dcs.heuristic.closed_potentially_winning_loops > 0)]

    def child_node_state(self, transition):
        """Whether
        s ′ is winning, losing, none,
        or not yet
        explored."""
        res = [0.0, 0.0, 0.0]
        if transition.child is not None:
            res = [float(str(transition.child.status.toString()) == "GOAL"),
                   float(str(transition.child.status.toString()) == "ERROR"),
                   float(str(transition.child.status.toString()) == "NONE")]
        return res

    def uncontrollable_neighborhood(self, transition):
        return [float(transition.state.uncontrollableUnexploredTransitions > 0),
                float(transition.state.uncontrollableTransitions > 0),
                float(transition.child is None or transition.child.uncontrollableUnexploredTransitions > 0),
                float(transition.child is None or transition.child.uncontrollableTransitions > 0)
                ]

    def explored_state_child(self, transition):
        f1 = len(self.composition.out_edges(str(transition.state.toString()))) != transition.state.unexploredTransitions
        f2 = transition.child is not None and len(self.composition.out_edges(str(transition.child.toString()))) != transition.state.unexploredTransitions
        return [float(f1), float(f2)]

    def child_dealdlock(self, transition):
        return [float(transition.child is not None and len(transition.child.getTransitions()) == 0)]

    def isLastExpanded(self, transition):
        return [float(transition.state == transition.dcs.heuristic.lastExpandedTo), float(transition.state == transition.dcs.heuristic.lastExpandedFrom)]

    def remove_indices(self, transition_label: str):
        res = ""
        for c in transition_label:
            if not c.isdigit(): res += c
        return res

    def get_transition_features_size(self):
        if self.nfeatures is None:
            return len(self.compute_features(self.composition.getFrontier()[0]))
        else:
            return self.nfeatures

    def compute_features(self, transition):
        res = []
        for feature_method in self._feature_methods:
            res += feature_method(transition)

        if self.nfeatures is None:
            self.nfeatures = len(res)

        return res

    def compute_feature_of_list(self, transactions):
        return [self.compute_features(t) for t in transactions]