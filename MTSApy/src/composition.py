import networkx as nx
import jpype.imports
from bidict import bidict
import sys

if not jpype.isJVMStarted():
    if "linux" in sys.platform:
        jpype.startJVM(classpath=['mtsa.jar'])  # For Linux
    else:
        jpype.startJVM(f"C:\\Program Files\\Java\\jdk-21\\bin\\server\\jvm.dll", '-ea', classpath=['C:/Users/diort/Downloads/mtsa/maven-root/mtsa/target/mtsa.jar'])  # For Windows

NONBLOCKING = True
if NONBLOCKING:
    print("WARNING: Runing NonBlocking environment")
    from MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking import DirectedControllerSynthesisNonBlocking, FeatureBasedExplorationHeuristic, DCSForPython
else:
    from MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking import DirectedControllerSynthesisBlocking, FeatureBasedExplorationHeuristic, DCSForPython

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
    #  - Custom (n and k are ingnored and you can pass a custom path in the start_composition function)

    def __init__(self, problem, n, k, fsp_path):
        super().__init__()
        self._problem, self._n, self._k = problem, n, k
        self._fsp_path = fsp_path
        self._initial_state = None
        self._state_machines = []
        self._frontier = []
        self._started, self._completed = False, False
        self._alphabet = []
        self._no_indices_alphabet = []
        self._number_of_goals = 0
        self._expansion_order = []
        self.javaEnv = None

    def reset_from_copy(self):
        return self.__class__(self._problem, self._n, self._k, self._fsp_path).start_composition()

    def start_composition(self):
        assert (self._initial_state is None)
        self._started = True
        if self._problem == "Custom":
            problem_path = self._fsp_path
        else:
            problem_path = f"{self._fsp_path}/{self._problem}/{self._problem}-{self._n}-{self._k}.fsp"
        c = FeatureBasedExplorationHeuristic.compileFSP(problem_path)
        ltss_init = c.getFirst()
        # TODO: turn it into a dictionary that goes from the state machine name into its respective digraph
        self._state_machines = [m.name for m in ltss_init.machines]
        self.javaEnv = DCSForPython(None, None, 10000, ltss_init)
        assert (self.javaEnv is not None)

        self.javaEnv.startSynthesis(problem_path)
        self._initial_state = self.javaEnv.dcs.initial
        self.add_node(self._initial_state)
        self._alphabet = [e for e in self.javaEnv.dcs.alphabet.actions]
        self._alphabet.sort()
        return self

    def expand(self, idx):
        assert (not self.javaEnv.isFinished()), "Invalid expansion, composition is already solved"
        assert (idx < len(self.getFrontier()) and idx >= 0), "Invalid index"
        self.javaEnv.expandAction(idx)  # TODO check this is the same index as in the python frontier list
        new_state_action = self.getLastExpanded()
        controllability, label = self.getLastExpanded().action.isControllable(), self.getLastExpanded().action.toString()
        self.add_node(self.last_expansion_child_state())
        self.add_edge(new_state_action.state, self.last_expansion_child_state(), controllability=controllability,
                      label=label, action_with_features=new_state_action)
        self._expansion_order.append(self.getLastExpanded())

    def last_expansion_child_state(self):
        return self.javaEnv.heuristic.lastExpandedTo

    def last_expansion_source_state(self):
        return self.javaEnv.heuristic.lastExpandedFrom

    def getFrontier(self): return self.javaEnv.heuristic.explorationFrontier

    def getLastExpanded(self): return self.javaEnv.heuristic.lastExpandedStateAction

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
        return {"problem": self._problem, "n": self._n, "k": self._k}


class CompositionAnalyzer:
    """class used to get Composition information, usable as hand-crafted features
        TODO this class will be replaced by object-oriented Feature class
    """

    def __init__(self, composition):
        self.composition = composition
        self.nfeatures = None
        assert (self.composition._started)

        self._no_indices_alphabet = list(set([self.remove_indices(str(e)) for e in composition._alphabet]))
        self._no_indices_alphabet.sort()
        self._fast_no_indices_alphabet_dict = dict()
        for i in range(len(self._no_indices_alphabet)): self._fast_no_indices_alphabet_dict[
            self._no_indices_alphabet[i]] = i
        self._fast_no_indices_alphabet_dict = bidict(self._fast_no_indices_alphabet_dict)
        self._feature_methods = [self.event_label_feature, self.state_label_feature, self.controllable
            , self.marked_state, self.current_phase, self.child_node_state,
                                 self.uncontrollable_neighborhood, self.explored_state_child, self.isLastExpanded]

    def test_features_on_transition(self, transition):
        res = []
        for compute_feature in self._feature_methods:
            res += compute_feature(transition)
        return [float(e) for e in res]

    def event_label_feature(self, transition):
        """
        Determines the label of ℓ in A E p .
        """
        feature_vec_slice = [0 for _ in self._no_indices_alphabet]
        self._set_transition_type_bit(feature_vec_slice, transition.action)
        # print(no_idx_label, feature_vec_slice)
        return feature_vec_slice

    def _set_transition_type_bit(self, feature_vec_slice, transition):
        no_idx_label = self.remove_indices(str(transition.toString()))
        feature_vec_slice_pos = self._fast_no_indices_alphabet_dict[no_idx_label]
        feature_vec_slice[feature_vec_slice_pos] = 1

    def state_label_feature(self, transition):
        """
        Determines the labels of the explored
            transitions that arrive at s.
        """
        feature_vec_slice = [0 for _ in self._no_indices_alphabet]
        arriving_to_s = transition.state.getParents()
        for trans in arriving_to_s: self._set_transition_type_bit(feature_vec_slice, trans.getFirst())
        return feature_vec_slice

    def controllable(self, transition):
        return [float(transition.action.isControllable())]

    def marked_state(self, transition):
        """Whether s and s ′ ∈ M E p ."""
        # Le falta una feature a esta funcion (deberia devolver una lista de 2 elementos)
        return [float(transition.childMarked)]

    def current_phase(self, transition):
        return [float(self.composition.javaEnv.dcs.heuristic.goals_found > 0),
                float(self.composition.javaEnv.dcs.heuristic.marked_states_found > 0),
                float(self.composition.javaEnv.dcs.heuristic.closed_potentially_winning_loops > 0)]

    def child_node_state(self, transition):
        """Whether
        s ′ is winning, losing, none,
        or not yet
        explored."""
        res = [0, 0, 0]
        if transition.child is not None:
            res = [float(transition.child.status.toString() == "GOAL"),
                   float(transition.child.status.toString() == "ERROR"),
                   float(transition.child.status.toString() == "NONE")]
        return res

    def uncontrollable_neighborhood(self, transition):
        return [float(transition.state.uncontrollableUnexploredTransitions > 0),
                float(transition.state.uncontrollableTransitions > 0),
                float(transition.child is None or transition.child.uncontrollableUnexploredTransitions > 0),
                float(transition.child is None or transition.child.uncontrollableTransitions > 0)
                ]

    def explored_state_child(self, transition):
        f1 = float(len(self.composition.out_edges(transition.state)) != transition.state.unexploredTransitions)
        f2 = float(transition.child is not None and len(self.composition.out_edges(transition.child)) != transition.state.unexploredTransitions)
        return [f1, f2]

    def isLastExpanded(self, transition):
        return [float(self.composition.getLastExpanded() == transition)]

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