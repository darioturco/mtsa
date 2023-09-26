class Environment:
    def __init__(self, contexts, normalize_reward):
        """Environment base class.
            TODO are contexts actually part of the concept of an RL environment?
            """
        self.contexts = contexts
        self.normalize_reward = normalize_reward

    def reset_from_copy(self):
        # Descomentar
        self.contexts = None #[CompositionAnalyzer(context.composition.reset_from_copy()) for context in self.contexts]

        return self

    def get_number_of_contexts(self):
        return len(self.contexts)

    def get_contexts(self):
        return self.contexts

    def step(self, action_idx, context_idx=0):
        composition_graph = self.contexts[context_idx].composition
        composition_graph.expand(action_idx)  # TODO refactor. Analyzer should not be the expansion medium
        if not composition_graph._javaEnv.isFinished():
            return self.actions(), self.reward(), False, {}
        else:
            return None, self.reward(), True, self.get_results()

    def get_results(self, context_idx=0):
        composition_dg = self.contexts[context_idx].composition
        return {
            "synthesis time(ms)": float(composition_dg._javaEnv.getSynthesisTime()),
            "expanded transitions": int(composition_dg._javaEnv.getExpandedTransitions()),
            "expanded states": int(composition_dg._javaEnv.getExpandedStates())
        }

    def reward(self):
        # TODO ?normalize like Learning-Synthesis?
        return -1

    def state(self):
        raise NotImplementedError

    def actions(self, context_idx=0):
        # TODO refactor
        return self.contexts[context_idx].composition.getFrontier()