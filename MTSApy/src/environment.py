
### El enviroment debe tener:
#  - una funcion de init
#  - una funion "reset"
#  - una funcion "step"

"""
class Environment:
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        # Assert
        pass

"""
class Environment:
    def __init__(self, context, normalize_reward):
        """Environment base class.
            TODO are contexts actually part of the concept of an RL environment?
            """
        self.context = context
        self.normalize_reward = normalize_reward
        self.info = context.composition.get_info()

    def reset(self):
        # Reset the enviroment
        self.context.composition = self.context.composition.reset_from_copy()
        return self.actions()

    def get_context(self):
        return self.context

    def step(self, action_idx):
        composition_graph = self.context.composition
        composition_graph.expand(action_idx)  # TODO refactor. Analyzer should not be the expansion medium
        if not composition_graph.javaEnv.isFinished():
            return self.actions(), self.reward(), False, {}
        else:
            return None, self.reward(), True, self.get_info()

    def get_info(self):
        composition_dg = self.context.composition
        return {
            "synthesis time(ms)": float(composition_dg.javaEnv.getSynthesisTime()),
            "expanded transitions": int(composition_dg.javaEnv.getExpandedTransitions()),
            "expanded states": int(composition_dg.javaEnv.getExpandedStates())
        }

    def reward(self):
        if self.normalize_reward:
            # TODO: Implement the normalize reward
            # return -1 / self.problem_size
            raise NotImplementedError
        else:
            return -1

    def state(self):
        raise NotImplementedError

    def actions(self):
        return self.context.composition.getFrontier()

    def get_nfeatures(self):
        return self.context.get_transition_features_size()

    def close(self):
        pass
