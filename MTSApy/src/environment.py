
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
import numpy as np


class Environment:
    def __init__(self, context, normalize_reward):
        """Environment base class.
            TODO are contexts actually part of the concept of an RL environment?
            """
        self.context = context
        self.normalize_reward = normalize_reward
        self.info = context.composition.get_info()

    def reset(self, new_composition=None):
        # Reset the enviroment
        if new_composition is None:
            self.context.composition = self.context.composition.reset_from_copy()
        else:
            self.context.composition = new_composition.reset_from_copy()
            self.info = new_composition.get_info()
        return self.states()

    def get_context(self):
        return self.context

    def step(self, action_idx):
        composition_graph = self.context.composition
        composition_graph.expand(action_idx)  # TODO refactor. Analyzer should not be the expansion medium
        if composition_graph.javaEnv.isFinished():
            return None, self.reward(), True, self.get_info()
        else:
            return self.states(), self.reward(), False, {}


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

    def states(self):
        return self.context.composition.getFrontier()

    def get_nfeatures(self):
        return self.context.get_transition_features_size()

    def actions_to_features(self, actions):
        if actions is None:
            return []
        return [self.context.compute_features(action) for action in actions]

    def get_instance_info(self):
        info = self.context.composition.get_info()
        return info["problem"], info["n"], info["k"]

    def close(self):
        pass

#import gym
class FeatureEnvironment(object):
    def __init__(self, context, normalize_reward):
        self.env = Environment(context, normalize_reward)

        # self.state_size = env.observation_space.shape[0]
        #         self.num_actions = env.action_space.n

        self.observation_space_dim = self.env.context.get_transition_features_size()
        self.action_space = [0, 1]



    def reset(self):
        state = self.env.reset()
        return self.env.actions_to_features(state), False
        #return self.env.actions_to_features(state)[0], False

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return self.env.actions_to_features(state), reward, done, False, info


    def close(self):
        self.env.close()



