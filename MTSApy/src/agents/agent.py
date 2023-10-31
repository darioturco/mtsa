class Agent(object):

    def __init__(self, env):
        self._env = env

    def train(self):
        pass

    def get_action(self, state):
        pass

    def feature_vector_to_number(self, features):
        res = 0
        for i, f in enumerate(features):
            res += round(f) * (2 ** i)

        return res
