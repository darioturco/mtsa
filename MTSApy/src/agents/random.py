import random
from src.agents.agent import Agent

class RandomAgent(Agent):

    def __init__(self, env=None):
        super().__init__(env)

    def train(self, *args, **kwargs):
        pass

    def get_action(self, state, *args, **kwargs):
        n_actions = len(state)
        return random.randint(0, n_actions-1)

