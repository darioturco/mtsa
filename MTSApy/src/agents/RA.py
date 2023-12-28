import numpy as np
from src.agents.agent import Agent

class RA(Agent):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def get_name(self):
        return "RA"

    def get_action(self, s, epsilon, env=None):
        return env.env.context.composition.javaEnv.getActionFronAuxiliarHeuristic()
