import random

from src.composition import CompositionGraph, CompositionAnalyzer
from src.environment import Environment
from src.agents.dqn import DQN, NeuralNetwork, TorchModel
from src.experiments import *

FSP_PATH = "./fsp"
BENCHMARK_PROBLEMS = ["AT", "BW", "CM", "DP", "TA", "TL"]

def test_custom_instance(path, idxs=None):
    #d = CompositionGraph("AT", 3, 3, FSP_PATH)
    d = CompositionGraph("Custom", 1, 1, path)
    d.start_composition()
    context = CompositionAnalyzer(d)
    env = Environment(context, False)
    state = env.reset()
    finish = False
    rewards = []
    i = 0

    while not finish:
        if idxs is None:
            n_actions = len(state)
            idx = random.randint(0, n_actions-1)
        else:
            idx = idxs[i]

        state, reward, finish, info = env.step(idx)
        rewards.append(reward)
        i = i + 1

    env.close()



if __name__ == "__main__":
    #path = "F:\\UBA\\Tesis\\MTSApy\\fsp\\Blocking\\ControllableFSPs\\GR1Test10.lts" # For Windows
    #path = "F:\\UBA\\Tesis\\MTSApy\\fsp"  # For Windows
    #path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp/Blocking/ControllableFSPs/GR1Test10.lts" # For Linux
    path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp"  # For Linux
    #test_custom_instance(path, [0, 1, 1, 0, 0, 0, 0])

    #TestTrainInstance("Test").run(path, "DP", 2, 2)
    TrainSmallInstanceChackBigInstance().run("AT", 2, 2, 5, 5)


