import random

from src.composition import CompositionGraph, CompositionAnalyzer
from src.environment import Environment
from src.agents.dqn import DQN, NeuralNetwork, TorchModel
from src.experiments import *

FSP_PATH = "./fsp"
BENCHMARK_PROBLEMS = ["AT", "BW", "CM", "DP", "TA", "TL"]

if __name__ == "__main__":
    #path = "F:\\UBA\\Tesis\\MTSApy\\fsp\\Blocking\\ControllableFSPs\\GR1Test10.lts" # For Windows
    path = "F:\\UBA\\Tesis\\MTSApy\\fsp"  # For Windows
    #path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp/Blocking/ControllableFSPs/GR1Test10.lts" # For Linux
    #path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp"  # For Linux
    #test_custom_instance(path, [0, 1, 1, 0, 0, 0, 0])

    #TestTrainInstance("Test").run(path, "DP", 2, 2)
    #TrainSmallInstanceCheckBigInstance().run("TA", 2, 2, 15, 15, use_saved_agent=False)
    TrainSmallerInstanceCheckInAll().run("TA", 2, 2, 3, 3, use_saved_agent=False)


