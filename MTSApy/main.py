import random

from src.composition import CompositionGraph, CompositionAnalyzer
from src.environment import Environment
from src.agents.dqn import DQN, NeuralNetwork, TorchModel
from src.experiments import *

FSP_PATH = "./fsp"
BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]

if __name__ == "__main__":
    #path = "F:\\UBA\\Tesis\\MTSApy\\fsp\\Blocking\\ControllableFSPs\\GR1Test10.lts" # For Windows
    #path = "F:\\UBA\\Tesis\\MTSApy\\fsp"  # For Windows
    #path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp/Blocking/ControllableFSPs/GR1Test10.lts" # For Linux
    #path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp"  # For Linux
    #test_custom_instance(path, [0, 1, 1, 0, 0, 0, 0])
    #TestTrainInstance("Test").run(path, "TA", 2, 2)

    TrainSmallInstanceCheckBigInstance().run("TL", 2, 2, 4, 4, use_saved_agent=False)
    #TrainSmallerInstanceCheckInAll().run("TA", 2, 2, 3, 3, use_saved_agent=False)

    #RunRandomInAllInstances().run(5000, 100)


