import random

from src.composition import CompositionGraph, CompositionAnalyzer
from src.environment import Environment
from src.agents.dqn import DQN, NeuralNetwork, TorchModel
from src.experiments import *

FSP_PATH = "./fsp"
BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]

if __name__ == "__main__":
    #TrainSmallInstanceCheckBigInstance().run("TL", 2, 2, 4, 4, use_saved_agent=False)
    #TrainSmallInstanceCheckBigInstance().run("AT", 2, 2, 4, 4, use_saved_agent=False)
    #TrainSmallInstanceCheckBigInstance().run("BW", 2, 2, 4, 4, use_saved_agent=False)
    #TrainSmallInstanceCheckBigInstance().run("DP", 2, 2, 4, 4, use_saved_agent=False)
    #TrainSmallInstanceCheckBigInstance().run("TA", 2, 2, 4, 4, use_saved_agent=False)
    #TrainSmallInstanceCheckBigInstance().run("CM", 2, 2, 4, 4, use_saved_agent=False)

    TrainMCST().run()

    #TestTrainedInAllInstances().run("TA", 5000, onnx_path="./results/models/TA/TA-2-2-5600-partial.pth")
    #TestTrainedInAllInstances().run("AT", 5000)

    #RunRandomInAllInstances().run(5000, 100)
    #RunRAInAllInstances().run(5000)

    #TrainPPO().run()
    #TrainTrace().run()

    #TestAllPosiblesFix().run()

### Notas:


