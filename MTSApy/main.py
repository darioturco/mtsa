import random

from src.composition import CompositionGraph, CompositionAnalyzer
from src.environment import Environment
from src.agents.dqn import DQN, NeuralNetwork, TorchModel
from src.experiments import *

FSP_PATH = "./fsp"
BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]

if __name__ == "__main__":
    #TrainSmallInstanceCheckBigInstance().run("DP", 2, 2, 4, 4, use_saved_agent=False)

    #TestTrainedInAllInstances().run("TA", 5000, onnx_path="./results/models/TA/TA-2-2-5600-partial.pth")
    #TestTrainedInAllInstances().run("AT", 5000)

    #RunRandomInAllInstances().run(5000, 100)
    #RunRAInAllInstances().run(5000)

    TrainPPO().run()

### Notas:
    # En la consola esta corriendo DP
    # Y en Pycharm corre TA

