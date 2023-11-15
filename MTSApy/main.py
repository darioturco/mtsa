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

    #TestTrainedInAllInstances().run("AT", 5000, pth_path="./results/models/AT/1/AT-2-2-20100-partial.pth")
    #TestTrainedInAllInstances().run("BW", 5000, pth_path="./results/models/BW/1/BW-2-2-365300-partial.pth")
    #TestTrainedInAllInstances().run("DP", 5000, pth_path="./results/models/DP/1/DP-2-2-183500-partial.pth")
    #TestTrainedInAllInstances().run("TL", 5000, pth_path="./results/models/TL/1/TL-2-2-33000-partial.pth")
    #TestTrainedInAllInstances().run("TA", 5000, pth_path="./results/models/TA/1/TA-2-2-5600-partial.pth")
    #TestTrainedInAllInstances().run("CM", 5000, pth_path="./results/models/CM/1/CM-2-2-270-partial.pth")
    #TestTrainedInAllInstances().run("AT", 5000)

    #RunRandomInAllInstances().run(5000, 100)
    #RunRAInAllInstances().run(5000)


    #TrainPPO().run(["TA", "CM"])
    #TrainPPO().test("TA", 5000, pth_path=f"results/models/PPO/TA/TA-2-2-12000-partial")
    #TrainPPO().test("TL", 5000, pth_path=f"results/models/PPO/TL/TL-2-2-24000-partial")
    #TrainPPO().test("DP", 5000, pth_path=f"results/models/PPO/DP/DP-2-2-24000-partial")
    TrainPPO().test("BW", 5000, pth_path=f"results/models/PPO/BW/BW-2-2-24000-partial")

    #TrainPPO().test("AT", 5000, pth_path=f"results/models/PPO/AT/AT-2-2-24000-partial")
    #TrainPPO().test("CM", 5000, pth_path=f"results/models/PPO/CM/CM-2-2-24000-partial")

    #TrainGNN().run()

### Notas:
    # ...

