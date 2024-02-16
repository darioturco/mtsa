from src.experiments import *
from src.fixed_experiment import *
from src.instance_maker import *

FSP_PATH = "./fsp"
BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]

if __name__ == "__main__":
    instancia = "TA"
    #TrainSmallInstance().train(instancia, 2, 2, reward_shaping=False)
    #TestTrainedInAllInstances().pre_select(instancia, 1000, f"./results/models/{instancia}/IERL", 99999, f"./results/selection/IERL-{instancia}.csv")

    #TestTrainedInAllInstances().run("AT", 10000, pth_path="./results/models/AT/IERL/AT-2-2-5460-partial.pth")
    #TestTrainedInAllInstances().run("BW", 10000, pth_path="./results/models/BW/IERL/BW-2-2-6275-partial.pth")
    TestTrainedInAllInstances().run("DP", 1000, pth_path="./results/models/DP/2-2/DP-2-2-20-partial.pth")
    #TestTrainedInAllInstances().run("TL", 10000, pth_path="./results/models/curriculum/TL/TL-4-4-4260-partial.pth")
    #TestTrainedInAllInstances().run("TA", 10000, pth_path="./results/models/TA/IERL/TA-2-2-1455-partial.pth")
    #TestTrainedInAllInstances().run("CM", 10000, pth_path="./results/models/CM/1/CM-2-2-270-partial.pth")





### Notas:


