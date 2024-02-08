from src.experiments import *
from src.fixed_experiment import *
from src.instance_maker import *

FSP_PATH = "./fsp"
BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]

if __name__ == "__main__":
    instancia = "DP"
    TrainSmallInstance().train(instancia, 2, 2, reward_shaping=False)
    #TestTrainedInAllInstances().pre_select(instancia, 1000, f"./results/models/{instancia}/2-2", 99999, f"./results/selection/2-2-{instancia}.csv")

    #TestTrainedInAllInstances().run("AT", 10000, pth_path="./results/models/AT/ERL/AT-2-2-7630-partial.pth")
    #TestTrainedInAllInstances().run("BW", 10000, pth_path="./results/models/BW/ERL/BW-2-2-8285-partial.pth")
    #TestTrainedInAllInstances().run("DP", 10000, pth_path="./results/models/DP/ERL/DP-2-2-2930-partial.pth")
    #TestTrainedInAllInstances().run("TL", 10000, pth_path="./results/models/curriculum/TL/TL-4-4-4260-partial.pth")
    #TestTrainedInAllInstances().run("TA", 10000, pth_path="./results/models/TA/ERL/TA-2-2-2120-partial.pth")
    #TestTrainedInAllInstances().run("CM", 10000, pth_path="./results/models/CM/1/CM-2-2-270-partial.pth")




### Notas:


