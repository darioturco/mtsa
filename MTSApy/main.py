from src.experiments import *
from src.fixed_experiment import *
from src.instance_maker import *

if __name__ == "__main__":
    experiment_name = "RRL" # (10)
    instancia = "TL"
    TrainSmallInstance().train(instancia, 2, 2, experiment_name)
    TestTrainedInAllInstances().pre_select(instancia, 1000, f"./results/models/{instancia}/{experiment_name}", 99999, f"./results/selection/{experiment_name}-{instancia}.csv")

    #TestTrainedInAllInstances().run("AT", 10000, pth_path="./results/models/AT/GRL/AT-2-2-10-partial.pth")
    #TestTrainedInAllInstances().run("BW", 10000, pth_path="./results/models/BW/GRL/BW-2-2-4460-partial.pth")
    #TestTrainedInAllInstances().run("DP", 10000, pth_path="./results/models/DP/GRL/DP-2-2-9335-partial.pth")
    #TestTrainedInAllInstances().run("TA", 10000, pth_path="./results/models/TA/GRL/TA-2-2-1640-partial.pth")
    #TestTrainedInAllInstances().run("TL", 10000, pth_path="./results/models/curriculum/TL/TL-4-4-4260-partial.pth")
    #TestTrainedInAllInstances().run("CM", 10000, pth_path="./results/models/CM/1/CM-2-2-270-partial.pth")





### Notas:


