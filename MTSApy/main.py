from src.experiments import *
from src.fixed_experiment import *
from src.instance_maker import *

if __name__ == "__main__":
    experiment_name = "LRL"
    instance = "TL"
    #TrainSmallInstance().train(instancia, 2, 2, experiment_name)
    #TestTrainedInAllInstances().pre_select(instance, experiment_name, 1000, 99999,
    #                                        [(n, k) for n in range(2, 10) for k in range(2, 10)])
    #                                        [(3, 1), (5, 1)])

    #TestTrainedInAllInstances().run("AT", 10000, pth_path="./results/models/AT/RRL\AT-2-2-6290-partial.pth")
    #TestTrainedInAllInstances().run("BW", 10000, pth_path="./results/models/BW/RRL\BW-2-2-710-partial.pth")
    #TestTrainedInAllInstances().run("DP", 10000, pth_path="./results/models/DP/RRL\DP-2-2-6610-partial.pth")
    #TestTrainedInAllInstances().run("TA", 10000, pth_path="./results/models/TA/RRL\TA-2-2-2580-partial.pth")
    TestTrainedInAllInstances().run("TL", experiment_name, 10000, pth_path="./results/models/TL/LRL\TL-2-2-22240-partial.pth")
    #TestTrainedInAllInstances().run("CM", 10000, pth_path="./results/models/CM/1/CM-2-2-270-partial.pth")





### Notas:


