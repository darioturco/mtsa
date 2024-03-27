from src.experiments import *
from src.fixed_experiment import *
from src.instance_maker import *

if __name__ == "__main__":
    experiment_name = "b_100_instances_1_5"
    instancia = "TA"
    # TrainSmallInstance().train(instancia, 2, 2, experiment_name)

    print(f"Current time {datetime.datetime.now()}")
    TestTrainedInAllInstances().pre_select(instancia,
                                           1000,   # 1000
                                           f"./results/models/{instancia}/{experiment_name}",
                                           99999,
                                           f"./results/selection/{experiment_name}-{instancia}.csv",
                                           [(n, k) for n in range(1, 5) for k in range(1, 5)])
                                           #[(3, 1)])

    #TestTrainedInAllInstances().run("AT", 10000, pth_path="./results/models/AT/RRL\AT-2-2-6290-partial.pth")
    #TestTrainedInAllInstances().run("BW", 10000, pth_path="./results/models/BW/RRL\BW-2-2-710-partial.pth")
    #TestTrainedInAllInstances().run("DP", 10000, pth_path="./results/models/DP/RRL\DP-2-2-6610-partial.pth")
    #TestTrainedInAllInstances().run("TA", 10000, pth_path="results/models/TA/CRL/TA-2-2-130-partial.pth")
    #TestTrainedInAllInstances().run("TL", 10000, pth_path="results/models/TL/RRL/TL-2-2-19200-partial.pth")
    #TestTrainedInAllInstances().run("CM", 10000, pth_path="./results/models/CM/1/CM-2-2-270-partial.pth")





### Notas:


