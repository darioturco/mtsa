from src.experiments import *
from src.fixed_experiment import *
from src.instance_maker import *

FSP_PATH = "./fsp"
BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]

if __name__ == "__main__":
    instancia = "DP"
    TrainSmallInstanceCheckBigInstance().train(instancia, 2, 2, 4, 4, use_saved_agent=False, reward_shaping=True)
    #TestTrainedInAllInstances().pre_select(instancia, 2000, f"./results/models/{instancia}")







    #TestTrainedInAllInstances().run("AT", 15000, pth_path="./results/models/AT/1/AT-2-2-20100-partial.pth")
    #TestTrainedInAllInstances().run("BW", 15000, pth_path="./results/models/BW/3-3/BW-3-3-2300-partial.pth")
    #TestTrainedInAllInstances().run("DP", 15000, pth_path="./results/models/DP/2-2/DP-2-2-4100-partial.pth")
    #TestTrainedInAllInstances().run("TL", 15000, pth_path="./results/models/curriculum/TL/TL-4-4-4260-partial.pth")
    #TestTrainedInAllInstances().run("TA", 15000, pth_path="./results/models/TA/1/TA-2-2-5600-partial.pth")
    #TestTrainedInAllInstances().run("CM", 15000, pth_path="./results/models/CM/1/CM-2-2-270-partial.pth")

    #RunRandomInAllInstances().run(15000, 100)
    #RunRAInAllInstances().run(15000, "CM", True, False)


    #TrainPPO().train(["TL"])

    #TrainSmallInstanceCheckBigInstance().curriculum_train("DP", [{"n": 2, "k": 2, "seconds": None, "max_steps": 300000, "max_eps": 10000, "freq_save": 100},
    #                                                                {"n": 3, "k": 3, "seconds": None, "max_steps": 300000, "max_eps": 8000, "freq_save": 10},
    #                                                                {"n": 4, "k": 4, "seconds": None, "max_steps": 300000, "max_eps": 5000, "freq_save": 1}])

    #InstanceMaker(c=5, nc=3, lts=3, substates=5, or_p=0.7, assumptions=1, guaranties=5, p=0.1).make(f"./fsp/Syntetic/test.fsp")
    #for i in range(100):
    #    InstanceMaker(c=5, nc=5, lts=16, substates=5, or_p=0.7, assumptions=0, guaranties=5, p=0.1).make(f"./fsp/Syntetic/instance{i}.fsp")



### Notas:

