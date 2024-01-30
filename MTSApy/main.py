from src.experiments import *
from src.fixed_experiment import *
from src.instance_maker import *

FSP_PATH = "./fsp"
BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]

if __name__ == "__main__":
    instancia = "TA"
    #TrainSmallInstance().train(instancia, 2, 2, reward_shaping=False)
    TestTrainedInAllInstances().pre_select(instancia, 1000, f"./results/models/{instancia}/Adam-ERL")

    #TestTrainedInAllInstances().run("AT", 10000, pth_path="./results/models/AT/2-2\AT-2-2-1900-partial.pth")
    #TestTrainedInAllInstances().run("BW", 10000, pth_path="./results/models/BW/2-2\BW-2-2-4000-partial.pth")
    #TestTrainedInAllInstances().run("DP", 10000, pth_path="./results/models/DP/2-2\DP-2-2-6000-partial.pth")
    #TestTrainedInAllInstances().run("TL", 10000, pth_path="./results/models/curriculum/TL/TL-4-4-4260-partial.pth")
    #TestTrainedInAllInstances().run("TA", 10000, pth_path="./results/models/TA/2-2\TA-2-2-6300-partial.pth")
    #TestTrainedInAllInstances().run("CM", 10000, pth_path="./results/models/CM/1/CM-2-2-270-partial.pth")





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
    # Estoy corriendo Adam, aparentemente le va mucho mejor que SGD
    # En pycharm tengo seleccionando DP
    # En la primera terminal tengo preseleccionando AT
    # En la segunda terminal tengo preseleccionando BW
    # En la tercera terminal tengo preseleccionando TA
