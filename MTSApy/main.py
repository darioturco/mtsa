from src.experiments import *
from src.fixed_experiment import *

FSP_PATH = "./fsp"
BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]

if __name__ == "__main__":
    instancia = "TL"
    TrainSmallInstanceCheckBigInstance().train(instancia, 2, 2, 4, 4, use_saved_agent=False, reward_shaping=True)
    #TestTrainedInAllInstances().pre_select(instancia, 2000, f"./results/models/{instancia}")







    #TestTrainedInAllInstances().run("AT", 15000, pth_path="./results/models/AT/1/AT-2-2-20100-partial.pth")
    #TestTrainedInAllInstances().run("BW", 15000, pth_path="./results/models/BW/3-3/BW-3-3-2300-partial.pth")
    #TestTrainedInAllInstances().run("DP", 15000, pth_path="./results/models/DP/2-2/DP-2-2-4100-partial.pth")
    #TestTrainedInAllInstances().run("TL", 15000, pth_path="./results/models/TL/1/TL-2-2-33000-partial.pth")
    #TestTrainedInAllInstances().run("TA", 15000, pth_path="./results/models/TA/1/TA-2-2-5600-partial.pth")
    #TestTrainedInAllInstances().run("CM", 15000, pth_path="./results/models/CM/1/CM-2-2-270-partial.pth")
    #TestTrainedInAllInstances().run("AT", 5000)

    #RunRandomInAllInstances().run(5000, 100)
    #RunRAInAllInstances().run(15000)
    #RunRAInAllInstances().run2(15000, instancia)


    #TrainPPO().train(["TL"])

    #TrainSmallInstanceCheckBigInstance().curriculum_train("DP", [{"n": 2, "k": 2, "seconds": None, "max_steps": 300000, "max_eps": 10000, "freq_save": 100},
    #                                                                {"n": 3, "k": 3, "seconds": None, "max_steps": 300000, "max_eps": 8000, "freq_save": 10},
    #                                                                {"n": 4, "k": 4, "seconds": None, "max_steps": 300000, "max_eps": 5000, "freq_save": 1}])


### Notas:
    # En pycharm tengo DP corriendo y luego AT
    # En la primera consola tengo BW corriendo y luego TA
