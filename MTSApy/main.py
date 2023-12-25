from src.experiments import *
from src.fixed_experiment import *

FSP_PATH = "./fsp"
BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]

if __name__ == "__main__":
    #instancia = "DP"
    #TrainSmallInstanceCheckBigInstance().train(instancia, 2, 2, 4, 4, use_saved_agent=False, reward_shaping=True)
    #TestTrainedInAllInstances().pre_select(instancia, 2000, f"./results/models/{instancia}")







    #TestTrainedInAllInstances().run("AT", 15000, pth_path="./results/models/AT/1/AT-2-2-20100-partial.pth")
    #TestTrainedInAllInstances().run("BW", 15000, pth_path="./results/models/BW/3-3/BW-3-3-2300-partial.pth")
    #TestTrainedInAllInstances().run("DP", 15000, pth_path="./results/models/DP/2-2/DP-2-2-4100-partial.pth")
    #TestTrainedInAllInstances().run("TL", 15000, pth_path="./results/models/TL/1/TL-2-2-33000-partial.pth")
    #TestTrainedInAllInstances().run("TA", 15000, pth_path="./results/models/TA/1/TA-2-2-5600-partial.pth")
    #TestTrainedInAllInstances().run("CM", 15000, pth_path="./results/models/CM/1/CM-2-2-270-partial.pth")
    #TestTrainedInAllInstances().run("AT", 5000)

    #RunRandomInAllInstances().run(5000, 100)
    RunRAInAllInstances().run(15000, "CM", True, False)


    #TrainPPO().train(["TL"])

    #TrainSmallInstanceCheckBigInstance().curriculum_train("DP", [{"n": 2, "k": 2, "seconds": None, "max_steps": 300000, "max_eps": 10000, "freq_save": 100},
    #                                                                {"n": 3, "k": 3, "seconds": None, "max_steps": 300000, "max_eps": 8000, "freq_save": 10},
    #                                                                {"n": 4, "k": 4, "seconds": None, "max_steps": 300000, "max_eps": 5000, "freq_save": 1}])


### Notas:

"""
Runing DQN Agent in instance: AT 2-2
DQN Agent: 
 Syntesis Time: 196.0ms
 Expanded Transactions: 41
 Expanded states: 28

Runing DQN Agent in instance: AT 2-3
DQN Agent: 
 Syntesis Time: 77.0ms
 Expanded Transactions: 48
 Expanded states: 34

Runing DQN Agent in instance: AT 2-4
DQN Agent: 
 Syntesis Time: 319.0ms
 Expanded Transactions: 78
 Expanded states: 53

Runing DQN Agent in instance: AT 2-5
DQN Agent: 
 Syntesis Time: 559.0ms
 Expanded Transactions: 89
 Expanded states: 64

Runing DQN Agent in instance: AT 2-6
DQN Agent: 
 Syntesis Time: 499.0ms
 Expanded Transactions: 96
 Expanded states: 73

Runing DQN Agent in instance: AT 2-7
DQN Agent: 
 Syntesis Time: 617.0ms
 Expanded Transactions: 109
 Expanded states: 84

Runing DQN Agent in instance: AT 2-8
DQN Agent: 
 Syntesis Time: 883.0ms
 Expanded Transactions: 131
 Expanded states: 97

Runing DQN Agent in instance: AT 2-9
DQN Agent: 
 Syntesis Time: 990.0ms
 Expanded Transactions: 135
 Expanded states: 106

Runing DQN Agent in instance: AT 2-10
DQN Agent: 
 Syntesis Time: 1246.0ms
 Expanded Transactions: 148
 Expanded states: 117

Runing DQN Agent in instance: AT 2-11
DQN Agent: 
 Syntesis Time: 1775.0ms
 Expanded Transactions: 173
 Expanded states: 130

Runing DQN Agent in instance: AT 2-12
DQN Agent: 
 Syntesis Time: 2380.0ms
 Expanded Transactions: 187
 Expanded states: 141

Runing DQN Agent in instance: AT 2-13
DQN Agent: 
 Syntesis Time: 3043.0ms
 Expanded Transactions: 201
 Expanded states: 152

Runing DQN Agent in instance: AT 2-14
DQN Agent: 
 Syntesis Time: 3014.0ms
 Expanded Transactions: 200
 Expanded states: 161

Runing DQN Agent in instance: AT 2-15
DQN Agent: 
 Syntesis Time: 4259.0ms
 Expanded Transactions: 229
 Expanded states: 174

Runing DQN Agent in instance: AT 3-2
DQN Agent: 
 Syntesis Time: 12.0ms
 Expanded Transactions: 15
 Expanded states: 16

Runing DQN Agent in instance: AT 3-3
DQN Agent: 
 Syntesis Time: 2106.0ms
 Expanded Transactions: 356
 Expanded states: 221

Runing DQN Agent in instance: AT 3-4
DQN Agent: 
 Syntesis Time: 5581.0ms
 Expanded Transactions: 546
 Expanded states: 339

Runing DQN Agent in instance: AT 3-5
DQN Agent: 
 Syntesis Time: 14598.0ms
 Expanded Transactions: 842
 Expanded states: 513

Runing DQN Agent in instance: AT 3-6
DQN Agent: 
 Syntesis Time: 34738.0ms
 Expanded Transactions: 1125
 Expanded states: 691

Runing DQN Agent in instance: AT 3-7
DQN Agent: 
 Syntesis Time: 72799.0ms
 Expanded Transactions: 1475
 Expanded states: 934

Runing DQN Agent in instance: AT 3-8
DQN Agent: 
 Syntesis Time: 145553.0ms
 Expanded Transactions: 1991
 Expanded states: 1266

Runing DQN Agent in instance: AT 3-9
DQN Agent: 
 Syntesis Time: 253144.0ms
 Expanded Transactions: 2528
 Expanded states: 1634

Runing DQN Agent in instance: AT 3-10
DQN Agent: 
 Syntesis Time: 438370.0ms
 Expanded Transactions: 3148
 Expanded states: 2070

Runing DQN Agent in instance: AT 3-11
DQN Agent: 
 Syntesis Time: 670918.0ms
 Expanded Transactions: 3735
 Expanded states: 2538

Runing DQN Agent in instance: AT 3-12
DQN Agent: 
 Syntesis Time: 1106637.0ms
 Expanded Transactions: 4697
 Expanded states: 3179

Runing DQN Agent in instance: AT 3-13
DQN Agent: 
 Syntesis Time: 1662912.0ms
 Expanded Transactions: 5608
 Expanded states: 3856

Runing DQN Agent in instance: AT 3-14
DQN Agent: 
 Syntesis Time: 2420025.0ms
 Expanded Transactions: 6647
 Expanded states: 4625

Runing DQN Agent in instance: AT 3-15
DQN Agent: 
 Syntesis Time: 3356606.0ms
 Expanded Transactions: 7655
 Expanded states: 5443

Runing DQN Agent in instance: AT 4-2
DQN Agent: 
 Syntesis Time: 24.0ms
 Expanded Transactions: 15
 Expanded states: 16

Runing DQN Agent in instance: AT 4-3
DQN Agent: 
 Syntesis Time: 400.0ms
 Expanded Transactions: 64
 Expanded states: 65

Runing DQN Agent in instance: AT 4-4
DQN Agent: 
 Syntesis Time: 142497.0ms
 Expanded Transactions: 3836
 Expanded states: 2260

Runing DQN Agent in instance: AT 4-5
DQN Agent: 
 Syntesis Time: 480597.0ms
 Expanded Transactions: 6125
 Expanded states: 3540

Runing DQN Agent in instance: AT 4-6
DQN Agent: 
 Syntesis Time: 1489534.0ms
 Expanded Transactions: 9685
 Expanded states: 5514

Runing DQN Agent in instance: AT 4-7
DQN Agent: 
 Syntesis Time: 4018679.0ms
 Expanded Transactions: 13869
 Expanded states: 7883

Runing DQN Agent in instance: AT 4-8
DQN Agent: 
 Syntesis Time: 7800174.0ms
 Expanded Transactions: 15001
 Expanded states: 9680

DQN Agent in instance: AT 4-9: Failed
DQN Agent in instance: AT 4-10: Failed
DQN Agent in instance: AT 4-11: Failed
DQN Agent in instance: AT 4-12: Failed
DQN Agent in instance: AT 4-13: Failed
DQN Agent in instance: AT 4-14: Failed
DQN Agent in instance: AT 4-15: Failed
Runing DQN Agent in instance: AT 5-2
DQN Agent: 
 Syntesis Time: 32.0ms
 Expanded Transactions: 15
 Expanded states: 16

Runing DQN Agent in instance: AT 5-3
DQN Agent: 
 Syntesis Time: 504.0ms
 Expanded Transactions: 64
 Expanded states: 65

Runing DQN Agent in instance: AT 5-4
DQN Agent: 
 Syntesis Time: 4205.0ms
 Expanded Transactions: 325
 Expanded states: 326

Runing DQN Agent in instance: AT 5-5
DQN Agent: 
 Syntesis Time: 3851992.0ms
 Expanded Transactions: 15001
 Expanded states: 11023

DQN Agent in instance: AT 5-6: Failed
DQN Agent in instance: AT 5-7: Failed
DQN Agent in instance: AT 5-8: Failed
DQN Agent in instance: AT 5-9: Failed
DQN Agent in instance: AT 5-10: Failed
DQN Agent in instance: AT 5-11: Failed
DQN Agent in instance: AT 5-12: Failed
DQN Agent in instance: AT 5-13: Failed
DQN Agent in instance: AT 5-14: Failed
DQN Agent in instance: AT 5-15: Failed
Runing DQN Agent in instance: AT 6-2
DQN Agent: 
 Syntesis Time: 41.0ms
 Expanded Transactions: 15
 Expanded states: 16

Runing DQN Agent in instance: AT 6-3
DQN Agent: 
 Syntesis Time: 614.0ms
 Expanded Transactions: 64
 Expanded states: 65

Runing DQN Agent in instance: AT 6-4
DQN Agent: 
 Syntesis Time: 5907.0ms
 Expanded Transactions: 325
 Expanded states: 326

Runing DQN Agent in instance: AT 6-5
DQN Agent: 
 Syntesis Time: 140455.0ms
 Expanded Transactions: 1956
 Expanded states: 1957

Runing DQN Agent in instance: AT 6-6
DQN Agent: 
 Syntesis Time: 7409197.0ms
 Expanded Transactions: 15001
 Expanded states: 14208

DQN Agent in instance: AT 6-7: Failed
DQN Agent in instance: AT 6-8: Failed
DQN Agent in instance: AT 6-9: Failed
DQN Agent in instance: AT 6-10: Failed
DQN Agent in instance: AT 6-11: Failed
DQN Agent in instance: AT 6-12: Failed
DQN Agent in instance: AT 6-13: Failed
DQN Agent in instance: AT 6-14: Failed
DQN Agent in instance: AT 6-15: Failed
Runing DQN Agent in instance: AT 7-2
DQN Agent: 
 Syntesis Time: 63.0ms
 Expanded Transactions: 15
 Expanded states: 16

Runing DQN Agent in instance: AT 7-3
DQN Agent: 
 Syntesis Time: 432.0ms
 Expanded Transactions: 64
 Expanded states: 65

Runing DQN Agent in instance: AT 7-4
DQN Agent: 
 Syntesis Time: 7587.0ms
 Expanded Transactions: 325
 Expanded states: 326

Runing DQN Agent in instance: AT 7-5
DQN Agent: 
 Syntesis Time: 194795.0ms
 Expanded Transactions: 1956
 Expanded states: 1957

Runing DQN Agent in instance: AT 7-6
DQN Agent: 
 Syntesis Time: 6899838.0ms
 Expanded Transactions: 13699
 Expanded states: 13700

Runing DQN Agent in instance: AT 7-7
DQN Agent: 
 Syntesis Time: 23315357.0ms
 Expanded Transactions: 15001
 Expanded states: 15002

DQN Agent in instance: AT 7-8: Failed
DQN Agent in instance: AT 7-9: Failed
DQN Agent in instance: AT 7-10: Failed
DQN Agent in instance: AT 7-11: Failed
DQN Agent in instance: AT 7-12: Failed
DQN Agent in instance: AT 7-13: Failed
DQN Agent in instance: AT 7-14: Failed
DQN Agent in instance: AT 7-15: Failed
Runing DQN Agent in instance: AT 8-2
DQN Agent: 
 Syntesis Time: 64.0ms
 Expanded Transactions: 15
 Expanded states: 16

Runing DQN Agent in instance: AT 8-3
DQN Agent: 
 Syntesis Time: 744.0ms
 Expanded Transactions: 64
 Expanded states: 65

Runing DQN Agent in instance: AT 8-4
DQN Agent: 
 Syntesis Time: 9160.0ms
 Expanded Transactions: 325
 Expanded states: 326

Runing DQN Agent in instance: AT 8-5
DQN Agent: 
 Syntesis Time: 239676.0ms
 Expanded Transactions: 1956
 Expanded states: 1957

Runing DQN Agent in instance: AT 8-6
DQN Agent: 
 Syntesis Time: 8936159.0ms
 Expanded Transactions: 13699
 Expanded states: 13700

Runing DQN Agent in instance: AT 8-7
DQN Agent: 
 Syntesis Time: 27182883.0ms
 Expanded Transactions: 15001
 Expanded states: 15002

DQN Agent in instance: AT 8-8: Failed
DQN Agent in instance: AT 8-9: Failed
DQN Agent in instance: AT 8-10: Failed
DQN Agent in instance: AT 8-11: Failed
DQN Agent in instance: AT 8-12: Failed
DQN Agent in instance: AT 8-13: Failed
DQN Agent in instance: AT 8-14: Failed
DQN Agent in instance: AT 8-15: Failed
Runing DQN Agent in instance: AT 9-2
DQN Agent: 
 Syntesis Time: 66.0ms
 Expanded Transactions: 15
 Expanded states: 16

Runing DQN Agent in instance: AT 9-3
DQN Agent: 
 Syntesis Time: 785.0ms
 Expanded Transactions: 64
 Expanded states: 65

Runing DQN Agent in instance: AT 9-4
DQN Agent: 
 Syntesis Time: 10799.0ms
 Expanded Transactions: 325
 Expanded states: 326

Runing DQN Agent in instance: AT 9-5
DQN Agent: 
 Syntesis Time: 293233.0ms
 Expanded Transactions: 1956
 Expanded states: 1957

Runing DQN Agent in instance: AT 9-6
DQN Agent: 
 Syntesis Time: 11251196.0ms
 Expanded Transactions: 13699
 Expanded states: 13700

Runing DQN Agent in instance: AT 9-7

DQN Agent: 
 Syntesis Time: 496.0ms
 Expanded Transactions: 15
 Expanded states: 16

Runing DQN Agent in instance: AT 10-3
DQN Agent: 
 Syntesis Time: 1133.0ms
 Expanded Transactions: 64
 Expanded states: 65

Runing DQN Agent in instance: AT 10-4
DQN Agent: 
 Syntesis Time: 14485.0ms
 Expanded Transactions: 325
 Expanded states: 326

Runing DQN Agent in instance: AT 10-5
DQN Agent: 
 Syntesis Time: 373624.0ms
 Expanded Transactions: 1956
 Expanded states: 1957

Runing DQN Agent in instance: AT 10-6
DQN Agent: 
 Syntesis Time: 13887348.0ms
 Expanded Transactions: 13699
 Expanded states: 13700
 
Runing DQN Agent in instance: AT 10-7


Runing DQN Agent in instance: AT 11-2
DQN Agent: 
 Syntesis Time: 479.0ms
 Expanded Transactions: 15
 Expanded states: 16

Runing DQN Agent in instance: AT 11-3
DQN Agent: 
 Syntesis Time: 1321.0ms
 Expanded Transactions: 64
 Expanded states: 65

Runing DQN Agent in instance: AT 11-4
DQN Agent: 
 Syntesis Time: 15748.0ms
 Expanded Transactions: 325
 Expanded states: 326

Runing DQN Agent in instance: AT 11-5
DQN Agent: 
 Syntesis Time: 424045.0ms
 Expanded Transactions: 1956
 Expanded states: 1957

Runing DQN Agent in instance: AT 11-6
...
"""


