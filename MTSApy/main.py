from src.experiments import TestTrainedInAllInstances, TrainSmallInstance
from src.fixed_experiment import *
from src.instance_maker import *
import cProfile

if __name__ == "__main__":
    experiment_name = "CRL"
    instance = "DP"

    TrainSmallInstance().train(instance, 2, 2, experiment_name)

    TestTrainedInAllInstances().select_with_java(instance, experiment_name, 1000, 4000)

    TestTrainedInAllInstances().test_with_java(instance, experiment_name, 10000, None)





### Notas:
