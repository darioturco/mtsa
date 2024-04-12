from src.experiments import TestTrainedInAllInstances, TrainSmallInstance
from src.fixed_experiment import *
from src.instance_maker import *
import cProfile

if __name__ == "__main__":
    experiment_name = "CRL"
    instance = "TA"

    TrainSmallInstance().train(instance, 2, 2, experiment_name)
    TestTrainedInAllInstances().select_with_java(instance, experiment_name, 1000, 1000)
    #TestTrainedInAllInstances().test_with_java(instance, experiment_name, 15000, None)

    #TestTrainedInAllInstances().test_with_java(instance, experiment_name, 10000, "./results/models/TA/CRL/TA-2-2-3370-partial.onnx")








### Notas:

    # Estoy buscando la mejor combinacion de features para TA, ya esta compilado un .jar que prueba unos efatures que pueden andar bien. Tengo que probar tambien que pasa si aumento el tamano de la red
    # Manan voy a dejar corriendo 2-2 (la seleccion)