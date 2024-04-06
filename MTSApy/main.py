from src.experiments import TestTrainedInAllInstances, TrainSmallInstance
from src.fixed_experiment import *
from src.instance_maker import *
import cProfile

if __name__ == "__main__":
    experiment_name = "CRL"
    instance = "DP"
    TrainSmallInstance().train(instance, 2, 2, experiment_name)

    TestTrainedInAllInstances().select_with_java(instance, experiment_name, 1000, 2500)

    TestTrainedInAllInstances().test_with_java(instance, experiment_name, 15000, None)





### Notas:

# Ver:  ["java", "-Xmx8g", "-XX:MaxDirectMemorySize=512m", "-classpath", "mtsa.jar",

# My java command(for selection):
# java -classpath mtsa.jar MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.DCSForPython -s -i DP -e "2-2" -b 1000
#
#
