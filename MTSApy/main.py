from src.experiments import TestTrainedInAllInstances, TrainSmallInstance
from src.fixed_experiment import *
from src.instance_maker import *
import cProfile

if __name__ == "__main__":
    experiment_name = "2-2"
    instance = "TA"
    #TrainSmallInstance().train(instance, 2, 2, experiment_name)



    TestTrainedInAllInstances().select_with_java(instance, experiment_name, 1000)

    #TestTrainedInAllInstances().test_with_java(instance, experiment_name, 10000, None)





    #TestTrainedInAllInstances().run("DP", experiment_name, 1000, pth_path="./DP-2-2-300-partial.onnx", save=False, instance_list=[(2,2)])



    #TestTrainedInAllInstances().run("AT", experiment_name, 10000, pth_path="./results/models/AT/RRL\AT-2-2-6290-partial.pth")
    #TestTrainedInAllInstances().run("BW", experiment_name, 10000, pth_path="./results/models/BW/RRL\BW-2-2-710-partial.pth")
    #TestTrainedInAllInstances().run("DP", experiment_name, 10000, pth_path="./results/models/DP/CRL\DP-2-2-4600-partial.pth", instance_list=[(n, k) for n in range(9, 16) for k in range(2, 16)])
    #TestTrainedInAllInstances().run("TA", experiment_name, 10000, pth_path="./results/models/TA/CRL\TA-2-2-1970-partial.pth")
    #TestTrainedInAllInstances().run("TL", experiment_name, 10000, pth_path="./results/models/TL/LRL\TL-2-2-22240-partial.pth")
    #TestTrainedInAllInstances().run("CM", experiment_name, 10000, pth_path="./results/models/CM/1/CM-2-2-270-partial.pth")

    #cProfile.run('from src.experiments import TestTrainedInAllInstances; TestTrainedInAllInstances().pre_select("DP", "CRL", 1000, 3, [(n, k) for n in range(2, 10) for k in range(2, 10)])')



### Notas:

# Ver:  ["java", "-Xmx8g", "-XX:MaxDirectMemorySize=512m", "-classpath", "mtsa.jar",

# My java command(for selection):
# java -classpath mtsa.jar MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.DCSForPython -s -i DP -e "2-2" -b 1000
#
#
