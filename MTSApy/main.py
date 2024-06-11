from src.experiments import TestTrainedInAllInstances, TrainSmallInstance

if __name__ == "__main__":
    experiment_name = "CRL"
    instance = "TA"

    #TrainSmallInstance().train(instance, 2, 2, experiment_name)
    #TestTrainedInAllInstances().select_with_java(instance, experiment_name, 1000)
    #TestTrainedInAllInstances().test_with_java(instance, experiment_name, 15000, None)

    TestTrainedInAllInstances().test_with_java(instance, experiment_name, 15000, "./results/models/TA/CRL-TA.onnx")


