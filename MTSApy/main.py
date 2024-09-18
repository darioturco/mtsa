from src.experiments import TestTrainedInAllInstances, TrainSmallInstance

if __name__ == "__main__":
    experiment_name = "ROLES_FEW"
    instance = "DP"

    TrainSmallInstance().train(instance, 2, 2, experiment_name)
    TestTrainedInAllInstances().select_with_java(instance, experiment_name, 1000)
    TestTrainedInAllInstances().test_with_java(instance, experiment_name, 15000, None)

    # onyx_path = "MTSApy/results/models/BW/CRL-BW-1.onnx"
    # TestTrainedInAllInstances().test_time_with_java(instance, experiment_name, 1800000, onyx_path)

    onyx_path = "./results/models/DP/ROLES_FEW/DP-2-2-14700-partial.onnx"
    # concrete_instance = "fsp/DP/DP-15-15.fsp"
    # TestTrainedInAllInstances().get_controller(concrete_instance, experiment_name, 15000, onyx_path)

