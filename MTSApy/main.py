from src.experiments import TestTrainedInAllInstances, TrainSmallInstance

if __name__ == "__main__":
    experiment_name = "CRL"
    instance = "BW"

    onyx_path = "MTSApy/results/models/BW/CRL-BW-1.onnx"

    # TrainSmallInstance().train(instance, 2, 2, experiment_name)
    # TestTrainedInAllInstances().select_with_java(instance, experiment_name, 1000)
    TestTrainedInAllInstances().test_with_java(instance, experiment_name, 15000, onyx_path)
    # concrete_instance = "MTSApy/fsp/BW/BW-2-1.fsp"
    # TestTrainedInAllInstances().get_controller(concrete_instance, experiment_name, 15000, )
