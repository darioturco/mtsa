from src.experiments import TestTrainedInAllInstances, TrainSmallInstance

if __name__ == "__main__":
    experiment_name = "CRL"
    #instance = "CM"

    #TrainSmallInstance().train(instance, 2, 2, experiment_name)
    #TestTrainedInAllInstances().select_with_java(instance, experiment_name, 1000)
    #TestTrainedInAllInstances().test_with_java(instance, experiment_name, 15000, None)



    #TestTrainedInAllInstances().test_time_with_java(instance, experiment_name, 1800000, f"./results/final/modelsfinal/{instance}/{experiment_name}-{instance}-1.onnx")

    TestTrainedInAllInstances().test_time_with_java("CM", experiment_name, 1800000,f"./results/final/modelsfinal/CM/{experiment_name}-CM-1.onnx")
    #TestTrainedInAllInstances().test_time_with_java("AT", experiment_name, 1800000,f"./results/final/modelsfinal/AT/{experiment_name}-AT-1.onnx")
    #TestTrainedInAllInstances().test_time_with_java("DP", experiment_name, 1800000,f"./results/final/modelsfinal/DP/{experiment_name}-DP-1.onnx")
    #TestTrainedInAllInstances().test_time_with_java("BW", experiment_name, 1800000,f"./results/final/modelsfinal/BW/{experiment_name}-BW-1.onnx")

    experiment_name = "RL"
    #TestTrainedInAllInstances().test_time_with_java("AT", experiment_name, 1800000,f"./results/final/modelsfinal/AT/{experiment_name}-AT-1.onnx")
    #TestTrainedInAllInstances().test_time_with_java("DP", experiment_name, 1800000,f"./results/final/modelsfinal/DP/{experiment_name}-DP-1.onnx")
    #TestTrainedInAllInstances().test_time_with_java("BW", experiment_name, 1800000,f"./results/final/modelsfinal/BW/{experiment_name}-BW-1.onnx")
    #TestTrainedInAllInstances().test_time_with_java("CM", experiment_name, 1800000,f"./results/final/modelsfinal/CM/{experiment_name}-CM-1.onnx")
    #TestTrainedInAllInstances().test_time_with_java("TA", experiment_name, 1800000,f"./results/final/modelsfinal/TA/{experiment_name}-TA-1.onnx")

