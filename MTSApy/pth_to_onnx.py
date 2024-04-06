import os
from src.agents.dqn import OnnxModel, TorchModel
from src.experiments import TestTrainedInAllInstances

# TODO: hacer funcinar no funca ahorita
def convert_from_folder(instance, experiment_name, input_folder, output_folder):
    all_models = {os.path.join(r, file) for r, d, f in os.walk(input_folder) for file in f}

    models_folder = f"./results/models/{instance}/{experiment_name}/"
    model_path = f"{models_folder}DP-2-2-4460-partial.pth"

    test = TestTrainedInAllInstances()
    path = test.get_fsp_path()
    args = test.default_args()
    env = test.get_environment(instance, experiment_name, 2, 2, path, args)
    nfeatures = env.get_nfeatures()


    print(f"nfeatures = {nfeatures}")

    model = TorchModel.load(nfeatures, model_path, args)

    OnnxModel(model).save("./DP-2-2-4460-partial")

def convert_model(instance, experiment_name, input_path, output_path):
    test = TestTrainedInAllInstances()
    path = test.get_fsp_path()
    args = test.default_args()
    env = test.get_environment(instance, experiment_name, 2, 2, path, args)
    nfeatures = env.get_nfeatures()


    print(f"nfeatures = {nfeatures}")

    model = TorchModel.load(nfeatures, input_path, args)

    OnnxModel(model).save(output_path)


if __name__ == "__main__":
    instance = "DP"
    experiment_name = "CRL"

    #convert_model(instance, experiment_name, f"./results/models_pth/{instance}/{experiment_name}/DP-2-2-4470-partial.pth", f"./{instance}-2-2-4470-partial")
    convert_from_folder("DP", "LRL", "./DP-2-2-4460-partial.onnx")

