import random

from src.composition import CompositionGraph, CompositionAnalyzer
from src.environment import Environment
from src.agents.dqn import DQN, NeuralNetwork, TorchModel

FSP_PATH = "./fsp"
BENCHMARK_PROBLEMS = ["AT", "BW", "CM", "DP", "TA", "TL"]
random.seed(10)

def test_custom_instance(path, idxs=None):
    #d = CompositionGraph("AT", 3, 3, FSP_PATH)
    d = CompositionGraph("Custom", 1, 1, path)
    d.start_composition()
    context = CompositionAnalyzer(d)
    env = Environment(context, False)
    state = env.reset()
    finish = False
    rewards = []
    i = 0

    while not finish:
        if idxs is None:
            n_actions = len(state)
            idx = random.randint(0, n_actions-1)
        else:
            idx = idxs[i]

        state, reward, finish, info = env.step(idx)
        rewards.append(reward)
        i = i + 1

    env.close()

def test_custom_instance_with_agent(path, instance, n, k):
    d = CompositionGraph(instance, n, k, path)
    d.start_composition()
    context = CompositionAnalyzer(d)
    env = Environment(context, False)
    state = env.reset()
    nfeatures = env.get_nfeatures()
    args = {"nn_size": [16],
            "learning_rate": 0.01,
            "momentum": 0.1,
            "nesterov": 0.1,
            "weight_decay": 0.1,
            "first_epsilon": 0.5,
            "buffer_size": 500,
            "n_step": 25,
            "last_epsilon": 0.001,
            "epsilon_decay_steps": 5,
            "exp_replay": 500,
            "target_q": 0.1,
            "reset_target_freq": 50,
            "batch_size": 300
            }

    neural_network = NeuralNetwork(nfeatures, args["nn_size"]).to("cpu")
    nn_model = TorchModel(nfeatures, network=neural_network, args=args)
    agent = DQN(env, nn_model, args, save_file=None, verbose=False)
    agent.train(seconds=None, max_steps=None, max_eps=None, save_freq=200000, last_obs=None, early_stopping=False, save_at_end=False, results_path=None, top=10)
    finish = False
    rewards = []
    i = 0

    while not finish:
        idx = agent.get_action(state, 0)
        state, reward, finish, info = env.step(idx)
        rewards.append(reward)
        i = i + 1

    env.close()

if __name__ == "__main__":
    #path = "F:\\UBA\\Tesis\\MTSApy\\fsp\\Blocking\\ControllableFSPs\\GR1Test10.lts" # For Windows
    path = "F:\\UBA\\Tesis\\MTSApy\\fsp"  # For Windows
    #path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp/Blocking/ControllableFSPs/GR1Test1.lts" # For Linux
    #test_custom_instance(path, [0, 1, 1, 0, 0, 0, 0])
    test_custom_instance_with_agent(path, "AT", 3, 3)



