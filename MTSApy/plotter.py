import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn

OUTPUT_FOLDER = "./results/plots/"
BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]
SCALE = 1000

def graph_individual_training_process(experiment_name, sliding_window=5, save_path=None, use_steps=False, problems=None, graph_loss=False):
    def get_info_of_instance(rl_path, sliding_window):
        data_instance = pd.read_csv(rl_path, names=['Step', "Expansion", "Reward", "Loss"])
        rewards = list(data_instance["Reward"])
        episodes = range(len(rewards) - sliding_window)
        steps = [-sum(rewards[:eps]) for eps in range(len(episodes))]
        rewards_win = [np.mean(rewards[eps: eps + sliding_window]) for eps in episodes]
        losses = [np.mean(data_instance["Loss"][eps: eps + sliding_window]) for eps in range(len(episodes))]
        return rewards, episodes, steps, rewards_win, losses

    if problems is None:
        problems = BENCHMARK_PROBLEMS

    for instance in problems:
        rl_path = f"./results/training/{instance}-{experiment_name}.csv"
        rewards, episodes, steps, rewards_win, losses = get_info_of_instance(rl_path, sliding_window)

        if use_steps:
            x = steps
            x_label = 'Steps'
        else:
            x = episodes
            x_label = 'Episodes'

        plt.clf()
        plt.plot(x, rewards_win, label="RL")
        plt.xlabel(x_label)
        plt.ylabel('Reward')
        plt.title(f"{experiment_name} - {instance}")
        plt.legend()
        if save_path is not None:
            plt.savefig(f"{save_path}/{instance}.png")
        plt.show()

        if graph_loss:
            plt.clf()
            plt.plot(x, losses, label="Loss")
            plt.xlabel(x_label)
            plt.ylabel('Loss')
            plt.title(f"{experiment_name} - {instance}")
            plt.show()

def check_method_in_instance(instance, method):
    if method == "RA":
        data = pd.read_csv(f"./results/csv/Ready Abstraction.csv")
    else:
        data = pd.read_csv(f"./results/csv/{method}-{instance}.csv")

    instance_matrix = []

    for n in range(15, 0, -1):
        row = []
        for k in range(1, 16):
            if method == "RA":
                value = int(data[(data["N"] == n) & (data["K"] == k) & (data["Instance"] == instance)]["Transitions"])
            else:
                value = int(data[(data["N"] == n) & (data["K"] == k)]["Transitions"])

            row.append(value)

        instance_matrix.append(row)

    instance_matrix = np.array(instance_matrix, dtype=int)
    sn.heatmap(instance_matrix, annot=False, fmt=".0f")

    plt.title(f"{instance} - {method}")
    plt.xlabel("K")
    plt.ylabel("N")
    plt.xticks(np.arange(0.5, 15.5, 1), range(1, 16))
    plt.yticks(np.arange(14.5, -0.5, -1), range(1, 16))

    plt.show()
    return instance_matrix

def get_data_info(data, instance, budget, instances_solved):
    ra_instance = data[data["Instance"] == instance]
    aux = ra_instance[(ra_instance["Transitions"] < budget) & (ra_instance["Transitions"] > 0)]
    solved = aux.count()["Instance"]
    if instances_solved:
        return solved
    else:
        res = (225 - solved) * budget + aux["Transitions"].sum()
        return int(res / SCALE)

def get_rl_info(method, instance, budget, instances_solved):
    try:
        rl_data = pd.read_csv(f"./results/csv/{method}-{instance}.csv")
    except FileNotFoundError:
        return 0

    aux = rl_data[rl_data["Transitions"] < budget]
    solved = aux.count()["Instance"]
    if instances_solved:
        return solved
    else:
        res = (225 - solved) * budget + int(aux["Transitions"].sum())
        return int(res / SCALE)

def get_data_for(budget, instances, data, instances_solved):
    random_data = pd.read_csv("./results/csv/random.csv")
    ra_data = pd.read_csv("./results/csv/RA.csv")
    bfs_data = pd.read_csv("./results/csv/BFS.csv")

    for instance in instances:
        for method in data.keys():
            if method == "Random":
                data["Random"][instance] = get_data_info(random_data, instance, budget, instances_solved)
            elif method == "RA":
                data["RA"][instance] = get_data_info(ra_data, instance, budget, instances_solved)
            elif method == "BFS":
                data["BFS"][instance] = get_data_info(bfs_data, instance, budget, instances_solved)
            else:
                data[method][instance] = get_rl_info(method, instance, budget, instances_solved)

    return data

def comparative_bar_plot(data=None, instances_solved=True):
    instances = BENCHMARK_PROBLEMS[::-1]
    budgets = [1000, 2500, 5000, 10000, 15000]
    if data is None:
        data = []
        for b in budgets:
            data_schema = {"Random": {}, "BFS": {}, "RL": {}, "CRL": {}, "RA": {}}
            data.append((b, get_data_for(b, instances, data_schema, instances_solved)))
    else:
        for _ in budgets:
            data.append(data)

    for b, d in data:
        title = f"Budget = {b}"
        comparative_bar_plot_data(d, instances, title)
def comparative_bar_plot_data(data, instances, title):
    plt.clf()
    heights = []
    for i in instances:
        heights.append(np.array([data[algo][i] for algo in data.keys()]))

    for h in range(len(heights) - 1, -1, -1):
        plt.bar(list(data.keys()), sum([heights[i] for i in range(h + 1)]), label=instances[h])

    for x, (k, v) in enumerate(data.items()):
        totals = [0]
        for i in instances:
            totals.append(totals[-1] + v[i])

        for i in range(len(totals) - 1):
            plt.text(x, (totals[i + 1] - totals[i]) / 2 + totals[i] - 10, f'{(totals[i + 1] - totals[i])}',
                     ha='center', va='bottom')

        plt.text(x, totals[-1]+10, f'{totals[-1]}', ha='center', va='bottom')

    plt.title(title)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    print("Plotting...")

    ### Plot of individual training process
    graph_individual_training_process("CRL", sliding_window=500, save_path='./results/plots', use_steps=True, problems=["DP"])
    #graph_individual_training_process("CRL", sliding_window=500, save_path='./results/plots', use_steps=True, problems=["AT"])
    #graph_individual_training_process("CRL", sliding_window=500, save_path='./results/plots', use_steps=True, problems=["BW"])
    #graph_individual_training_process("CRL", sliding_window=500, save_path='./results/plots', use_steps=True, problems=["CM"])
    #graph_individual_training_process("CRL", sliding_window=500, save_path='./results/plots', use_steps=True, problems=["TL"])
    #graph_individual_training_process("CRL", sliding_window=500, save_path='./results/plots', use_steps=True, problems=["TA"])



    ### Plot of comparative add bar plot
    comparative_bar_plot(data=None, instances_solved=True)  # Based on amount of solved instances
    comparative_bar_plot(data=None, instances_solved=False) # Based on amount of expansions



    ### Heatmap for each instance family
    #for method_name in ["CRL", "RA"]:
        #check_method_in_instance("BW", method_name)
        #check_method_in_instance("DP", method_name)
        #check_method_in_instance("AT", method_name)
        #check_method_in_instance("TA", method_name)
        #check_method_in_instance("TL", method_name)
        #check_method_in_instance("CM", method_name)