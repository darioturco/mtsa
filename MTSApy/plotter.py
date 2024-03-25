import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import pandas as pd
import numpy as np
import seaborn as sn

OUTPUT_FOLDER = "./results/plots/"
#BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]
BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL"]
#SCALE = 1
SCALE = 1000

def graph_individual_training_process(sliding_window=5, save_path=None, use_steps=False, problems=None, graph_loss=False):
    #random_data = pd.read_csv("./results/csv/random budget=5000 repetitions=100.csv")
    random_data = pd.read_csv("./results/csv/random.csv")
    ra_data = pd.read_csv("./results/csv/Ready Abstraction.csv")
    limit = 1000000 // 25

    if problems is None:
        problems = BENCHMARK_PROBLEMS

    for instance in problems:
        rl_path = f"./results/training/{instance}-2-2-partial.csv"
        rewards, episodes, steps, rewards_win, losses = get_info_of_instance(rl_path, sliding_window, limit)

        #random_mean = -int(random_data[(random_data["Instance"] == instance) & (random_data["N"] == 2) & (random_data["K"] == 2)]["Transitions (mean)"])
        #random_max = -int(random_data[(random_data["Instance"] == instance) & (random_data["N"] == 2) & (random_data["K"] == 2)]["Transitions (min)"])
        #ra_value = -int(ra_data[(ra_data["Instance"] == instance) & (ra_data["N"] == 2) & (ra_data["K"] == 2)]["Transitions"])

        if use_steps:
            x = steps
            x_label = 'Steps'
        else:
            x = episodes
            x_label = 'Episodes'

        plt.clf()
        plt.plot(x, rewards_win, label="RL")
        #plt.axhline(y=int(random_mean), color='g', linestyle='--', label="Random Mean")
        #plt.axhline(y=int(random_max), color='g', linestyle='-', label="Random Max")
        #plt.axhline(y=int(ra_value), color='red', linestyle='-', label="RA")
        plt.xlabel(x_label)
        plt.ylabel('Reward')
        plt.title(instance)
        plt.legend()
        if save_path is not None:
            plt.savefig(f"{save_path}/{instance}.png")
        plt.show()

        if graph_loss:
            plt.clf()
            plt.plot(x, losses, label="Loss")
            plt.xlabel(x_label)
            plt.ylabel('Loss')
            plt.title(instance)
            plt.show()

def get_info_of_instance(rl_path, sliding_window, limit):
    data_instance = pd.read_csv(rl_path, names=['Step', "Expansion", "Reward", "Loss"])
    rewards = list(data_instance["Reward"])
    episodes = range(len(rewards) - sliding_window)
    steps = [-sum(rewards[:eps]) for eps in range(len(episodes))]
    rewards_win = [np.mean(rewards[eps: eps + sliding_window]) for eps in episodes]
    losses = [np.mean(data_instance["Loss"][eps: eps + sliding_window]) for eps in range(len(episodes))]
    return rewards[:limit], episodes[:limit], steps[:limit], rewards_win[:limit], losses[:limit]

def data_csv(instance, method):
    if method == "RA":
        return pd.read_csv(f"./results/csv/Ready Abstraction.csv")
    else:
        return pd.read_csv(f"./results/csv/{method}-{instance}.csv")

def check_method_in_instance(instance, method):
    data = data_csv(instance, method)

    instance_matrix = []
    for n in range(15, 0, -1):
        row = []
        for k in range(1, 16):
            if method == "RA":
                value = int(data[(data["N"] == n) & (data["K"] == k) & (data["Instance"] == instance)]["Transitions"])
            else:
                value = int(data[(data["N"] == n) & (data["K"] == k)]["Transitions"])

            #print(f"N = {n} - K = {k} -> {method}: {value}")
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

def get_random_info(random_data, instance, budget, instances_solved):
    random_instance = random_data[random_data["Instance"] == instance]
    aux = random_instance[(random_instance["Transitions (mean)"] < budget) & (random_instance["Transitions (mean)"] > 0)]
    solved = aux.count()["Failed"]
    if instances_solved:
        return solved
    else:
        res = (225 - solved) * budget + int(aux["Transitions (mean)"].sum())
        return int(res / SCALE)


def get_ra_info(ra_data, instance, budget, instances_solved):
    ra_instance = ra_data[ra_data["Instance"] == instance]
    aux = ra_instance[ra_instance["Transitions"] < budget]
    solved = aux.count()["Instance"]
    if instances_solved:
        return solved
    else:
        res = (225 - solved) * budget + aux["Transitions"].sum()
        return int(res / SCALE)

def get_rl_info(method, instance, budget, instances_solved):
    # CM no esta corrido aun
    if instance == "CM":
        return 18

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
    ra_data = pd.read_csv("./results/csv/Ready Abstraction.csv")

    for instance in instances:
        for method in data.keys():
            if method == "Random":
                data["Random"][instance] = get_random_info(random_data, instance, budget, instances_solved)
            elif method == "RA":
                data["RA"][instance] = get_ra_info(ra_data, instance, budget, instances_solved)
            else:
                data[method][instance] = get_rl_info(method, instance, budget, instances_solved)

    return data

def comparative_bar_plot(data=None, instances_solved=True):
    instances = BENCHMARK_PROBLEMS[::-1]
    budgets = [1000, 2500, 5000, 10000]
    if data is None:
        data = []
        for b in budgets:
            data_schema = {"Random": {}, "LRL": {}, "CRL": {}, "RRL": {}, "RA": {}}
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
    #graph_individual_training_process(sliding_window=500, save_path='./results/plots', use_steps=True, problems=BENCHMARK_PROBLEMS)
    #graph_individual_training_process(sliding_window=1000, save_path=None, use_steps=True, problems=["BW"])

    #graph_training_process(sliding_window=100, repetitions=5, save_path='./results/plots', use_steps=True)

    # Budget of 10000
    comparative_bar_plot(data=None, instances_solved=False)
    #comparative_bar_plot(data={"Random": {"AT": 59, "BW": 44, "DP": 62, "TA": 60, "TL": 134, "CM": 18},
    #                           "2-2": {"AT": 85, "BW": 53, "DP": 101, "TA": 60, "TL": 225, "CM": 18},
    #                           #"ERL": {"AT": 87, "BW": 57, "DP": 150, "TA": 60, "TL": 225, "CM": 0},
    #                           #"IERL": {"AT": 90, "BW": 56, "DP": 150, "TA": 60, "TL": 225, "CM": 0},
    #                           "LRL": {"AT": 90, "BW": 104, "DP": 150, "TA": 60, "TL": 225, "CM": 24},
    #                           "GRL": {"AT": 80, "BW": 54, "DP": 108, "TA": 60, "TL": 225, "CM": 24},
    #                           "BFS": {"AT": 53, "BW": 50, "DP": 61, "TA": 60, "TL": 201, "CM": 17},
    #                           "RA Mejora": {"AT": 68, "BW": 136, "DP": 150, "TA": 60, "TL": 225, "CM": 22}})



    #method_name = "CRL"
    #check_method_in_instance("BW", method_name)
    #check_method_in_instance("DP", method_name)
    #check_method_in_instance("AT", method_name)
    #check_method_in_instance("TA", method_name)

    #method_name = "RA"
    #check_method_in_instance("BW", method_name)
    #check_method_in_instance("DP", method_name)
    #check_method_in_instance("AT", method_name)
    #check_method_in_instance("TA", method_name)