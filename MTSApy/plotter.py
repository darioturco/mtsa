import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn

OUTPUT_FOLDER = "./results/plots/"
#BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]
#BENCHMARK_PROBLEMS = ["AT", "BW", "TA", "TL", "CM"]
BENCHMARK_PROBLEMS = ["DP"]

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
                #print(data[data["N"] == n][data["K"] == k][data["Method"] == method]["Transitions"])
                value = int(data[(data["N"] == n) & (data["K"] == k) & (data["Method"] == method)]["Transitions"])

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

def comparative_bar_plot(data=None):
    instances = ["AT", "BW", "DP", "TA", "TL", "CM"][::-1]
    if data is None:
        random_data = pd.read_csv("./results/csv/random.csv")
        ra_data = pd.read_csv("./results/csv/Ready Abstraction.csv")
        data = {"Random": {}, "RL": {}, "RA": {}}

        for instance in instances:
            random_instance = random_data[random_data["Instance"] == instance]
            data["Random"][instance] = random_instance[random_instance["Failed"] < 99].count()["Failed"]

            ra_instance = ra_data[ra_data["Instance"] == instance]
            data["RA"][instance] = ra_instance[ra_instance["Failed"] == False].count()["Failed"]

            rl_instance = pd.read_csv(f"./results/csv/{instance}.csv")
            data["RL"][instance] = rl_instance[rl_instance["Failed"] == False].count()["Failed"]

    print(data)

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

    plt.legend()
    plt.show()



if __name__ == "__main__":
    print("Plotting...")
    #graph_individual_training_process(sliding_window=500, save_path='./results/plots', use_steps=True, problems=BENCHMARK_PROBLEMS)
    #graph_individual_training_process(sliding_window=1000, save_path=None, use_steps=True, problems=["BW"])

    #graph_training_process(sliding_window=100, repetitions=5, save_path='./results/plots', use_steps=True)

    # Budget of 10000
    comparative_bar_plot(data={"Random": {"AT": 59, "BW": 44, "DP": 62, "TA": 60, "TL": 134, "CM": 18},
                               "2-2": {"AT": 85, "BW": 53, "DP": 101, "TA": 60, "TL": 255, "CM": 18},
                               #"ERL": {"AT": 87, "BW": 57, "DP": 150, "TA": 60, "TL": 255, "CM": 0},
                               #"IERL": {"AT": 90, "BW": 56, "DP": 150, "TA": 60, "TL": 255, "CM": 0},
                               "LRL": {"AT": 90, "BW": 104, "DP": 150, "TA": 60, "TL": 255, "CM": 24},
                               "GRL": {"AT": 80, "BW": 54, "DP": 108, "TA": 60, "TL": 255, "CM": 24},
                               "BFS": {"AT": 53, "BW": 50, "DP": 61, "TA": 60, "TL": 201, "CM": 17},
                               "RA Mejora": {"AT": 68, "BW": 136, "DP": 150, "TA": 60, "TL": 255, "CM": 22}})



    #method_name = "LRL"
    #check_method_in_instance("BW", method_name)
    #check_method_in_instance("DP", method_name)
    #check_method_in_instance("AT", method_name)
    #check_method_in_instance("TA", method_name)

    #method_name = "RA"
    #check_method_in_instance("BW", method_name)
    #check_method_in_instance("DP", method_name)
    #check_method_in_instance("AT", method_name)
    #check_method_in_instance("TA", method_name)