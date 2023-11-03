import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn

OUTPUT_FOLDER = "./results/plots/"
BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]
#BENCHMARK_PROBLEMS = ["AT", "BW", "TA", "TL", "CM"]
#BENCHMARK_PROBLEMS = ["TA"]

def graph_individual_training_process(sliding_window=5, save_path=None, use_steps=False):
    #random_data = pd.read_csv("./results/csv/random budget=5000 repetitions=100.csv")
    random_data = pd.read_csv("./results/csv/random.csv")
    ra_data = pd.read_csv("./results/csv/Ready Abstraction.csv")
    limit = 1200000
    graph_loss = True

    for instance in BENCHMARK_PROBLEMS:
        rl_path = f"./results/training/{instance}-2-2-partial1.csv"
        rewards, episodes, steps, rewards_win, losses = get_info_of_instance(rl_path, sliding_window, limit)

        random_mean = -int(random_data[(random_data["Instance"] == instance) & (random_data["N"] == 2) & (random_data["K"] == 2)]["Transitions (mean)"])
        random_max = -int(random_data[(random_data["Instance"] == instance) & (random_data["N"] == 2) & (random_data["K"] == 2)]["Transitions (min)"])
        ra_value = -int(ra_data[(ra_data["Instance"] == instance) & (ra_data["N"] == 2) & (ra_data["K"] == 2)]["Transitions"])

        if use_steps:
            x = steps
            x_label = 'Steps'
        else:
            x = episodes
            x_label = 'Episodes'

        plt.clf()
        plt.plot(x, rewards_win, label="RL")
        plt.axhline(y=int(random_mean), color='g', linestyle='--', label="Random Mean")
        plt.axhline(y=int(random_max), color='g', linestyle='-', label="Random Max")
        plt.axhline(y=int(ra_value), color='red', linestyle='-', label="RA")
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
    data_instance = pd.read_csv(rl_path, names=['Step', "Reward", "Loss"])
    rewards = list(data_instance["Reward"])
    episodes = range(len(rewards) - sliding_window)
    steps = [sum(rewards[:eps]) for eps in range(len(episodes))]
    rewards_win = [-np.mean(rewards[eps: eps + sliding_window]) for eps in episodes]
    losses = [np.mean(data_instance["Loss"][eps: eps + sliding_window]) for eps in range(len(episodes))]
    return rewards[:limit], episodes[:limit], steps[:limit], rewards_win[:limit], losses[:limit]

def graph_training_process(sliding_window=5, repetitions=5, save_path=None, use_steps=False):
    random_data = pd.read_csv("./results/csv/random.csv")
    ra_data = pd.read_csv("./results/csv/Ready Abstraction.csv")
    limit = 10000

    for instance in BENCHMARK_PROBLEMS:
        plt.clf()
        rewards_all, episodes_all, steps_all, rewards_win_all = [], [], [], []
        for r in range(repetitions):
            rl_path = f"./results/training/{instance}-2-2-partial{r+1}.csv"
            rewards, episodes, steps, rewards_win, _ = get_info_of_instance(rl_path, sliding_window, limit)
            rewards_all.append(rewards)
            episodes_all.append(episodes)
            steps_all.append(steps)
            rewards_win_all.append(rewards_win)


            plt.plot(steps if use_steps else episodes, rewards_win, label=None, alpha=0.2, color='#1f77b4')

        random_mean = -int(
            random_data[(random_data["Instance"] == instance) & (random_data["N"] == 2) & (random_data["K"] == 2)][
                "Transitions (mean)"])
        random_max = -int(
            random_data[(random_data["Instance"] == instance) & (random_data["N"] == 2) & (random_data["K"] == 2)][
                "Transitions (min)"])
        ra_value = -int(
            ra_data[(ra_data["Instance"] == instance) & (ra_data["N"] == 2) & (ra_data["K"] == 2)]["Transitions"])

        # Grafico el promedio de las 5 lineas
        arg_min_size = np.argmin([len(steps) for steps in steps_all])
        min_size = len(steps_all[arg_min_size])

        y = []
        for i in range(min_size):
            y.append(np.mean([rewards_win_all[r][i] for r in range(repetitions)]))



        plt.plot(steps_all[arg_min_size] if use_steps else episodes_all[0], y, label="RL", color='#1f77b4')
        plt.axhline(y=int(random_mean), color='g', linestyle='--', label="Random Mean")
        plt.axhline(y=int(random_max), color='g', linestyle='-', label="Random Max")
        plt.axhline(y=int(ra_value), color='red', linestyle='-', label="RA")
        plt.xlabel('Steps' if use_steps else 'Episodes')
        plt.ylabel('Reward')
        plt.title(instance)
        plt.legend()
        if save_path is not None:
            plt.savefig(f"{save_path}/{instance}_all.png")
        plt.show()

def compare_random_and_RL():
    #random_data = pd.read_csv("./results/csv/random budget=5000 repetitions=100.csv")
    random_data = pd.read_csv("./results/csv/random.csv")
    for instance in BENCHMARK_PROBLEMS:
        data_instance = pd.read_csv(f"./results/csv/{instance}.csv")
        random_data_instance = random_data[random_data["Instance"] == instance]


        #instance_matrix = [[int(data_instance[data_instance["N"] == n and data_instance["K"] == k]["Transitions"][0]) for k in range(2, 16)] for n in range(2, 16)]
        instance_matrix = []
        for n in range(2, 16):
            row = []
            for k in range(2, 16):
                rl_value = int(data_instance[data_instance["N"] == n][data_instance["K"] == k]["Transitions"])
                random_value = int(random_data_instance[random_data_instance["N"] == n][random_data_instance["K"] == k]["Transitions (mean)"])
                print(f"N = {n} - K = {k} -> RL: {rl_value} - Random: {random_value}")
                row.append(rl_value - random_value)
            instance_matrix.append(row)

        instance_matrix = np.array(instance_matrix, dtype=int)
        sn.heatmap(instance_matrix, annot=True, fmt=".0f")

        plt.xticks(np.arange(0.5, 14.5, 1), range(2, 16))
        plt.yticks(np.arange(0.5, 14.5, 1), range(2, 16))

        plt.show()

def comparative_bar_plot():
    random_data = pd.read_csv("./results/csv/random.csv")
    ra_data = pd.read_csv("./results/csv/Ready Abstraction.csv")
    for instance in BENCHMARK_PROBLEMS:
        print(instance)

    data = {"Random": {"TA": 1, "DP": 2, "BW": 5},
            "RL": {"TA": 4, "DP": 3, "BW": 6}}

    instances = ["TA", "DP", "BW"]
    heigths = []
    for i in instances:
        heigths.append(np.array([data[algo][i] for algo in data.keys()]))

    for h in range(len(heigths) - 1, -1, -1):
        plt.bar(list(data.keys()), sum([heigths[i] for i in range(h + 1)]), label=instances[len(instances) - 1 - h])

    for x, (k, v) in enumerate(data.items()):
        totals = [0]
        for i in instances:
            totals.append(totals[-1] + v[i])

        for i in range(len(totals) - 1):
            plt.text(x, (totals[i + 1] - totals[i]) / 2 + totals[i] - 0.25, f'{(totals[i + 1] - totals[i])}',
                     ha='center', va='bottom')

        plt.text(x, totals[-1], f'{totals[-1]}', ha='center', va='bottom')

    plt.legend()
    plt.show()



if __name__ == "__main__":
    print("Plotting...")
    #graph_individual_training_process(sliding_window=500, save_path='./results/plots', use_steps=True)
    #graph_individual_training_process(sliding_window=500, save_path=None, use_steps=True)

    graph_training_process(sliding_window=500, repetitions=5, save_path='./results/plots', use_steps=True)



    #compare_random_and_RL()



