import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn

OUTPUT_FOLDER = "./results/plots/"
BENCHMARK_PROBLEMS = ["AT", "BW", "DP", "TA", "TL", "CM"]
#BENCHMARK_PROBLEMS = ["DP", "TA", "CM"]

def graph_training_process(sliding_window=5, save_path=None):
    #random_data = pd.read_csv("./results/csv/random budget=5000 repetitions=100.csv")
    random_data = pd.read_csv("./results/csv/random.csv")
    ra_data = pd.read_csv("./results/csv/Ready Abstraction.csv")

    for instance in BENCHMARK_PROBLEMS:
        data_instance = pd.read_csv(f"./results/training/{instance}-2-2-partial.csv", names=['Step', "Reward"])   # Cambiarle el nombre a {instance}-2-2
        rewards = list(data_instance["Reward"])
        episodes = range(len(rewards) - sliding_window)
        rewards_win = [-np.mean(rewards[eps : eps+sliding_window]) for eps in episodes]

        random_mean = -int(random_data[(random_data["Instance"] == instance) & (random_data["N"] == 2) & (random_data["K"] == 2)]["Transitions (mean)"])
        random_max = -int(random_data[(random_data["Instance"] == instance) & (random_data["N"] == 2) & (random_data["K"] == 2)]["Transitions (min)"])
        ra_value = -int(ra_data[(ra_data["Instance"] == instance) & (ra_data["N"] == 2) & (ra_data["K"] == 2)]["Transitions"])

        plt.plot(episodes, rewards_win, label="RL")
        plt.axhline(y=ra_value, color='r', linestyle='-', label="RA")
        plt.axhline(y=random_mean, color='g', linestyle='--', label="Random Mean")
        plt.axhline(y=random_max, color='g', linestyle='-', label="Random Max")
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title(instance)
        plt.legend()
        if save_path is not None:
            plt.savefig(f"{save_path}/{instance}.png")
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

    pass


if __name__ == "__main__":
    print("Plotting...")
    graph_training_process(sliding_window=32, save_path='./results/plots')
    #compare_random_and_RL()



