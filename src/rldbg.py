import os
import sys
import json
import yaml

from rlstats import entropy

import numpy as np
from matplotlib import pyplot as plt

plt.style.use("seaborn-whitegrid")

dirname = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(dirname)
experiments_dir = os.path.join(project_dir, "experiments")
logs_dir = os.path.join(project_dir, "logs")


def plot_running_return_stats(ax, epochs, mu, n_avg):
    # Not the most efficient solution, but works just fine for now.
    get_subsequence = lambda i: mu[:i] if i - n_avg < 0 else mu[i - n_avg : i]
    running_mu = [np.mean(get_subsequence(i)) for i in range(len(mu))]
    ax.plot(epochs, running_mu, lw=1.5, color="orange")


def plot_return_stats(params, ax, logs, plot_running_return=True):
    epochs = list(range(1, len(logs) + 1))

    # rewards = [
    #     [np.sum(eps_rewards) for eps_rewards in epoch["rewards"]]
    #     for epoch in logs
    # ]
    rewards = [epoch["rewards"] for epoch in logs]

    mu = np.array([np.mean(row) for row in rewards])
    sigma = np.array([np.std(row) for row in rewards])

    ax.plot(epochs, mu, lw=1.5, color="blue")
    ax.fill_between(
        epochs, mu + sigma, mu - sigma, facecolor="lightblue", alpha=0.7
    )

    if plot_running_return:
        plot_running_return_stats(ax, epochs, mu, params["n_avg"])

    ax.set_xlabel("epochs")
    ax.set_ylabel("return")


def plot_entropy_stats(params, ax, logs):
    epochs = list(range(1, len(logs) + 1))

    # entropy_stats = [
    #     [np.mean(list(map(entropy, eps_probs))) for eps_probs in epoch["probs"]]
    #     for epoch in logs
    # ]
    entropy_stats = [epoch["entropy"] for epoch in logs]

    mu = np.array([np.mean(row) for row in entropy_stats])
    sigma = np.array([np.std(row) for row in entropy_stats])

    ax.plot(epochs, mu, lw=1.5, color="blue")
    ax.fill_between(
        epochs, mu + sigma, mu - sigma, facecolor="lightblue", alpha=0.7
    )

    min_entropy = np.ones(len(epochs) + 1) * params["min_entropy"]
    max_entropy = np.ones(len(epochs) + 1) * params["max_entropy"]

    ax.plot([0] + epochs, min_entropy, color="black")
    ax.plot([0] + epochs, max_entropy, color="black")

    ax.set_xlim(0, len(epochs))

    ax.set_xlabel("epochs")
    ax.set_ylabel("entropy")


def main(params_yaml_path):
    with open(params_yaml_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    filename = os.path.join(logs_dir, params["logs"])

    fig, subplots = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(f"Statistics for {params['env']} environment.")

    axes = subplots.flatten()

    with open(filename, "r") as fp:
        logs = json.load(fp)

        plot_return_stats(params, axes[0], logs)
        plot_entropy_stats(params, axes[1], logs)

    plt.show()


if __name__ == "__main__":
    main(os.path.join(experiments_dir, sys.argv[1]))
