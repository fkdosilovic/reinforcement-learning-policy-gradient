import numpy as np


def calc_episode_return(rewards, gamma: float = 1.0):
    rewards = np.array(rewards)

    gammas = gamma * np.ones_like(rewards)
    gammas[0] = 1.0
    gammas = np.cumprod(gammas)

    return np.sum(rewards * gammas)


def calc_average_return(rewards_batch, gamma: float = 1.0):
    return np.mean(
        [
            calc_episode_return(eps_rewards, gamma)
            for eps_rewards in rewards_batch
        ]
    )


def entropy(probs):
    import torch

    if isinstance(probs, torch.Tensor):
        return -torch.sum(torch.log(probs) * probs).item()

    return -np.sum(np.log(probs) * probs)


def calc_average_entropy(prob_batch):
    return np.mean(
        [np.mean(list(map(entropy, eps_probs))) for eps_probs in prob_batch]
    )
