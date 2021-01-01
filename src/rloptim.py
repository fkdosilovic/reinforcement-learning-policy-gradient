import torch
from torch import distributions

import numpy as np


def compute_reward_to_go(rewards, gamma: float = 1.0):
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.as_tensor(rewards, dtype=torch.float32)

    rewards = torch.flip(rewards, dims=(0,))

    cumm_sum = torch.zeros_like(rewards)
    cumm_sum[0] = rewards[0]
    for i, reward in enumerate(rewards[1:]):
        cumm_sum[i + 1] = reward + gamma * cumm_sum[i]

    return torch.flip(cumm_sum, dims=(0,))


# def compute_baseline(rewards_batch):
#     max_len = max(map(len, rewards_batch))

#     rewards = torch.FloatTensor(
#         [rwds + [0] * (max_len - len(rwds)) for rwds in rewards_batch]
#     )

#     m = len(rewards_batch)
#     baselines = compute_reward_to_go(torch.sum(rewards, axis=0)) / m
#     return baselines


def policy_update(
    batch,
    agent,
    optimizer,
    gamma: float = 1.0,
):
    # baseline = compute_baseline([rewards for _, _, _, rewards in batch])

    # # Normalize.
    # baseline -= torch.mean(baseline)
    # baseline /= torch.std(baseline)  # Possible division by zero!

    loss = 0
    for (_, episode_actions, episode_probs, episode_rewards) in batch:
        assert (
            len(episode_actions) == len(episode_probs) == len(episode_rewards)
        )

        # Compute rewards_to_go
        rewards_to_go = compute_reward_to_go(episode_rewards, gamma)

        # Since Var(X - E[X]) = Var(X), we can standardize the rewards by
        # writing:
        rewards_to_go -= torch.mean(rewards_to_go)
        rewards_to_go /= torch.std(rewards_to_go)  # Possible division by zero!

        episode_probs_action = zip(episode_probs, episode_actions)
        for i, (action_probs, action) in enumerate(episode_probs_action):
            dist = distributions.Categorical(probs=action_probs)
            advantage = rewards_to_go[i]
            loss += -dist.log_prob(action) * advantage

    # Clear gradients.
    optimizer.zero_grad()

    # Compute gradients.
    loss.backward()

    # Gradient descent.
    optimizer.step()
