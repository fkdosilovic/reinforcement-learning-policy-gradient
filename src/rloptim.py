import torch
from torch import distributions


def compute_reward_to_go(rewards, gamma: float = 1.0):
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.as_tensor(rewards, dtype=torch.float32)

    rewards = torch.flip(rewards, dims=(0,))

    cumm_sum = torch.zeros_like(rewards)
    cumm_sum[0] = rewards[0]
    for i, reward in enumerate(rewards[1:]):
        cumm_sum[i + 1] = reward + gamma * cumm_sum[i]

    return torch.flip(cumm_sum, dims=(0,))


def compute_baseline(rewards_batch, gamma=1.0):
    max_len = max(map(len, rewards_batch))

    rewards = torch.FloatTensor(
        [rwds + [0] * (max_len - len(rwds)) for rwds in rewards_batch]
    )

    m = len(rewards_batch)
    baselines = compute_reward_to_go(torch.sum(rewards, axis=0), gamma) / m
    return baselines


def vpg(batch, agent, optimizer, gamma=1.0):
    """Implements a vanilla policy gradient with time-dependend average
    baseline."""

    baseline = compute_baseline([rewards for _, _, _, rewards in batch], gamma)
    baseline -= torch.mean(baseline)  # Var(X - E[X]) = Var(X)
    baseline /= torch.std(baseline)  # Possible division by zero!

    loss = 0
    for (_, eps_actions, eps_probs, eps_rewards) in batch:
        assert len(eps_actions) == len(eps_probs) == len(eps_rewards)

        # Compute rewards-to-go.
        rewards_to_go = compute_reward_to_go(eps_rewards, gamma)
        rewards_to_go -= torch.mean(rewards_to_go)  # Var(X - E[X]) = Var(X)
        rewards_to_go /= torch.std(rewards_to_go)  # Possible division by zero!

        episode_probs_action = zip(eps_probs, eps_actions)
        for i, (action_probs, action) in enumerate(episode_probs_action):
            dist = distributions.Categorical(probs=action_probs)
            advantage = rewards_to_go[i] - baseline[i]
            loss += -dist.log_prob(action) * advantage

    loss /= len(batch)

    # Clear gradients.
    optimizer.zero_grad()

    # Compute gradients.
    loss.backward()

    # Gradient descent.
    optimizer.step()
