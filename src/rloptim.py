import numpy as np

rms_accum_grad = None


def compute_reward_to_go(rewards):
    if not isinstance(rewards, np.ndarray):
        rewards = np.array(rewards)

    rewards = rewards[::-1]
    return np.cumsum(rewards)[::-1]


def compute_time_dependent_baseline(rewards_batch):
    max_len = max(map(len, rewards_batch))

    rewards = np.array(
        [rwds + [0] * (max_len - len(rwds)) for rwds in rewards_batch]
    )

    m = len(rewards_batch)
    baselines = compute_reward_to_go(np.sum(rewards, axis=0)) / m
    return baselines


def vpg_policy_update(
    episode_batch,
    agent,
    learning_rate: float = 0.001,
    rho: float = 0.9,
    epsilon: float = 1e-8,
):
    baselines = compute_time_dependent_baseline(
        [rewards for _, _, _, rewards in episode_batch]
    )

    dw = np.zeros_like(agent.policy.w)
    for i, (states, probs, actions, rewards) in enumerate(episode_batch):
        # assert len(probs) == len(actions) == len(rewards)

        rewards_to_go = compute_reward_to_go(rewards)
        for i, (state, prob, action) in enumerate(zip(states, probs, actions)):
            e = np.zeros(agent.n_actions)
            e[action] = 1

            dw += agent.policy.grad_w(e, prob, state) * (
                rewards_to_go[i] - baselines[i]
            )

    dw /= len(episode_batch)

    # RMSProp accumulation step.
    global rms_accum_grad
    if rms_accum_grad is None:
        print(rms_accum_grad)
        rms_accum_grad = np.zeros_like(agent.policy.w)
    rms_accum_grad = rho * rms_accum_grad + (1 - rho) * (dw * dw)

    # Policy update.
    agent.policy.w += (learning_rate / np.sqrt(epsilon + rms_accum_grad)) * dw
