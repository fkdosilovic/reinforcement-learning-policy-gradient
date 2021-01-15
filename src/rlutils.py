def extract_action_probs(batch):
    return [episode_probs for _, _, episode_probs, _ in batch]


def extract_rewards(batch):
    return [eps_rewards for _, _, _, eps_rewards in batch]
