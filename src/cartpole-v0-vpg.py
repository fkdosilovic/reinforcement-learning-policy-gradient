import gym
import numpy as np

from models import MultinomialLogisticRegression


class Agent:
    def __init__(self, n_features, n_actions):
        self.n_actions = n_actions
        self.actions = list(range(n_actions))
        self.policy = MultinomialLogisticRegression(
            n_features=n_features,
            n_actions=n_actions,
        )

    def sample_action(self, state):
        probs = self.policy.forward(state)
        return (np.random.choice(a=self.actions, p=probs), probs)


def simulate(env, agent, render=False):
    traj_states = []
    traj_probs = []
    traj_actions = []
    traj_rewards = []

    prev_state = env.reset()
    while True:
        if render:
            env.render()

        action, probs = agent.sample_action(state=prev_state)
        next_state, reward, is_terminal, _ = env.step(action)

        traj_states.append(prev_state)
        traj_probs.append(probs)
        traj_actions.append(action)
        traj_rewards.append(reward)

        if is_terminal:
            break

        prev_state = next_state

    return (traj_states, traj_probs, traj_actions, traj_rewards)


def compute_reward_to_go(rewards):
    rewards = np.array(rewards)[::-1]
    return np.cumsum(rewards)[::-1]


def policy_update(episode_batch, agent, learning_rate: float = 0.0025):
    dw = np.zeros_like(agent.policy.w)
    db = np.zeros_like(agent.policy.b)

    for i, (states, probs, actions, rewards) in enumerate(episode_batch):
        assert len(probs) == len(actions) == len(rewards)

        rewards_to_go = compute_reward_to_go(rewards)
        for i, (state, prob, action) in enumerate(zip(states, probs, actions)):
            e = np.zeros(agent.n_actions)
            e[action] = 1

            dw += agent.policy.grad_w(e, prob, state) * rewards_to_go[i]
            db += agent.policy.grad_b(e, prob) * rewards_to_go[i]

    m = len(episode_batch)
    dw /= m
    db /= m

    agent.policy.w += learning_rate * dw
    agent.policy.b += learning_rate * db


def main():
    env = gym.make("CartPole-v0")
    agent = Agent(n_features=4, n_actions=2)

    n_episodes = 100
    mb_size = 16
    episode_dbg = 10

    for episode in range(n_episodes):
        batch = [simulate(env, agent) for _ in range(mb_size)]
        policy_update(batch, agent)

        average_return = sum(
            [sum(rewards) for _, _, _, rewards in batch]
        ) / len(batch)

        print(f"Average return: {average_return}")

        if episode % episode_dbg == 0:
            simulate(env, agent, True)

    simulate(env, agent, True)
    env.close()


if __name__ == "__main__":
    main()
