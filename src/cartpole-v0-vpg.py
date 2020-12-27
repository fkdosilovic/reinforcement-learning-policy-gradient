import gym
import numpy as np

from models import MultinomialLogisticRegression
from sklearn.preprocessing import PolynomialFeatures


transform = PolynomialFeatures(degree=1).fit_transform


class Agent:
    def __init__(self, n_features, n_actions):
        self.n_actions = n_actions
        self.actions = list(range(n_actions))
        self.policy = MultinomialLogisticRegression(
            n_features=n_features,
            n_actions=n_actions,
        )

    def sample_action(self, state):
        probs = self.policy.forward(state).flatten()
        return (np.random.choice(a=self.actions, p=probs), probs)


def simulate(env, agent, render=False):
    traj_states = []
    traj_probs = []
    traj_actions = []
    traj_rewards = []

    prev_state = transform(env.reset().reshape(1, -1))
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

        prev_state = transform(next_state.reshape(1, -1))

    return (traj_states, traj_probs, traj_actions, traj_rewards)


def compute_reward_to_go(rewards):
    rewards = np.array(rewards)[::-1]
    return np.cumsum(rewards)[::-1]


def policy_update(episode_batch, agent, learning_rate: float = 0.0025):
    dw = np.zeros_like(agent.policy.w)

    for i, (states, probs, actions, rewards) in enumerate(episode_batch):
        assert len(probs) == len(actions) == len(rewards)

        rewards_to_go = compute_reward_to_go(rewards)
        for i, (state, prob, action) in enumerate(zip(states, probs, actions)):
            e = np.zeros(agent.n_actions)
            e[action] = 1

            dw += agent.policy.grad_w(e, prob, state) * rewards_to_go[i]

    dw /= len(episode_batch)

    agent.policy.w += learning_rate * dw


def main():
    env = gym.make("CartPole-v0")
    agent = Agent(n_features=5, n_actions=2)

    n_episodes = 70
    mb_size = 32
    episode_dbg = 20

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
