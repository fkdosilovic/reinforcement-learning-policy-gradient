import gym
import random


class RandomAgent:
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def select_action(self, state):
        return random.randrange(0, self.n_actions)


def main():
    env = gym.make("CartPole-v0")
    agent = RandomAgent(n_actions=2)

    n_episodes = 5
    for episode in range(n_episodes):
        prev_state = env.reset()

        while True:
            env.render()

            action = agent.select_action(state=prev_state)
            next_state, reward, is_terminal, _ = env.step(action)

            if is_terminal:
                break

            prev_state = next_state

    env.close()


if __name__ == "__main__":
    main()
