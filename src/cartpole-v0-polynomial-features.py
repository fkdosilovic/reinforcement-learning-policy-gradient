import gym
import numpy as np

from models import MultinomialLogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import binom

from rlsim import simulate, evaluate
from rloptim import vpg_policy_update
import rleval

degree = 3
transform = PolynomialFeatures(degree=degree).fit_transform


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

    def choose_action(self, state):
        """Selects action with the highest probability."""
        return self.policy.predict(state)


def main():
    env = gym.make("CartPole-v0")

    observation_space_d = np.product(env.observation_space.shape)
    action_space_d = env.action_space.n
    n_features = int(binom(observation_space_d + degree, degree))

    agent = Agent(n_features=n_features, n_actions=action_space_d)

    n_epochs = 25
    mb_size = 16
    episode_dbg = 10

    for epoch in range(n_epochs):
        batch = [simulate(env, agent, transform) for _ in range(mb_size)]
        vpg_policy_update(batch, agent, learning_rate=0.035)

        average_return = sum(
            [sum(rewards) for _, _, _, rewards in batch]
        ) / len(batch)

        print(f"Average return for {epoch + 1}th epoch is {average_return}.")

        if epoch % episode_dbg == 0:
            simulate(env, agent, transform, True)

    simulate(env, agent, transform, True)

    if evaluate(
        env,
        agent,
        rleval.CARTPOLE_V0_EPISODES,
        rleval.check_cartpole_v0,
        transform,
    ):
        print("You've successfully solved the environment.")
    else:
        print("You did not solve the environment.")

    env.close()


if __name__ == "__main__":
    main()
