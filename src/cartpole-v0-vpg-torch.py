import gym
import numpy as np
from scipy.special import binom
from torch import optim

import rleval
from rlsim import simulate, evaluate
from rloptim import policy_update
from agent import LinearAgent


def main():
    degree = 1

    env = gym.make("CartPole-v0")

    observation_space_d = np.product(env.observation_space.shape)
    action_space_d = env.action_space.n
    n_features = int(binom(observation_space_d + degree, degree))

    agent = LinearAgent(
        n_features=n_features,
        n_actions=action_space_d,
        transform_degree=degree,
    )

    optimizer = optim.RMSprop(agent.policy.parameters(), lr=0.025)

    n_epochs = 20
    mb_size = 16
    episode_dbg = 10

    for epoch in range(n_epochs):
        batch = [simulate(env, agent) for _ in range(mb_size)]
        policy_update(batch, agent, optimizer)

        average_return = sum(
            [sum(rewards) for _, _, _, rewards in batch]
        ) / len(batch)

        print(f"Average return for {epoch + 1}th epoch is {average_return}.")

        if epoch % episode_dbg == 0:
            simulate(env, agent, True)

    simulate(env, agent, True)

    # for param in agent.policy.parameters():
    #     print(param.data)

    if evaluate(
        env,
        agent,
        rleval.CARTPOLE_V0_EPISODES,
        rleval.check_cartpole_v0,
    ):
        print("You've successfully solved the environment.")
    else:
        print("You did not solve the environment.")

    env.close()


if __name__ == "__main__":
    main()
