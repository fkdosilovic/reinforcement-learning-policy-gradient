import gym
import numpy as np
from torch import optim

import rlstats
import rleval
import rlsim
import rlutils
from rloptim import policy_update
from agent import DiscreteLinearAgent


def main():
    degree = 1

    env = gym.make("CartPole-v0")

    observation_space_d = np.product(env.observation_space.shape)
    action_space_d = env.action_space.n

    agent = DiscreteLinearAgent(
        n_features=observation_space_d,
        n_actions=action_space_d,
        degree=degree,
    )

    optimizer = optim.RMSprop(agent.policy.parameters(), lr=0.05)

    n_epochs = 20
    mb_size = 8
    episode_dbg = 10

    for epoch in range(n_epochs):
        batch = [rlsim.simulate(env, agent) for _ in range(mb_size)]
        policy_update(batch, agent, optimizer, 0.95)

        average_return = rlstats.calc_average_return(
            rlutils.extract_rewards(batch)
        )

        average_entropy = rlstats.calc_average_entropy(
            rlutils.extract_action_probs(batch)
        )

        print(
            f"Average return for {epoch + 1}th epoch is {average_return} \
with average entropy of {average_entropy}."
        )

        if epoch % episode_dbg == 0:
            rlsim.simulate(env, agent, True)

    rlsim.simulate(env, agent, True)

    # for param in agent.policy.parameters():
    #     print(param.data)

    if rleval.evaluate(
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
