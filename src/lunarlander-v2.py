import gym
import numpy as np

import torch
from torch import optim

import rleval
import rlsim
import rloptim
import rlstats
import rlutils
from agent import DiscreteMLPAgent


def main():
    env = gym.make("LunarLander-v2")

    observation_space_d = np.product(env.observation_space.shape)
    action_space_d = np.product(env.action_space.n)
    layers = [observation_space_d, 128, 128, action_space_d]

    agent = DiscreteMLPAgent(layers)

    optimizer = optim.Adam(agent.policy.parameters(), lr=0.01)

    n_epochs = 200
    mb_size = 32
    episode_dbg = 20

    for epoch in range(n_epochs):
        batch = [rlsim.simulate(env, agent) for _ in range(mb_size)]
        rloptim.policy_update(batch, agent, optimizer, gamma=0.99)

        average_return = rlstats.calc_average_return(
            rlutils.extract_rewards(batch)
        )

        average_entropy = rlstats.calc_average_entropy(
            rlutils.extract_action_probs(batch)
        )

        if np.isnan(average_entropy):
            break

        print(
            f"Average return for {epoch + 1}th epoch is {average_return:.2f} \
with average entropy of {average_entropy:.2f}."
        )

        if epoch % episode_dbg == 0:
            rlsim.simulate(env, agent, True)

    rlsim.simulate(env, agent, True)

    model_name = f"{n_epochs}_{mb_size}_{layers}"
    torch.save(agent.policy, f"lunarlander2_{model_name}.pth")

    # for param in agent.policy.parameters():
    #     print(param.data)

    if rleval.evaluate(
        env,
        agent,
        rleval.LUNARLANDER_V2_EPISODES,
        rleval.check_lunarlander_v2,
    ):
        print("You've successfully solved the environment.")
    else:
        print("You did not solve the environment.")

    env.close()


if __name__ == "__main__":
    main()
