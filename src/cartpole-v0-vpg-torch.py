import os
import sys
import json
import yaml

import gym

import numpy as np

from torch import optim

import rlstats
import rleval
import rlsim
import rlutils
import rloptim

from agent import DiscreteLinearAgent

dirname = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(dirname)
experiments_dir = os.path.join(project_dir, "experiments")
logs_dir = os.path.join(project_dir, "logs")


def main(params_yaml_path):
    with open(params_yaml_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    env = gym.make(params["env"])

    observation_space_d = np.product(env.observation_space.shape)
    action_space_d = env.action_space.n

    agent = DiscreteLinearAgent(
        n_features=observation_space_d,
        n_actions=action_space_d,
        degree=params["degree"],
    )

    optimizer = optim.RMSprop(agent.policy.parameters(), lr=params["lr"])

    n_epochs = params["n_epochs"]
    batch_size = params["batch_size"]
    n_dbg = params["n_dbg"]

    log_data = []
    for epoch in range(n_epochs):
        batch = [rlsim.simulate(env, agent) for _ in range(batch_size)]
        rloptim.vpg(batch, agent, optimizer, params["gamma"])

        log_data.append(
            {
                "epoch": epoch + 1,
                "probs": rlutils.extract_action_probs(batch),
                "rewards": rlutils.extract_rewards(batch),
            }
        )

        average_return = rlstats.calc_average_return(
            rlutils.extract_rewards(batch)
        )

        average_entropy = rlstats.calc_average_entropy(
            rlutils.extract_action_probs(batch)
        )

        print(
            f"Average return for {epoch}th epoch is {average_return:.2f} \
with average entropy of {average_entropy:.2f}."
        )

        if epoch % n_dbg == 0:
            rlsim.simulate(env, agent, True)

    rlsim.simulate(env, agent, True)

    # Save logs.
    logs_fn = os.path.join(logs_dir, params["logs"])
    with open(logs_fn, "w") as fp:
        json.dump(log_data, fp, cls=rlutils.CustomEncoder)

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
    main(os.path.join(experiments_dir, sys.argv[1]))
