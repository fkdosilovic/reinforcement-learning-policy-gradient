import os
import sys
import json
import yaml

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

dirname = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(dirname)
experiments_dir = os.path.join(project_dir, "experiments")
logs_dir = os.path.join(project_dir, "logs")


def main(params_yaml_path):
    with open(params_yaml_path) as fp:
        params = yaml.load(fp, Loader=yaml.FullLoader)

    env = gym.make(params["env"])

    observation_space_d = np.product(env.observation_space.shape)
    action_space_d = np.product(env.action_space.n)
    layers = [observation_space_d, *params["hidden_layers"], action_space_d]

    agent = DiscreteMLPAgent(layers)
    optimizer = optim.RMSprop(agent.policy.parameters(), lr=params["lr"])

    n_epochs = params["n_epochs"]
    batch_size = params["batch_size"]
    episode_dbg = params["n_dbg"]

    log_data = []
    for epoch in range(n_epochs):
        batch = [rlsim.simulate(env, agent) for _ in range(batch_size)]
        rloptim.vpg(batch, agent, optimizer, params["gamma"])

        # Compute stuff for the log.
        rewards = [
            np.sum(eps_rewards)
            for eps_rewards in rlutils.extract_rewards(batch)
        ]

        entropy = [
            np.mean(list(map(rlstats.entropy, eps_probs)))
            for eps_probs in rlutils.extract_action_probs(batch)
        ]

        log_data.append(
            {"epoch": epoch + 1, "entropy": entropy, "rewards": rewards}
        )

        avg_return = rlstats.calc_average_return(rlutils.extract_rewards(batch))

        avg_entropy = rlstats.calc_average_entropy(
            rlutils.extract_action_probs(batch)
        )

        if np.isnan(avg_entropy) or avg_entropy < params["min_entropy"]:
            print("Your agent's policy achieved desired entropy.")
            break

        print(
            f"Average return for {epoch + 1}th epoch is {avg_return:.2f} \
with average entropy of {avg_entropy:.2f}."
        )

        if epoch % episode_dbg == 0:
            rlsim.simulate(env, agent, True)

        if epoch % 20 == 0 and epoch > 0:
            # Save logs.
            logs_fn = os.path.join(logs_dir, params["logs"])
            with open(logs_fn, "w") as fp:
                json.dump(log_data, fp, cls=rlutils.CustomEncoder)

            # Save the learned model.
            model_name = (
                f"{n_epochs}_{batch_size}_{layers}_{epoch}_{avg_return:.2f}"
            )
            torch.save(agent.policy, f"lunarlander2_{model_name}.pth")

    # Provide last simulation before testing.
    rlsim.simulate(env, agent, True)

    # Save the final model.
    model_name = f"{n_epochs}_{batch_size}_{layers}"
    torch.save(agent.policy, f"lunarlander2_{model_name}_final.pth")

    # Evaluate the final learned model.
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
    main(os.path.join(experiments_dir, sys.argv[1]))
