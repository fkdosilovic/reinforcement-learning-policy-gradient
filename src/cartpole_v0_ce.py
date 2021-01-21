import os
import sys

import yaml

import gym

import numpy as np

from torch.nn import utils

import rleval
import rlsim
import rloptim

from agent import DiscreteLinearAgent

dirname = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(dirname)
experiments_dir = os.path.join(project_dir, "experiments")
logs_dir = os.path.join(project_dir, "logs")


def main(params_yaml_path):
    with open(params_yaml_path) as fp:
        params = yaml.load(fp, Loader=yaml.FullLoader)

    env = gym.make(params["env"])

    observation_space_d = np.product(env.observation_space.shape)
    action_space_d = env.action_space.n

    agent = DiscreteLinearAgent(
        n_features=observation_space_d,
        n_actions=action_space_d,
        degree=params["degree"],
    )

    mu = rloptim.cross_entropy(env, agent, params)
    utils.vector_to_parameters(mu, agent.policy.parameters())

    # Provide last simulation before testing.
    rlsim.simulate(env, agent, True)

    # Save the final model.
    # model_name = f"{n_epochs}_{batch_size}_{layers}"
    # torch.save(agent.policy, f"cartploe_{model_name}_final.pth")

    # Evaluate the final learned model.
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
    condition = "experiments" not in sys.argv[1]
    main(
        os.path.join(experiments_dir if condition else project_dir, sys.argv[1])
    )
