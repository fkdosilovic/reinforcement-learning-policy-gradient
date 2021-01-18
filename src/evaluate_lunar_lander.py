import os
import sys
import yaml

import gym

import numpy as np

import torch

import rlsim
import rleval

from agent import DiscreteMLPAgent

dirname = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(dirname)
experiments_dir = os.path.join(project_dir, "experiments")
models_dir = os.path.join(project_dir, "models")


def main(params_yaml_path):
    with open(params_yaml_path) as fp:
        params = yaml.load(fp, Loader=yaml.FullLoader)

    env = gym.make(params["env"])

    observation_space_d = np.product(env.observation_space.shape)
    action_space_d = np.product(env.action_space.n)
    layers = [observation_space_d, *params["hidden_layers"], action_space_d]

    # Load saved torch model.
    agent = DiscreteMLPAgent(layers)
    agent.policy = torch.load(os.path.join(models_dir, params["model"]))

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
