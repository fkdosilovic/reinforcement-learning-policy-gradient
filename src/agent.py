import torch
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from scipy.special import binom


def create_multilayer_perceptron(layers_dim):
    layers = []
    for i, layer_d in enumerate(layers_dim[1:]):
        layers.append(torch.nn.Linear(layers_dim[i], layer_d))
        layers.append(torch.nn.ReLU(inplace=True))

    layers = layers[:-1]  # Remove the ReLU after the output layer.
    layers.append(torch.nn.Softmax(dim=1))
    layers.append(torch.nn.Flatten(start_dim=0))

    return torch.nn.Sequential(*layers)


class DiscreteLinearAgent:
    def __init__(self, n_features, n_actions, degree=1):
        self.params = (n_features, n_actions, degree)
        n_features = int(binom(n_features + degree, degree))
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(n_features, n_actions, bias=False),
            torch.nn.Softmax(dim=1),
            torch.nn.Flatten(start_dim=0),
        )

        self.transform = PolynomialFeatures(degree=degree).fit_transform

    def sample_action(self, state):
        """Samples an action."""
        state = self.transform(state)
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)

        probs = torch.FloatTensor(self.policy.forward(state))
        return (torch.multinomial(probs, 1), probs)

    def choose_action(self, state):
        """Selects action with the highest probability."""
        state = self.transform(state)
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)

        return torch.argmax(self.policy.forward(state)), None


class DiscreteMLPAgent:
    """Implements an agent with neural network policy."""

    def __init__(self, layers):
        self.params = (layers,)
        self.policy = create_multilayer_perceptron(layers)

    def sample_action(self, state):
        """Samples an action."""
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)

        probs = self.policy(state)
        return (torch.multinomial(probs, 1), probs)

    def choose_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)

        return torch.argmax(self.policy(state)), None
