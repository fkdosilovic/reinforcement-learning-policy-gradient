import torch
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from scipy.special import binom


class DiscreteLinearAgent:
    def __init__(self, n_features, n_actions, degree=1):
        n_features = int(binom(n_features + degree, degree))
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(n_features, n_actions, bias=False),
            torch.nn.Softmax(dim=1),
            torch.nn.Flatten(start_dim=0),
        )

        self.transform = PolynomialFeatures(degree=degree).fit_transform

    def sample_action(self, state: np.ndarray):
        """Samples an action."""
        state = self.transform(state)
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)

        probs = torch.FloatTensor(self.policy.forward(state))
        return (torch.multinomial(probs, 1), probs)

    def choose_action(self, state):
        """Selects action with the highest probability."""
        state = self.transform(state)
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)

        return torch.argmax(self.policy.forward(state)), None
