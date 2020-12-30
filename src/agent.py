import torch
import numpy as np

from sklearn.preprocessing import PolynomialFeatures


def _transform(degree: int = 1):
    return PolynomialFeatures(degree=degree).fit_transform


class LinearAgent:
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        transform_degree: int = 1,
    ):
        self.transform = _transform(degree=transform_degree)
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=n_features,
                out_features=n_actions,
                bias=False,
            ),
            torch.nn.Softmax(dim=1),
            torch.nn.Flatten(start_dim=0),
        )

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
