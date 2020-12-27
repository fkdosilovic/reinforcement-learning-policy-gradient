import numpy as np
from scipy.special import softmax

# from sklearn.preprocessing import PolynomialFeatures


class MultinomialLogisticRegression:
    """A multinomial logistic regression (aka. softmax regression) model."""

    def __init__(self, n_features: int, n_actions: int):
        rng = np.random.default_rng()
        self.w = rng.random(size=(n_actions, n_features))
        self.b = rng.random(size=(n_actions))

    def forward(self, x: np.ndarray):
        return softmax(np.matmul(self.w, x) + self.b)

    def predict(self, x: np.ndarray):
        return np.argmax(self.forward(x))

    @staticmethod
    def grad_w(e: np.ndarray, prob: np.ndarray, x: np.ndarray):
        return np.outer(e - prob, x.T)

    @staticmethod
    def grad_b(e: np.ndarray, prob: np.ndarray):
        return e - prob
