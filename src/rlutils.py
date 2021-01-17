from json import JSONEncoder

import numpy as np

from torch import Tensor


class CustomEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            return format(obj, ".2f")
        if isinstance(obj, (np.ndarray, Tensor)):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def extract_action_probs(batch):
    return [episode_probs for _, _, episode_probs, _ in batch]


def extract_rewards(batch):
    return [eps_rewards for _, _, _, eps_rewards in batch]
