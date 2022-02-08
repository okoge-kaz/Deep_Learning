import numpy as np


def softmax(a: np.ndarray) -> np.ndarray:
    c = np.max(a)
    exp_a = np.exp(a - c)  # overflow対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
