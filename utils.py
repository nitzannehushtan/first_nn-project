import math
import numpy as np


def sigmoid(vector: np.ndarray) -> np.ndarray:
    return np.array(list(map(lambda x: 1 / (1 + (math.e ** x)), vector)))


def relu(vector: np.ndarray) -> np.ndarray:
    return np.array(list(map(lambda x: x if x > 0 else 0, vector)))
