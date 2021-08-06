import math
from random import random

from serialization import Serializable
import numpy as np

from utils import sigmoid


class Node:

    def __init__(self, data: float, index: int, layer_index: int):
        self.data = data
        self.index = index
        self.layer_index = layer_index

    def set_node(self, layer=None, weights=None):
        self.data = sigmoid(sum([node.data * weights[node.index] for node in layer.nodes]))

