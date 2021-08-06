from random import random
from node import Node

import numpy as np


class Layer:

    def __init__(self, nodes: np.ndarray = None, layer_ind: int = None, prev_layer=None, hidden=0):
        if hidden > 0:
            self.nodes = [Node(random(), ind, layer_ind) for ind in range(hidden)]
        else:
            self.nodes = nodes
        self.nodes_data = np.array([node.data for node in self.nodes])
        self.layer_ind = layer_ind
        self.prev_layer = prev_layer

    def set_layer(self, nodes):
        self.nodes = nodes

    def get_size(self):
        return len(self.nodes)

    def print_layer(self):
        layer = "Layer {}:\n".format(self.layer_ind)
        for node in self.nodes:
            layer += "\t({}): {}\n".format(node.index, node.data)
        print(layer)
