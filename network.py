from layer import Layer

import os
from typing import List

import numpy as np
# save np.load
from utils import sigmoid, relu

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


class Network:

    WEIGHT_PATH = "weights.npy"
    BIASES_PATH = "biases.npy"

    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.current_layer = 1
        self.weights = self.init_weights()
        self.biases = self.init_biases()

    def advance_layer(self):
        self.layers[self.current_layer].nodes_data = relu((self.weights[self.current_layer] @
                                                           self.layers[self.current_layer - 1].nodes_data) +
                                                          self.biases[self.current_layer])
        self.current_layer += 1

    def predict(self):
        print("The prediction is: {}".format(np.argmax(self.layers[-1].nodes_data)))

    def init_weights(self) -> np.ndarray:
        if not os.path.exists(self.WEIGHT_PATH):
            with open(self.WEIGHT_PATH, 'wb') as weights_file:
                weights = np.array([np.random.rand(
                    len(self.layers[i].nodes), len(self.layers[i-1].nodes)) for i in range(len(self.layers))],
                    dtype=object)
                np.save(weights_file, weights)
        with open(self.WEIGHT_PATH, 'rb') as weights_file:
            weights = np.load(weights_file)
        return weights

    def init_biases(self) -> np.ndarray:
        if not os.path.exists(self.BIASES_PATH):
            with open(self.BIASES_PATH, 'wb') as biases_file:
                biases = np.array([np.random.rand(len(self.layers[i].nodes)) for i in range(len(self.layers))],
                                  dtype=object)
                np.save(biases_file, biases)
        with open(self.BIASES_PATH, 'rb') as biases_file:
            biases = np.load(biases_file)
        return biases


if __name__ == '__main__':
    pass
