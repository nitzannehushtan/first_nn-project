from typing import List

import idx2numpy
import cv2 as cv

from layer import Layer
from network import Network
from node import Node
from utils import sigmoid

DATA_DIR = "C:\\Nitzan\\nn-project\\data"


def display_image(image):
    cv.imshow("Image", image)
    cv.waitKey(0)


def get_dataset_as_array(data_set):
    return idx2numpy.convert_from_file(data_set)


def get_pixels_as_nodes(raw_image) -> List[Node]:
    nodes = []
    ind = 0
    for row in raw_image:
        for pixel in row:
            nodes.append(Node(sigmoid(pixel), ind, 0))
            ind += 1
    return nodes


if __name__ == '__main__':
    training_samples = get_dataset_as_array(DATA_DIR + '/train-images-idx3-ubyte')
    training_labels = get_dataset_as_array(DATA_DIR + '/train-labels-idx1-ubyte')
    test_samples = get_dataset_as_array(DATA_DIR + '/t10k-images-idx3-ubyte')
    test_labels = get_dataset_as_array(DATA_DIR + '/t10k-labels-idx1-ubyte')

    image = test_samples[0]

    layer_0 = Layer(get_pixels_as_nodes(image), 0)
    layer_1 = Layer(layer_ind=1, prev_layer=layer_0, hidden=10)
    layer_2 = Layer(layer_ind=2, prev_layer=layer_1, hidden=10)

    final_layer = Layer(layer_ind=3, prev_layer=layer_2, hidden=10)

    predictor = Network([layer_0, layer_1, layer_2, final_layer])

    # while predictor.current_layer < len(predictor.layers):
    #     predictor.layers[predictor.current_layer].print_layer()
    #     predictor.advance_layer()

