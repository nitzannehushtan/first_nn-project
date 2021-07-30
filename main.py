
import idx2numpy
import cv2 as cv

DATA_DIR = "C:\\Nitzan\\nn-project\\data"


def display_image(image):
    cv.imshow("Image", image)
    cv.waitKey(0)


def get_dataset_as_array(data_set):
    return idx2numpy.convert_from_file(data_set)


if __name__ == '__main__':
    training_samples = DATA_DIR + '/train-images-idx3-ubyte'
    arr = get_dataset_as_array(training_samples)
    display_image(arr[0])
