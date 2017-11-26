from utils.data_utils import load_data
from utils.vis import Presenter
from network import simple_network
import numpy as np

class ImageOnDemand(object):
    def __init__(self, images):
        self.images = images
        self.counter = 0

    def get(self):
        self.counter += 1
        return self.images[self.counter]


def helper(image_and_label_provider, network_func):
    image, label = image_and_label_provider.get()
    return image, np.argmax(network_func(image)), label


def run_interactive():
    training_data, validation_data, test_data = load_data()
    net = simple_network.Neural_Network(784, 30, 10)
    net.train(training_data, 5, 10, 3.0, test_data=test_data)

    image_provider = ImageOnDemand(test_data)
    Presenter(lambda: helper(image_provider, net.forward_propagate)).go()


if __name__ == "__main__":
    run_interactive()
