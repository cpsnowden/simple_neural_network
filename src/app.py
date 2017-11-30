import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from soln.utils.vis import Presenter
from soln.network import oo_network
from soln.utils.data_utils import load_data


class ImageOnDemand(object):
    def __init__(self, images):
        self.images = images
        self.counter = 0

    def get(self):
        self.counter += 1
        return self.images[self.counter]


def plot_network(net):
    G = nx.Graph()
    layout = {}
    for id, n in enumerate(net.neurons):
        n.global_id = id
        G.add_node(id)
        layout[id] = np.array([n.layer_n, -2000 + n.id_in_layer * 4000 / float(net.layer_sizes[n.layer_n])])
    for id, edge in enumerate(net.edges):
        G.add_edge(edge.input.global_id, edge.output.global_id)
    nx.draw(G, layout, node_size=2, edge_color='grey')
    plt.show()


def helper(image_and_label_provider, network_func, get_neuron_activations_as_numpy_func):
    image, label = image_and_label_provider.get()
    prediction = network_func(image)
    return image, prediction, label, get_neuron_activations_as_numpy_func()


def run_interactive():
    training_data, validation_data, test_data = load_data()
    #
    # net = simple_network.NeuralNetwork(784, 30, 10)
    # net.train(training_data, 5, 10, 3.0, test_data=test_data)

    net = oo_network.NN(784, 30, 10)
    net.train(training_data, 1, 10, 3.0, test_data=test_data)

    image_provider = ImageOnDemand(test_data)
    Presenter(net.number_of_layers, lambda: helper(image_provider, net.predict,
                                                   net.get_neuron_activations_as_numpy)).go()


if __name__ == "__main__":
    run_interactive()
