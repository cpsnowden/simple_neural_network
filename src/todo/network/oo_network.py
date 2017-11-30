from soln.utils.data_utils import load_data

import itertools
import pprint
import random
from operator import add
import numpy as np
import soln.network.simple_network


class VoltageHolder(object):
    def __init__(self, initial_voltage):
        self._voltage = initial_voltage

    @property
    def voltage(self):
        return self._voltage

    @voltage.setter
    def voltage(self, new_voltage):
        self._voltage = new_voltage

class Neuron(object):
    def __init__(self, initial_bias, layer_n, id_in_layer, is_hidden=True):
        # TODO duplication voltage field - try inheritance
        self._voltage = 0
        self._bias = initial_bias
        self._activation = 0
        self._input_edge = []
        self._output_edge = []
        self._layer_n = layer_n
        self._id_in_layer = id_in_layer
        self._is_hidden = is_hidden

    def add_input(self, edge):
        self._input_edge.append(edge)

    def add_output(self, edge):
        self._output_edge.append(edge)

    def conduct_forward(self):
        # z = w^T a + b
        # TODO allow a neuron to conduct the voltage from the input edges, activate and then pass that voltage onto its output edges
        pass

    @property
    def voltage(self):
        return self._voltage

    @voltage.setter
    def voltage(self, new_voltage):
        self._voltage = new_voltage

    @property
    def input_edge(self):
        return self._input_edge

    @property
    def output_edge(self):
        return self._output_edge

    @property
    def activation(self):
        return self._activation

    @property
    def layer_n(self):
        return self._layer_n

    @property
    def id_in_layer(self):
        return self._id_in_layer

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, new_bias):
        self._bias = new_bias

    @staticmethod
    def activation_func(voltage):
        # TODO define activation function
        pass

    @staticmethod
    def activation_func_prime(voltage):
        # TODO define derivative of the activation function
        pass

    def __repr__(self):
        return "<Neuron (" + str(self.layer_n) + "," + str(self.id_in_layer) + ") " \
               + str(self.voltage) + "V " \
               + str(self.bias) + " bias>"


class Probe(VoltageHolder):
    def __init__(self, id):
        super(Probe, self).__init__(0.0)
        self._id = id

    @property
    def id(self):
        return self._id

    def conduct(self):
        return self.voltage

    def __repr__(self):
        return "<Probe {} {}V>".format(self.id, self.voltage)


class Edge(VoltageHolder):
    def __init__(self, initial_weight, input, output):
        super(Edge, self).__init__(0.0)
        self._weight = initial_weight
        self._input = input
        self._output = output

    def conduct(self):
        return self.weight * self.voltage

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, new_weight):
        self._weight = new_weight

    @property
    def voltage(self):
        return self._voltage

    @voltage.setter
    def voltage(self, new_voltage):
        self._voltage = new_voltage

    def __repr__(self):
        return "<Edge {}OH {}V>".format(self.weight, self.voltage)


class Network(object):
    def __init__(self, *layer_sizes):

        self.number_of_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

        self.neuron_layers = []
        self.neurons = []
        self.edges = []
        self.input_probes = []
        self.output_probes = []

        for layer_n, layer_size in enumerate(layer_sizes):
            layer = []
            self.neuron_layers.append(layer)
            for id_in_layer in range(0, layer_size):
                if layer_n == 0:
                    bias = 0
                    neuron = Neuron(bias, layer_n, id_in_layer, False)
                else:
                    neuron = Neuron(random.gauss(0, 1), layer_n, id_in_layer, True)

                layer.append(neuron)
                self.neurons.append(neuron)

        #TODO connect up the neurons with edges!

        for i, neuron in enumerate(self.neuron_layers[0]):
            input_probe = Probe(i)
            self.input_probes.append(input_probe)
            neuron.add_input(input_probe)
        for i, neuron in enumerate(self.neuron_layers[-1]):
            output_probe = Probe(i)
            self.output_probes.append(output_probe)
            neuron.add_output(output_probe)

    def get_neuron_activations_as_numpy(self):
        return [np.array([[n.activation] for n in layer]) for layer in self.neuron_layers]

    def predict(self, input):
        probes = self.propagate_forward(input)
        prediction = max(probes, key=lambda probe: probe.voltage).id
        return prediction

    def propagate_forward(self, inputs):

        for probe, input_voltage in zip(self.input_probes, inputs):
            probe.voltage = input_voltage[0]

        for layer in self.neuron_layers:
            for neuron in layer:
                neuron.conduct_forward()
        return self.output_probes

    def train(self, training_data, how_many_times_to_train, stochastic_batch_size, step_size, test_data=None):

        for edge in self.edges:
            edge.delta_weight = []
        for neuron in self.neurons:
            neuron.delta_bias = []

        N = len(training_data)
        for training_time in range(how_many_times_to_train):
            random.shuffle(training_data)
            batches = [training_data[k:k + stochastic_batch_size] for k in range(0, N, stochastic_batch_size)]
            self.train_from_batches(batches, step_size, test_data)
            print("Finished {}/{} training sets".format(training_time + 1, how_many_times_to_train))

    def train_from_batches(self, batches, delta, test_data, fast=True):
        if fast:
            self.train_from_batches_fast(batches, delta, test_data)
        else:
            self.train_from_batches_slow(batches, delta)

    def train_from_batches_fast(self, batches, delta, test_data):

        biases = self.get_biases_as_numpy_array()
        weights = self.get_weights_as_numpy_array()
        for batch in batches:
            weights, biases = soln.network.simple_network.NeuralNetwork.train_from_batch(batch, delta, weights, biases)

        if test_data is not None:
            fast_network = soln.network.simple_network.NeuralNetwork(*self.layer_sizes)
            fast_network.biases = biases
            fast_network.weights = weights
            print("Hits: {} out of {}".format(fast_network.evaluate_hits(test_data), len(test_data)))

        for layer, new_weights_for_layer in zip(self.neuron_layers[1:], weights):
            for neuron, new_weights_for_edges_on_neuron in zip(layer, new_weights_for_layer):
                for edge, new_weight in zip(neuron.input_edge, new_weights_for_edges_on_neuron):
                    edge.weight = new_weight

        for layer, new_bias_for_layer in zip(self.neuron_layers[1:], biases):
            for i, new_bias in enumerate(new_bias_for_layer):
                neuron = layer[i]
                neuron.bias = new_bias[0]

    def train_from_batches_slow(self, batches, delta):
        number_of_batches = len(batches)
        for i, batch in enumerate(batches):
            for input_v, output_i in batch:
                self.propagate_forward(input_v)
                self.backwards_propagate(output_i)

            for edge in self.edges:
                edge.weight -= delta * sum(edge.delta_weight) / float(len(edge.delta_weight))
                edge.delta_bias = []
            for neuron in self.neurons:
                neuron.bias -= delta * sum(neuron.delta_bias) / float(len(neuron.delta_bias))
                neuron.delta_bias = []
            print("Running batch {}/{}".format(i, number_of_batches))

    def backwards_propagate(self, expected_outputs):
        #TODO HARD - propagate backwards the delta

        for output_probe in self.output_probes:
            output_probe.error = self.cost_derivative(output_probe.voltage, expected_outputs[output_probe.id][0])

        for layer_n_r, layer in enumerate(reversed(self.neuron_layers)):
            for neuron in layer:
                Network.take_error(neuron)
                if layer_n_r < self.number_of_layers - 1:
                    Network.propagate_error_back(neuron)

    def get_biases_as_numpy_array(self):
        return [np.array([neuron.bias for neuron in layer]).reshape(-1, 1) for layer in self.neuron_layers[1:]]

    def get_weights_as_numpy_array(self):
        return [np.array([[edge.weight for edge in neuron.input_edge]
                          for neuron in layer])
                for layer in self.neuron_layers[1:]]

    def evaluate_hits(self, test_data):
        """
        test_data is a list of tuples of the form (input - a numpy (x,1) array, label - int)
        """
        #TODO count number of times that the prediction matches the label
        inputs_and_expected_outputs = [(max(self.propagate_forward(input), key=lambda x: x.voltage).id, output)
                                       for input, output in test_data]
        return len([(output, expected) for output, expected in inputs_and_expected_outputs if output == expected])

    @staticmethod
    def cost_derivative(output, expected_output):
        return output - expected_output

    @staticmethod
    def take_error(neuron):
        neuron.error = reduce(add, map(lambda x: x.error, neuron.output_edge)) * Neuron.activation_func_prime(neuron.voltage)
        neuron.delta_bias.append(neuron.error)

    @staticmethod
    def propagate_error_back(neuron):
        for input_edge in neuron.input_edge:
            input_edge.error = neuron.error
            input_edge.delta_weight.append(neuron.activation * neuron.error)

    def __str__(self):
        return pprint.pformat(self.neuron_layers)


if __name__ == "__main__":
    training_data, validation_data, test_data = load_data()
    net = Network(784, 30, 10)
    net.train(training_data, 5, 10, 3.0, test_data=test_data)
