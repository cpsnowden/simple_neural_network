import numpy as np
from operator import add
import random
import itertools
import pprint


class VoltageHolder(object):
    def __init__(self, initial_voltage):
        self._voltage = initial_voltage

    @property
    def voltage(self):
        return self._voltage

    @voltage.setter
    def voltage(self, new_voltage):
        self._voltage = new_voltage


class Neuron(VoltageHolder):
    def __init__(self, initial_bias, layer_n, id_in_layer):
        super(Neuron, self).__init__(0.0)
        self._bias = initial_bias
        self._activation = 0
        self._input_edge = []
        self._output_edge = []
        self._layer_n = layer_n
        self._id_in_layer = id_in_layer

    def add_input(self, edge):
        self._input_edge.append(edge)

    def add_output(self, edge):
        self._output_edge.append(edge)

    def activate(self):
        # z = wT a + b
        self.voltage = reduce(add, map(lambda x: x.conduct(), self._input_edge)) + self.bias
        self._activation = Neuron.sigmoid(self.voltage)
        for output_edge in self._output_edge:
            output_edge.voltage = self._activation

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
    def sigmoid(z):
        return 1.0 / (1.0 - np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        return Neuron.sigmoid(z) * (1 - Neuron.sigmoid(z))

    def __repr__(self):
        return "<Neuron (" + str(self.layer_n) + "," + str(self._id_in_layer) + ") " \
               + str(self.voltage) + "V " \
               + str(self.bias) + " bias>"


class Probe(VoltageHolder):
    def __init__(self, id):
        super(Probe, self).__init__(0.0)
        self.id = id

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




class NN(object):
    def __init__(self, *layer_sizes):

        self.layers = []
        self.number_of_layers = len(layer_sizes)

        for layer_n, layer_size in enumerate(layer_sizes):
            layer = []
            self.layers.append(layer)
            for id_in_layer in range(0, layer_size):
                layer.append(Neuron(random.gauss(0, 1), layer_n, id_in_layer))

        for index in xrange(len(layer_sizes) - 1):
            left_neuron_layer = self.layers[index]
            right_neuron_layer = self.layers[index + 1]
            for left, right in itertools.product(left_neuron_layer, right_neuron_layer):
                edge = Edge(random.gauss(0, 1), left, right)
                left.add_output(edge)
                right.add_input(edge)

        self.input_probes = []
        for i, neuron in enumerate(self.layers[0]):
            input_probe = Probe(i)
            self.input_probes.append(input_probe)
            neuron.add_input(input_probe)

        self.output_probes = []
        for i, neuron in enumerate(self.layers[-1]):
            output_probe = Probe(i)
            self.output_probes.append(output_probe)
            neuron.add_output(output_probe)

    def propagate_forward(self, inputs):

        for probe, input_voltage in zip(self.input_probes, inputs):
            probe.voltage = input_voltage

        for layer in self.layers:
            for neuron in layer:
                neuron.activate()
            print(self)

        return self.output_probes

    def train(self, training_data, number_of_time_to_train, stochastic_batch_size, step_size, test_data=None):

        #transform the mode into a matrix model, then train then copy the weights and biases back

        N = len(training_data)
        for training_time in range(number_of_time_to_train):
            random.shuffle(training_data)
            batches = [training_data[k:k + stochastic_batch_size] for k in range(0, N, stochastic_batch_size)]
            N = len(batches)
            for i, batch in enumerate(batches):
                self.train_from_batch(batch, step_size)
                print("Running batch {}/{}".format(i,N))
            if test_data is not None:
                print("{} training run, hits: {} out of {}".format(training_time, self.evaluate_hits(test_data),
                                                                   len(test_data)))
            else:
                print("Finished {}/{} training sets", training_time + 1, number_of_time_to_train)

    def evaluate_hits(self, test_data):
        inputs_and_expected_outputs  = [(max(self.propagate_forward(input), key=lambda x:x.voltage()).id, output)
                                        for input,output in test_data]
        return len([(output, expected) for output, expected in inputs_and_expected_outputs if output == expected])


    def train_from_batch_using_ooo(self, batch, delta):
        delta_w = [np.zeros((len(l2), len(l1))) for l1, l2 in zip(self.layers[:-1], self.layers[1:])]
        delta_b = [np.zeros((len(layer), 1)) for layer in self.layers[1:]]
        N = len(batch)
        for input_v, output_i in batch:
            delta_b_inc, delta_w_inc = self.backwards_propagate(input_v, output_i)
            delta_w = map(add, delta_w_inc, delta_w)
            delta_b = map(add, delta_b_inc, delta_b)

        # delta_w
        # [[[            ]
        #   [            ]
        #   [            ]
        #               ]]...]
        # delta_b
        # [[      Layer1      ],[      Layer2      ],[     Layer3      ] ...]

        self.update_weights(delta_w, delta / N)
        self.update_biases(delta_b, delta / N)

    def train_from_batch(self, batch, delta, fast=False):
        if fast:
            self.train_from_batch_fast(batch, delta)
        else:
            self.train_from_batch_using_ooo(batch, delta)

    def train_from_batch_fast(self, batch, delta):
        #convert to matrix then do everthing as matrix format might require() tain() to be done a matrix mult
        pass


    def update_weights(self, delta_weights, step_size):
        for layer, delta_weight_for_layer in zip(self.layers[1:], delta_weights):
            for i, delta_weight_on_output in enumerate(delta_weight_for_layer):
                neuron = layer[i]
                for k, delta_weight in enumerate(delta_weight_on_output):
                    edge = neuron.input_edge[k]
                    edge.weight = edge.weight - step_size * delta_weight

    def update_biases(self, delta_biases, step_size):
        for layer,delta_bias_for_layer in zip(self.layers[1:], delta_biases):
            for i, delta_bias in enumerate(delta_bias_for_layer):
                neuron = layer[i]
                neuron.bias = neuron.bias - step_size * delta_bias[0]

    def __str__(self):
        return pprint.pformat(self.layers)

    def backwards_propagate(self, input_v, output_i):

        for probe, input_voltage in zip(self.input_probes, input_v):
            probe.voltage = input_voltage[0]

        delta_w = [None] * (self.number_of_layers -1)
        delta_b = [None] * (self.number_of_layers -1)

        for probe, input_voltage in zip(self.input_probes, input_v):
            probe.voltage = input_voltage[0]

        activations = []
        voltages = []
        for layer in self.layers:
            for neuron in layer:
                neuron.activate()
            voltages.append(np.array(map(lambda x: x.voltage, layer)).reshape(-1,1))
            activations.append(np.array(map(lambda x: x.activation, layer)).reshape(-1,1))

        delta = self.cost_deriv(activations[-1], output_i) * Neuron.sigmoid_prime(voltages[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.number_of_layers):
            v = voltages[-l]
            sp = Neuron.sigmoid(v)
            w_lplus1 = np.array([[edge.weight for edge in neuron._output_edge]
                                    for neuron in self.layers[-l]])
            delta = np.dot(w_lplus1, delta) * sp
            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return delta_b, delta_w

    def cost_deriv(self, output, expected_output):
        return output - expected_output


