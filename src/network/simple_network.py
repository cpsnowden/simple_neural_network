import numpy as np
import random
from operator import add

class Neural_Network(object):
    def __init__(self, *layer_sizes):
        self.number_of_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def forward_propagate(self, input):
        """
        Propagate forward an input returning the networks output

        :param input: (n,1) array of inputs where n is the number of neurons in the first layer
        :type input: ndarray
        """
        for b, w in zip(self.biases, self.weights):
            input = Neural_Network.sigmoid(np.dot(w, input) + b)
        return input

    def train(self, training_data, number_of_time_to_train, stochastic_batch_size, delta, test_data=None):
        """

        :param training_data:
        :type training_data:
        :param training_times:
        :type training_times:
        :param stochastic_batch_size:
        :type stochastic_batch_size:
        :param delta:
        :type delta:
        :param test_data:
        :type test_data:
        """
        N = len(training_data)
        for training_time in range(number_of_time_to_train):
            random.shuffle(training_data)
            batches = [training_data[k:k + stochastic_batch_size] for k in range(0, N, stochastic_batch_size)]
            for batch in batches:
                self.train_from_batch(batch, delta)
            if test_data is not None:
                print("{} training run, hits: {} out of {}".format(training_time, self.evaluate_hits(test_data),
                                                                   len(test_data)))
            else:
                print("Finished {}/{} training sets", training_time + 1, number_of_time_to_train)

    def train_from_batch(self, batch, delta):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        N = len(batch)
        for input, output in batch:
            grad_b_inc, grad_w_inc = self.backward_propagate(input, output)
            grad_b = map(add, grad_b, grad_b_inc)
            grad_w = map(add, grad_w, grad_w_inc)
            # grad_b = [nb+dnb for nb, dnb in zip(grad_b, grad_b_inc)]
            # grad_w = [nw+dnw for nw, dnw in zip(grad_w, grad_w_inc)]

        self.weights = [w - delta / N * grad_wl for w, grad_wl in zip(self.weights, grad_w)]
        self.biases  = [b - delta / N * grad_bl for b, grad_bl in zip(self.biases,  grad_b)]

    def evaluate_hits(self, test_data):
        outputs_and_expected_outputs = [(np.argmax(self.forward_propagate(input)), expected_output) for
                                        input, expected_output in test_data]
        return len([(output, expected) for output, expected in outputs_and_expected_outputs if output == expected])

    @staticmethod
    def sigmoid_dash(x):
        return Neural_Network.sigmoid(x) * (1-Neural_Network.sigmoid(x))

    @staticmethod
    def sigmoid(x):
        """
        If x is a numpy array, applies element wise
        """
        return 1.0 / (1.0 + np.exp(-x))

    def backward_propagate(self, input, output):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        activation = input
        activations = [input]
        zs = []
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = Neural_Network.sigmoid(z)
            activations.append(activation)

        delta = self.cost_deriv(activations[-1], output) * Neural_Network.sigmoid_dash(zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.number_of_layers):
            z = zs[-l]
            sp = Neural_Network.sigmoid(z)
            w_lplus1 = self.weights[-l + 1].transpose()
            delta = np.dot(w_lplus1, delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (grad_b, grad_w)

    def cost_deriv(self, activation, output):
        return activation - output