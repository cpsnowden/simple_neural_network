if __name__ == "__main__":
    from soln.utils.data_utils import load_data
    from soln.network.simple_network import NeuralNetwork
    training_data, validation_data, test_data = load_data()
    net = NeuralNetwork(784, 30, 10)
    net.train(training_data, 5, 10, 3.0, test_data=test_data)