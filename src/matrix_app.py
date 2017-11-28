if __name__ == "__main__":
    from utils.data_utils import load_data
    from network.simple_network import Neural_Network
    training_data, validation_data, test_data = load_data()
    net = Neural_Network(784, 30, 10)
    net.train(training_data, 5, 10, 3.0, test_data=test_data)