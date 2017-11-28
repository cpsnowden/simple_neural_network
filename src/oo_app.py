if __name__ == "__main__":
    from utils.data_utils import load_data
    from network.oo_network import NN
    training_data, validation_data, test_data = load_data()
    net = NN(784, 30, 10)
    net.train(training_data, 5, 10, 3.0, test_data=test_data)