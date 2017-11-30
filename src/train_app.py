from soln.utils.data_utils import load_data
from soln.network.oo_network import NN

if __name__ == "__main__":

    training_data, validation_data, test_data = load_data()
    net = NN(784, 30, 10)
    net.train(training_data, 5, 10, 3.0, test_data=test_data)