import numpy as np

import pickle


def load_pickled_data(data_path="../raw_data/mnist.pkl"):
    with open(data_path, "rb") as file:
        training_set, validation_set, test_set = pickle.load(file)
    return training_set, validation_set, test_set


def load_data(vectorise_training=True):
    training, validation, testing = load_pickled_data()
    return (transform(training, vectorise=vectorise_training), transform(validation), transform(testing))


def transform(data, vectorise=False):
    if vectorise:
        labels = [to_vector(x) for x in data[1]]
    else:
        labels = data[1]
    return zip([np.reshape(x, (784, 1)) for x in data[0]], labels)


def to_vector(x, size = 10):
    v = np.zeros((size, 1))
    v[x] = 1
    return v
