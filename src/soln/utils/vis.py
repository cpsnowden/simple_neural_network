import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.widgets import Button


class Presenter(object):
    def __init__(self, number_of_layers, image_and_label_retriever):
        self.image_and_label_retreiver = image_and_label_retriever
        self.fig, self.ax = plt.subplots(1, number_of_layers)

        plt.subplots_adjust(bottom=0.2)
        self.next_button = Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Next')
        self.next_button.on_clicked(self.onclick)
        self.onclick(None)

    @staticmethod
    def transform_image(image):
        return np.reshape(image, (-1, 28))

    def plot_image(self, image, prediction, label, activations):
        image = Presenter.transform_image(image)

        self.ax[0].matshow(image, cmap=matplotlib.cm.binary)
        plt.title("Label {}, \nPredicting {}".format(label, prediction))

        for i, activation in enumerate(activations[1:]):
            self.ax[i+1].matshow(activation, cmap=matplotlib.cm.binary)

        for ax in self.ax:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        plt.draw()

    def onclick(self, event):
        image, prediction, label, activations = self.image_and_label_retreiver()
        self.plot_image(image, prediction, label, activations)

    def go(self):
        plt.show()
