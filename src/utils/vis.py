import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.widgets import Button


class Presenter(object):
    def __init__(self, image_and_label_retriever):
        self.image_and_label_retreiver = image_and_label_retriever
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        self.next_button = Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Next')
        self.next_button.on_clicked(self.onclick)
        self.onclick(None)

    @staticmethod
    def transform_image(image):
        return np.reshape(image, (-1, 28))

    def plot_image(self, image, prediction, label):
        image = Presenter.transform_image(image)
        self.ax.matshow(image, cmap=matplotlib.cm.binary)
        plt.title("Output {}, {}".format(label, prediction))
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        plt.draw()

    def onclick(self, event):
        image, prediction, label = self.image_and_label_retreiver()
        self.plot_image(image, prediction, label)

    def go(self):
        plt.show()
