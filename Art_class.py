import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from collections import Counter
from colorutils import Color
from colorthief import ColorThief
import os


class Art(object):
    """docstring for """
    def __init__(self):
        self.filename = None
        self.image_array = None
        self.r_hist = None
        self.g_hist = None
        self.b_hist = None

    def load_image(self, filename):
        self.filename = filename
        self.image_array = misc.imread(filename)
        return None

    def show_image(self):
        plt.imshow(self.image_array)
        plt.show()
        return None

    def extract_colors(self, n_colors=5):
        pass

    def plot_rgb(self):
        # count rgb values
        r = Counter(self.image_array[:, :, 0].flatten())
        g = Counter(self.image_array[:, :, 1].flatten())
        b = Counter(self.image_array[:, :, 2].flatten())
        self.r_hist = {k: r[k] for k in range(256)}
        self.g_hist = {k: g[k] for k in range(256)}
        self.b_hist = {k: b[k] for k in range(256)}

        # plot the figure
        x = range(256)
        plt.ylim((0, 2500))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        ax1.fill_between(x, 0, self.r_hist.values(), facecolor='red')
        ax1.set_xlabel('Red')
        ax2.fill_between(x, 0, self.g_hist.values(), facecolor='green')
        ax2.set_xlabel('Green')
        ax3.fill_between(x, 0, self.b_hist.values(), facecolor='blue')
        ax3.set_xlabel('Blue')
        plt.show()


if __name__ == '__main__':
    art = Art()
    art.load_image('images/cats.png')
    art.plot_rgb()
