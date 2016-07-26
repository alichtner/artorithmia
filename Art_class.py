import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from collections import Counter
from colorutils import Color
from colorthief import ColorThief
import seaborn as sns
import cv2
import os


class Art(object):
    """
    Create an Art object and build a set of attributes and methods
    about the Art itself
    """
    def __init__(self):
        self.filename = None
        self.short_name = None
        self.image = None
        self.r_hist = None
        self.g_hist = None
        self.b_hist = None
        self.bluriness = None

    def load_image(self, filename):
        self.filename = filename
        self.image = misc.imread(filename)
        self.short_name = self.filename.split('/')[-1].split('.')[0]
        return None

    def show_image(self):
        plt.imshow(self.image)
        plt.show()
        return None

    def extract_colors(self, n_colors=5):
        pass

    def extract_blur(self):
        self.bluriness = cv2.Laplacian(self.image, cv2.CV_64F).var()

    def extract_symmetry(self):
        pass

    def extract_movement(self):
        pass

    def extract_rgb(self):
        # count rgb values
        r = Counter(self.image[:, :, 0].flatten())
        g = Counter(self.image[:, :, 1].flatten())
        b = Counter(self.image[:, :, 2].flatten())
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

    def get_palette(self, color_count=6, plot=False):
        color_thief = ColorThief(self.filename)
        # get the dominant color
        dominant_color = color_thief.get_color(quality=1)
        palette = color_thief.get_palette(color_count=color_count)
        if plot is True:
            fname = 'plots/' + self.short_name + '_palette.png'
            my_cols = []
            for col in palette:
                c = Color(col)
                my_cols.append(c.hex)
            sns.palplot(sns.color_palette(my_cols))
            plt.title(str(self.short_name) + '_Palette')
            plt.savefig(fname)


if __name__ == '__main__':
    images = ['images/cats.png', 'images/abstract.jpg', 'images/realism.jpg']
    for i in images:
        art = Art()
        art.load_image(i)
        art.extract_blur()
        print art.short_name, ' blurriness is: ', art.bluriness
        art.get_palette(plot=True)
