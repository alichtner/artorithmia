import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from collections import Counter
from colorutils import Color
from colorthief import ColorThief
from PIL import Image
from skimage import color, io
import seaborn as sns
import cv2
import os

import art_plot as viz


class Art(object):
    """
    Create an Art object and build a set of attributes and methods
    about the Art itself.

    INPUT: None
    OUTPUT: Art (object)
    """
    def __init__(self):
        self.filename = None

    def __str__(self):
        str = """\n\033[1m{}\033[0m is an instance of an Art object \
        \n\033[1maspect ratio\033[0m = {}\n"""
        return str.format(self.short_name, self.aspect_ratio)

    def load_image(self, filename):
        """
        Load image file and build attributes

        INPUT: filename (ex: jpg, png)
        OUTPUT: None
        """

        self.filename = filename
        self.image = misc.imread(filename)
        if len(self.image.shape) != 3:
            raise ValueError('Image is not the right dimensions')
        self.short_name = self.filename.split('/')[-1].split('.')[0]
        self.aspect_ratio = 1. * self.image.shape[1]/self.image.shape[0]
        self.extract_blur()
        self.extract_symmetry()
        self.get_rgb()
        self.get_hsv()

    def build_color_features(self):
        pass

    def build_composition_features(self):
        pass

    def build_style_features(self):
        pass

    def build_content_features(self):
        pass

    def build_meta_features(self):
        pass

    def parse_meta(self, json_obj):
        """
        Takes a json object and parses the values into attributes
        """
        self.styles = json_obj['styles']
        self.can_hang_without_frame = json_obj['can_hung_without_frame']
        self.surface = json_obj['surface']  # ex. 'canvas'
        self.published_at = json_obj['published_at']
        self.updatedAt = json_obj['updatedAt']
        self.createdAt = json_obj['createdAt']
        self.retail_price = json_obj['retail_price']
        self.style_other = json_obj['style_other']  # 'floral'
        self.objectId = json_obj['objectId']
        self.title = json_obj['title']
        self.is_framed = json_obj['is_framed']
        self.width = json_obj['width']  # inches
        self.primary_index = json_obj['primary_index']
        self.height = json_obj['height']    # inches
        self.public_id = json_obj['metadata']['public_id']
        self.sold = json_obj['sold']
        self.depth = json_obj['depth']
        self.no_of_likes = json_obj['no_of_likes']
        self.artist = json_obj['metadata']['public_id'].split('/')[0]
        self.medium = json_obj['medium']
        self.recommended_matted_border = json_obj['recommended_matted_border']
        #,metadata,profile,description,collection,is_primary,collection_index,published

    def show_image(self):
        sns.set_style("whitegrid", {'axes.grid': False})
        plt.imshow(self.image)
        plt.show()

    def show_thumbnail(self):
        Image.open(self.image).thumbnail((200, 200))
        return None

    def get_colors(self, n_colors=5):
        pass

    def get_rgb(self):
        self.red_bins = self.create_hist_vector(self.image, 0, 255, (0.0, 255))
        self.grn_bins = self.create_hist_vector(self.image, 1, 255, (0.0, 255))
        self.blue_bins = self.create_hist_vector(self.image, 2, 255, (0.0, 255))

    def get_hsv(self, plot=False):
        self.hsv_image = color.rgb2hsv(self.image)
        self.hue_bins = self.create_hist_vector(self.hsv_image, 0, 75, (0.0, 1))
        self.sat_bins = self.create_hist_vector(self.hsv_image, 1, 75, (0.0, 1))
        self.val_bins = self.create_hist_vector(self.hsv_image, 2, 75, (0.0, 1))
        self.primary_hue = np.argmax(self.hue_bins)
        self.primary_sat = np.argmax(self.sat_bins)
        self.primary_val = np.argmax(self.val_bins)
        if plot is True:
            viz.plot_hsv(self.hsv_image)

    def create_hist_vector(self, image, channel, bins, rng):
        counts, _ = np.histogram(image[:, :, channel].flatten(), bins, rng)
        return 1. * counts/counts.max()  # Scale the data

    def extract_blur(self, plot=False):
        # do on grayscale
        # check what the mean would give instead of variance
        self.bluriness = cv2.Laplacian(color.rgb2gray(self.image),
                                       cv2.CV_64F).var()
        if plot is True:
            self.lap = cv2.Laplacian(color.rgb2gray(self.image), cv2.CV_64F)
            plt.imshow(self.lap)
            plt.title('Laplacian of {}'.format(self.short_name))
            plt.show()

    def extract_symmetry(self):
        if len(self.image.shape) == 3:
            height, width, channels = self.image.shape
        else:
            height, width = self.image.shape
        if width % 2 != 0:
            width -= 1
            pixels = height * width
            left = self.image[:, :width/2]
            right = self.image[:, width/2:-1]
        else:
            pixels = height * width
            left = self.image[:, :width/2]
            right = self.image[:, width/2:]
        left_gray = color.rgb2gray(left)
        right_gray = color.rgb2gray(right)
        self.symmetry = np.abs(left_gray -
                               np.fliplr(right_gray)).sum()/(pixels/1.*2)

    def extract_movement(self):
        pass

    def plot_rgb(self):
        # count rgb values
        r = Counter(self.image[:, :, 0].flatten())
        g = Counter(self.image[:, :, 1].flatten())
        b = Counter(self.image[:, :, 2].flatten())
        self.r_hist = {k: r[k] for k in range(256)}
        self.g_hist = {k: g[k] for k in range(256)}
        self.b_hist = {k: b[k] for k in range(256)}

        # plot the figure
        x = range(256)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        ax1.fill_between(x, 0, self.r_hist.values(), facecolor='red')
        ax1.set_xlabel('Red')
        ax2.fill_between(x, 0, self.g_hist.values(), facecolor='green')
        ax2.set_xlabel('Green')
        ax3.fill_between(x, 0, self.b_hist.values(), facecolor='blue')
        ax3.set_xlabel('Blue')
        plt.show()

    def get_palette(self, color_count=6, plot=False, save=False):
        color_thief = ColorThief(self.filename)
        # get the dominant color
        dominant_color = color_thief.get_color(quality=1)
        print dominant_color
        palette = color_thief.get_palette(color_count=color_count)
        if plot is True:
            fname = 'plots/' + self.short_name + '_palette.png'
            my_cols = []
            for col in palette:
                c = Color(col)
                my_cols.append(c.hex)
            sns.palplot(sns.color_palette(my_cols))
            plt.title(str(self.short_name) + '_Palette')
            if save is True:
                plt.savefig(fname)
            else:
                plt.show()


if __name__ == '__main__':
    images = ['images/cats.png', 'images/abstract.jpg', 'images/realism.jpg']
    for i in images:
        art = Art()
        art.load_image(i)
        art.extract_blur()
        print art.short_name, 'blurriness is: ', art.bluriness
        art.get_palette(plot=True)
        print art.aspect_ratio
