import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import signal
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

    def load_image(self, filename, meta):
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
        self.build_color_features()
        self.build_composition_features()
        self.build_style_features()
        self.build_content_features()
        self.build_meta_features(meta)
        self.build_labels()

    def build_color_features(self):
        """
        Extract color related features from each piece of art.
        """
        self.get_rgb()
        self.get_hsv()

        self.primary_hue = np.argmax(self.hue_bins)
        self.avg_hue = self.hue_bins.mean()
        self.hue_var = self.hue_bins.var()

        self.primary_sat = np.argmax(self.sat_bins)
        self.avg_sat = self.sat_bins.mean()
        self.sat_var = self.sat_bins.var()

        self.primary_val = np.argmax(self.val_bins)
        self.avg_val = self.val_bins.mean()
        self.val_var = self.val_bins.var()

        self.colorfulness = None
        self.no_colors = None
        self.size_color_blocks_avg = None
        self.size_color_blocks_var = None

    def build_composition_features(self):
        self.aspect_ratio = 1. * self.image.shape[1]/self.image.shape[0]
        self.extract_symmetry()
        self.image_moment = None

    def build_style_features(self):
        self.extract_blur()

    def build_content_features(self):
        pass

    def build_meta_features(self, meta_dict):
        """
        Takes a json object and parses the values into attributes
        """
        self.artist = self.short_name.split('_')[0]
        self.styles = meta_dict['styles']
        self.can_hang_without_frame = meta_dict['can_hung_without_frame']
        self.surface = meta_dict['surface']  # ex. 'canvas'
        self.published_at = meta_dict['published_at']
        self.updatedAt = meta_dict['updatedAt']
        self.createdAt = meta_dict['createdAt']
        self.style_other = meta_dict['style_other']  # 'floral'
        self.objectId = meta_dict['objectId']
        self.title = meta_dict['title']
        self.is_framed = meta_dict['is_framed']
        self.primary_index = meta_dict['primary_index']
        self.width = meta_dict['width']  # inches
        self.height = meta_dict['height']    # inches
        self.public_id = meta_dict['metadata']['public_id']
        self.depth = meta_dict['depth']
        self.no_of_likes = meta_dict['no_of_likes']
        self.medium = meta_dict['medium']
        self.recommended_matted_border = meta_dict['recommended_matted_border']
        # get retail price
        self.retail_price = meta_dict['retail_price']
        if meta_dict['retail_price'] <= 0:
            self.retail_price = 0.0
        # get sold status
        self.sold = False
        if 'sold' in meta_dict.keys():
            self.sold = meta_dict['sold']
        # set size characteristics
        if self.width > 0:
            self.area = 1. * self.width * self.height
        else:
            self.width = 0.0
            self.height = 0.0
            self.area = 0.0

        #,metadata,colors, is_primary (artists key piece of art)
        # look at cloudinary colors

    def build_labels(self):
        self.labels = []
        col_labs = [('RED-ORANGE', (2, 6)), ('ORANGE',
                    (6, 10)), ('YELLOW-ORANGE', (10, 14)), ('YELLOW', (14, 18)),
                    ('YELLOW-GREEN', (18, 21)), ('GREEN', (21, 24)),
                    ('BLUE-GREEN', (24, 30)), ('BLUE', (30, 34)),
                    ('BLUE-VIOLET', (34, 38)), ('VIOLET', (38, 42)),
                    ('RED-VIOLET', (42, 46))]
        for col, rng in col_labs:
            if self.primary_hue > rng[0] and self.primary_hue <= rng[1]:
                self.labels.append(col)
        if len(self.labels) == 0:
            self.labels.append('RED')

    def show_thumbnail(self):
        pass

    def get_colors(self, n_colors=5):
        pass

    def get_rgb(self):
        self.red_bins = self.create_hist_vector(self.image, 0, 255, (0.0, 255))
        self.grn_bins = self.create_hist_vector(self.image, 1, 255, (0.0, 255))
        self.blue_bins = self.create_hist_vector(self.image, 2, 255, (0.0, 255))

    def get_hsv(self, plot=False):
        self.hsv_image = color.rgb2hsv(self.image)
        self.hue_bins = self.create_hist_vector(self.hsv_image, 0, 48, (0.0, 1))
        self.sat_bins = self.create_hist_vector(self.hsv_image, 1, 48, (0.0, 1))
        self.val_bins = self.create_hist_vector(self.hsv_image, 2, 48, (0.0, 1))
        self.hue_peaks = signal.find_peaks_cwt(self.hue_bins,np.arange(1,10), min_length=5)
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
        # currently this is only for horizontal symmetry
        if len(self.image.shape) == 3:
            height, width, _ = self.image.shape
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


    def show_image(self):
        print self.__str__()
        sns.set_style("whitegrid", {'axes.grid': False})
        plt.imshow(self.image)
        plt.show()

    def plot_bins(self, attr):
        sns.barplot(range(len(getattr(self, attr))), getattr(self, attr))
        plt.xlabel(attr)
        plt.show()

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

    def __str__(self):
        str = """
              \n\033[1m--- Art Attributes--- \033[0m\n
              \n\033[1maspect ratio\033[0m = {}
              \n\033[1mLabels\033[0m : {}
              \n\033[1mPrimary Hue\033[0m : {}
              """
        return str.format(self.aspect_ratio, self.labels, self.primary_hue)


if __name__ == '__main__':
    images = ['images/cats.png', 'images/abstract.jpg', 'images/realism.jpg']
    for i in images:
        art = Art()
        art.load_image(i)
        art.extract_blur()
        print art.short_name, 'blurriness is: ', art.bluriness
        art.get_palette(plot=True)
        print art.aspect_ratio
