import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import signal
import os
from collections import Counter
from colorutils import Color
from colorthief import ColorThief
from skimage import measure, color, io, filters
import seaborn as sns
from detect_peaks import detect_peaks   # found this online

import art_plot as viz


class Art(object):
    """
    Create an Art object and build a set of attributes and methods
    about the Art itself.

    Input:  None
    Output: Art (object)
    """
    def __init__(self, item_id=None):
        # item_id is included to index where an Art object is in a collection
        """
        Initialize an Art object

        Input:  item_id (int) the index of the artwork in the collection
        Output: None
        """
        self.filename = None
        self.item_id = item_id
        self.artist = None
        self.styles = None
        self.can_hang_without_frame = None
        self.surface = None
        self.published_at = None
        self.updatedAt = None
        self.createdAt = None
        self.style_other = None
        self.objectId = None
        self.title = None
        self.is_framed = None
        self.primary_index = None
        self.width = 0.0
        self.height = 0.0
        self.public_id = None
        self.depth = None
        self.no_of_likes = None
        self.medium = None
        self.recommended_matted_border = None
        self.url = None
        self.retail_price = 0.0
        self.sold = False
        self.area = 0.0
        self.school = None
        self.time_start = None
        self.time_end = None

    def load_image(self, filename, meta=None):
        """
        Load image file and build attributes

        Input:  filename (str) name of the file with extension ['.jpg', '.png']
                meta (bool) whether or not meta data features should be built
        Output: None
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
        if meta is not None:
            self.build_meta_features(meta)
        else:
            print 'No metadata: Clustering done only with image properties.'
        self.build_labels()

    def build_color_features(self):
        """
        Extract color related features from each piece of art.

        Input:  None
        Output: None
        """
        self.get_rgb()
        self.get_hsv()

        self.primary_hue = np.argmax(self.hue_bins)
        self.primary_sat = np.argmax(self.sat_bins)
        self.primary_val = np.argmax(self.val_bins)
        self.colorfulness = None
        self.no_colors = None
        self.size_color_blocks_avg = None
        self.size_color_blocks_var = None

    def build_composition_features(self):
        """
        Builds composition related features

        Input:  None
        Output: None
        """
        self.aspect_ratio = 1. * self.image.shape[1]/self.image.shape[0]
        self.extract_symmetry()
        i = np.zeros((20, 20), dtype=np.double)
        i[13:17, 13:17] = 1
        m = measure.moments(i)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]
        self.image_central_moments = measure.moments_central(i, cr, cc)

    def build_style_features(self):
        """
        Extract style features (ex. texture, bluriness)

        Input:  None
        Output: None
        """
        self.extract_blur()

    def build_content_features(self):
        pass

    def build_meta_features(self, meta):
        """
        Takes a json object and parses the values into attributes

        Input:  meta (dict) meta features
        Output: None
        """
        self.artist = self.short_name.split('_')[0]
        self.styles = meta['styles']
        self.can_hang_without_frame = meta['can_hung_without_frame']
        self.surface = meta['surface']  # ex. 'canvas'
        self.published_at = meta['published_at']
        self.updatedAt = meta['updatedAt']
        self.createdAt = meta['createdAt']
        self.style_other = meta['style_other']  # 'floral'
        self.objectId = meta['objectId']
        self.title = str(meta['title']) + ''
        self.is_framed = meta['is_framed']
        self.primary_index = meta['primary_index']
        self.width = meta['width'] # inches
        self.height = meta['height']  # inches
        self.public_id = meta['metadata']['public_id']
        self.depth = meta['depth']
        self.no_of_likes = meta['no_of_likes']
        self.medium = meta['medium']
        self.medium.append('')
        self.recommended_matted_border = meta['recommended_matted_border']
        self.url = meta['image']
        # get retail price
        self.retail_price = meta['retail_price']
        if meta['retail_price'] <= 0:
            self.retail_price = 0
        # get sold status
        self.sold = False
        if 'sold' in meta.keys():
            self.sold = meta['sold']
        # set size characteristics
        if self.width > 0.:
            self.area = 1. * self.width * self.height
        else:
            self.width = 0.0
            self.height = 0.0
            self.area = 0.0
        self.transform_url(width=400, height=300)

    def transform_url(self, width=400, height=400):
        url = self.url.split('/')
        url[6] = 'w_' + str(width)
        self.url = '/'.join(url)

    def build_labels(self):
        """
        Apply labels to art objects based on attributes.

        Input:  None
        Output: None
        """
        self.labels = {}
        # label the primary color of the image
        col_labs = [('RED-ORANGE', (2, 6)), ('ORANGE',
                    (6, 10)), ('YELLOW-ORANGE', (10, 14)), ('YELLOW', (14, 18)),
                    ('YELLOW-GREEN', (18, 21)), ('GREEN', (21, 24)),
                    ('BLUE-GREEN', (24, 30)), ('BLUE', (30, 34)),
                    ('BLUE-VIOLET', (34, 38)), ('VIOLET', (38, 42)),
                    ('RED-VIOLET', (42, 46))]
        for col, rng in col_labs:
            if self.primary_hue > rng[0] and self.primary_hue <= rng[1]:
                self.labels['color'] = col
        if len(self.labels) == 0:
            self.labels['color'] = 'RED'

        # label the saturation of the picture
        if max(self.sat_peaks) > 10:
            self.labels['vibrance'] = 'SATURATED'
        else:
            self.labels['vibrance'] = 'MUTED'

        # label the contrast (boldness?)
        if len(self.val_peaks) == 2 and (self.val_peaks[1] -
                                         self.val_peaks[0] > 10):
            self.labels['contrast'] = 'HIGH-CONTRAST'
        elif len(self.val_peaks) > 2:
            self.labels['contrast'] = 'HIGH-DEPTH'
        else:
            self.labels['contrast'] = 'LOW-CONTRAST'

        # colorful or not
        # check on the relative height of the peaks and if they are at least
        # a certain distance from eachother
        if len(self.hue_peaks) > 1:
            self.labels['colorlevel'] = 'MULTICOLOR'
        else:
            self.labels['colorlevel'] = 'SINGLE PALETTE'

    def get_rgb(self):
        """
        Bin RGB levels from each image.

        Input:  None
        Output: None
        """
        self.red_bins = self.create_hist_vector(self.image, 0, 255, (0.0, 255))
        self.grn_bins = self.create_hist_vector(self.image, 1, 255, (0.0, 255))
        self.blue_bins = self.create_hist_vector(self.image, 2, 255, (0.0, 255))

    def get_hsv(self, plot=False):
        """
        Extract HSV values for each image. Creates bins for the HSV vectors.
        Also finds the peaks in the HSV histograms

        Input:  None
        Output: None
        """
        self.hsv_image = color.rgb2hsv(self.image)
        self.hue_bins, self.avg_hue, self.hue_var = self.create_hist_vector(self.hsv_image, 0, 48, (0.0, 1))
        self.sat_bins, self.avg_sat, self.sat_var = self.create_hist_vector(self.hsv_image, 1, 32, (0.0, 1))
        self.val_bins, self.avg_val, self.val_var = self.create_hist_vector(self.hsv_image, 2, 32, (0.0, 1))
        # get the peaks
        self.hue_peaks = self.get_peaks(self.hue_bins, 0.5, 5)
        self.val_peaks = self.get_peaks(self.val_bins, 0.4, 5)
        self.sat_peaks = self.get_peaks(self.sat_bins, 0.4, 5)
        if plot is True:
            viz.plot_hsv(self.hsv_image)

    def get_peaks(self, bins, min_height=0.4, min_separation=4):
        """
        Find peaks in a signal.

        Input:  bins (np.array) histogram of values
                min_height (float) the min height for a peak to be counted,
                min_separation (int) min distance peak to peak allowed
        Output: peak indices (list)
        """
        return detect_peaks(bins, mph=min_height, mpd=min_separation, edge='both')

    def create_hist_vector(self, image, channel, bins, rng):
        channel_values = image[:, :, channel].flatten()
        counts, _ = np.histogram(channel_values, bins, rng)
        return (1. * counts/counts.max()), channel_values.mean(), channel_values.var()  # Scale the data

    def extract_blur(self, plot=False):
        """
        Calculate the variance of the 2nd derivative of the image to get blur.

        Input:  plot (bool) whether or not to show the image after Laplacian
        Output: None"""
        # do on grayscale
        # check what the mean would give instead of variance
        self.bluriness = filters.laplace(color.rgb2gray(self.image)).var()
        if plot is True:
            sns.set_style("whitegrid", {'axes.grid': False})
            self.lap = filters.laplace(color.rgb2gray(self.image))
            plt.imshow(self.lap)
            plt.title('Laplacian of {}'.format(self.short_name))
            plt.show()
            plt.imshow(self.lap)
            plt.show()

    def extract_symmetry(self):
        """
        Calculate the symmetry of the image by substracting left from right.

        Input:  None
        Output: None
        """
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
        """
        Method to plot the image and attributes.

        Input:  None
        Output: None
        """
        print self.__str__()
        sns.set_style("whitegrid", {'axes.grid': False})
        plt.imshow(self.image)
        plt.show()

    def plot_bins(self, attr):
        """
        Plot barplot of binned values.

        Input:  attr (str) the attribute to use when plotting
        Output: plot object
        """
        sns.barplot(range(len(getattr(self, attr))), getattr(self, attr))
        plt.xlabel(attr)
        plt.show()

    def plot_rgb(self):
        """
        Plot the RGB histrograms from an image.

        Input:  None
        Output: plot object
        """
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
        """
        Use colorthief to grab the palette from an image.

        Input:  color_count (int) number of colors to extract
                plot (bool) plot the palette or not
                save (bool) save the palette to a file
        Output: plot object or file
        """
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
        """
        Formats output for printing information about a work.

        Input:  None
        Output: output string of attributes
        """
        str = """
              \n\033[1m--- Art Attributes--- \033[0m\n
              \r\033[1mTitle: \033[0m {}
              \r\033[1maspect ratio:\033[0m {}
              \r\033[mBlur Level:\033[0m  {}

              \r\033[1mPrimary Hue\033[0m: {}
              \r\033[1mAverage Hue\033[0m: {}
              \r\033[1mHue Variance\033[0m: {}

              \r\033[1mPrimary Sat\033[0m: {}
              \r\033[1mAverage Sat\033[0m: {}
              \r\033[1mSat Variance\033[0m: {}

              \r\033[1mPrimary Val\033[0m: {}
              \r\033[1mAverage Val\033[0m: {}
              \r\033[1mVal Variance\033[0m: {}

              \r\033[1mRetail Price\033[0m : $ {}.00
              \r\033[1mHue Peaks:\033[0m {}
              \r\033[1mVal Peaks:\033[0m {}
              \r\033[1mSat Peaks:\033[0m {}

              \r\033[1mLabels\033[0m : {}
              """
        return str.format(self.title, self.aspect_ratio, self.bluriness,
                          self.primary_hue, self.avg_hue, self.hue_var,
                          self.primary_sat, self.avg_sat, self.sat_var,
                          self.primary_val, self.avg_val, self.val_var,
                          self.retail_price, self.hue_peaks, self.val_peaks,
                          self.sat_peaks, self.labels)
