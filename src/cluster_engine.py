# -- Desired Process --
# Load a directory of images
# Calculate features for all the images
# Cluster together
# Return representative images or images from a cluster

import os
import numpy as np
from scipy import misc
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import DistanceMetric
from collections import defaultdict
from collections import Counter


from Art_class import Art
import art_plot as viz


class ClusterArt(object):
    """
    Build and test clustering

    INPUT:
    OUTPUT:
    """
    def __init__(self):
        self.artwork = []
        self.n_artworks = None
        self.artists = []
        self.n_clusters = 7
        self.model = KMeans(self.n_clusters)
        self.features = None
        self.cluster_fit = None
        self.cluster_labels = None
        self.exemplar_images = None
        self.metadata = None

    def load_collection(self, images_filepath, add_meta=False):
        # add functionality to ignore .DS_store file
        for image in os.listdir(images_filepath):
            if image != '.DS_Store':
                art = Art()
                try:
                    art.load_image(images_filepath + image)
                except ValueError:
                    print 'Wrong dimensions'
                if add_meta is True:
                    image_meta = image.replace('_', '/').split('.')[0]
                    art.parse_meta(self.metadata['metadata']['public_id'] == image_meta)
                self.artwork.append(art)
            self.n_artworks = len(self.artwork)
        print '{} images added to collection'.format(len(os.listdir(images_filepath)))
        self.build_features()
        print " --- Building feature set --- "

    def load_json(self, json_filepath):
        with open(json_filepath) as f:
            self.metadata = json.load(f)

    def make_feature_row(self, art):
        single_values = np.array([art.avg_hue, art.avg_sat, art.avg_val, art.hue_var, art.sat_var, art.val_var,
                                  art.primary_hue, art.primary_sat, art.primary_val,
                                  art.symmetry, art.bluriness, art.aspect_ratio])
        return single_values
        #np.concatenate((single_values, art.red_bins, art.grn_bins,
                              # art.blue_bins, art.hue_bins, art.sat_bins,
                              # art.val_bins))

    def build_features(self):
        self.no_features = self.make_feature_row(self.artwork[0]).shape[0]
        self.features = np.empty((1, self.no_features))
        for art in self.artwork:
            self.artists.append(art.artist) # create list of artists
            row = self.make_feature_row(art).reshape(1, self.no_features)

            self.features = np.concatenate((self.features, row), axis=0)

    def get_all_features(self):
        # for art in self.artwork:
        #   features = art.__dict__.values()
        pass

    def fit(self):
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)
        self.cluster_fit = self.model.fit(self.features)
        print ('-- Running clustering on {} piece collection --'
               .format(len(self.artwork)))
        print self.model.score(self.features)

    def predict(self):
        self.cluster_labels = self.model.predict(self.features)

    def silhouette(self):
        silhouette_avg = silhouette_score(self.features, self.cluster_labels)
        print "For n_clusters = {} \n The average silhouette_score is : {}"\
              .format(self.n_clusters, silhouette_avg)

    def return_all(self):
        """
        Returns calculated features, predicted labels and predicted cluster centers
        """
        return self.features, self.cluster_labels, self.cluster_fit.cluster_centers_

    def show_opposites(self, feature):
        self.artwork.sort(key=lambda x: x.feature, reverse=True)
        print self.artwork[0].feature
        self.artwork[0].show_image()
        print self.artwork[-1].feature
        self.artwork[-1].show_image()


    def score_artist_clusters(self):
        # score artists in clusters
        # if an artists' work is all in one cluster give a score of 1
        # need way to penalize by the number of clusters
        # also this doesn't take into account if an artist has entirely different styles in their collection
        self.art_dict = defaultdict(list)
        for idx, artist in enumerate(self.artists):
            if artist not in self.art_dict:
                self.art_dict[artist] = [self.cluster_labels[idx]]
            else:
                self.art_dict[artist].append(self.cluster_labels[idx])
        for artist in self.art_dict.keys():
            self.art_dict[artist] = Counter(self.art_dict[artist])
            self.art_dict[artist] = 1.*max(self.art_dict[artist].values())/sum(self.art_dict[artist].values())
        self.artist_cluster_score = sum(self.art_dict.values())/float(len(self.art_dict.values()))

    def calc_art_distance(self):
        dist = DistanceMetric.get_metric('euclidean')
        self.art_distances = dist.pairwise(self.features)

    def plot_art_by_color(self, no_pieces=5):
        """
        Plot art by decreasing hue red -> blue -> red
        !!! (it's a wheel rememeber stupid!)
        """
        self.artwork.sort(key=lambda x: x.primary_hue, reverse=True)
        for idx in xrange(0, self.n_artworks, self.n_artworks/no_pieces):
            print self.artwork[idx].primary_hue
            self.artwork[idx].show_image()
            plt.plot(self.artwork[idx].hue_bins)
            plt.show()

    def plot_art_by_saturation(self, no_pieces=5):
        """
        Plot art by decreasing saturation
        """
        self.artwork.sort(key=lambda x: x.primary_sat, reverse=True)
        for idx in xrange(0, self.n_artworks, self.n_artworks/no_pieces):
            print self.artwork[idx].primary_sat
            self.artwork[idx].show_image()
            plt.plot(self.artwork[idx].sat_bins)
            plt.show()

    def plot_art_by_value(self, no_pieces=5):
        """
        Plot art by decreasing value
        """
        self.artwork.sort(key=lambda x: x.primary_val, reverse=True)
        for idx in xrange(0, self.n_artworks, self.n_artworks/no_pieces):
            print self.artwork[idx].primary_val
            self.artwork[idx].show_image()
            plt.plot(self.artwork[idx].val_bins)
            plt.show()

    def get_exemplar_images(self, plot=False):
        """
        Find images closest to cluster centers
        """
        self.exemplar_images, _ = pairwise_distances_argmin_min(
                                self.cluster_fit.cluster_centers_,
                                self.features, 'euclidean')
        if plot is True:
            for c in self.exemplar_images:
                self.artwork[c].show_image()

    def more_like_this(self, cluster=0, n_images=3):
        """
        Return n_images from a cluster

        INPUT: n_images (images to return), cluster (cluster label)
        OUTPUT: images from cluster
        """
        pass

    def plot_kmeans(self, x, y):
        viz.plot_kmeans(self.features, self.cluster_labels,
                        self.cluster_fit.cluster_centers_, x, y)


if __name__ == '__main__':
    f = 'collections/drizl/all_small/'
    cluster = ClusterArt()
    cluster.load_collection(f)
    cluster.build_features()
    cluster.fit()
    cluster.predict()
    cluster.score_artist_clusters()
    #features, labels, centers = cluster.return_all()
    #cluster.silhouette()
    #cluster.get_exemplar_images(plot=True)
    #cluster.plot_kmeans(0, 1)
