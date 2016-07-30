import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import misc
from collections import defaultdict
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import DistanceMetric

from Art_class import Art
import art_plot as viz


class ClusterArt(object):
    """
    Build features and cluster a collection of artwork.
    """
    def __init__(self):
        self.artwork = []
        self.artists = []

    def run(self):
        """
        Run the clustering engine.
        Note: Collection must of loaded first.
        """
        self.build_features()
        self.fit(5)
        self.predict()
        self.score()

    def load_collection_from_directory(self, images_filepath):
        # add work from directory or work from csv option
        # add functionality to ignore .DS_store file
        for image in os.listdir(images_filepath):
            if image != '.DS_Store':
                art = Art()
                try:
                    art.load_image(images_filepath + image)
                except ValueError:
                    print 'Wrong dimensions'
                self.artwork.append(art)
            self.n_artworks = len(self.artwork)
        print '{} images added to collection'.format(self.n_artworks)
        print " --- Building feature set --- "

    def load_collection_from_json(self, json_file, img_filepath=None):
        # use the json file from cloudinary to direct how the collection
        # should be built
        drizl = pd.read_json(json_file, orient='records')
        for row in xrange(len(drizl)):
            try:
                img_name = drizl['results'][row]['metadata']['public_id'].replace('/', '_')
                art = Art()
                art.load_image(img_filepath + img_name + '.jpg',
                               drizl['results'][row])
                print 'Loading Artwork: ', img_name
                self.artwork.append(art)
            except Exception:
                print 'No such file or directory'
        self.n_artworks = len(self.artwork)
        print '{} images added to collection'.format(self.n_artworks)
        print " --- Building feature set --- "
        self.build_features()

    def build_features(self):
        # Create a numpy array of features with each row being an image and
        # the columns being feature values. Also initialize a list of artists
        # for each work to be used when scoring the clustering
        # create first feature row to append to
        self.no_features = self.make_feature_row(self.artwork[0]).shape[0]
        self.artists = [self.artwork[0].artist]  # create first artist
        self.features = self.make_feature_row(self.artwork[0]).reshape(1, self.no_features)
        for art in self.artwork[1:len(self.artwork)]:
            self.artists.append(art.artist)  # create list of artists
            row = self.make_feature_row(art).reshape(1, self.no_features)
            self.features = np.concatenate((self.features, row), axis=0)
        #self.fill_sizes()

    def make_feature_row(self, art):
        # The featues to be extracted from each Art object
        # Color Features: hue, sat, val
        # Composition Features: symmetry, bluriness, aspect_ratio
        # Meta Features: retail_price, area, width, height
        # need to account for meta features when they aren't there

        color_features = np.array([art.primary_hue, art.avg_hue, art.hue_var,
                                   art.primary_sat, art.avg_sat, art.sat_var,
                                   art.primary_val, art.avg_val, art.val_var])
        comp_features = np.array([art.symmetry, art.bluriness,
                                  art.aspect_ratio])
        meta_features = np.array([art.retail_price, art.area, art.width,
                                  art.height])

        return np.concatenate((color_features, comp_features, meta_features))
        # np.concatenate((single_values, art.red_bins, art.grn_bins,
        # art.blue_bins, art.hue_bins, art.sat_bins,
        # art.val_bins))

    def fill_sizes(self):
        pass
        # # replaces ungiven values for art work size with averages
        # col_mean = stats.(self.features, axis=0)
        # print col_mean
        # #Find indicies that you need to replace
        # inds = np.where(self.features < -998)
        #
        # #Place column means in the indices. Align the arrays using take
        # self.features[inds]=np.take(col_mean,inds[1])

    def fit(self, n_clusters=5):
        """
        Fits clusters to the feature set.
        """
        self.n_clusters = n_clusters
        self.model = KMeans(self.n_clusters)
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)
        self.cluster_fit = self.model.fit(self.features)
        print ('-- Running clustering on {} piece collection --'
               .format(self.n_artworks))

    def predict(self):
        """
        Predict class labels for each piece of art.
        """
        self.cluster_labels = self.model.predict(self.features)

    def score(self):
        self.model_score = self.model.score(self.features)
        self.artist_cluster_score = self.score_artist_clusters()
        self.silhouette_avg = self.silhouette()
        print '\n\n  ---- CLUSTERING RESULTS ----  '
        print '\n    Number of artworks: ', self.n_artworks
        print '    Number of clusters: ', self.n_clusters
        print '\n  ---------------------------------  \n'
        print '    Average Cluster Size: '
        print '    Model Score: ', self.model_score
        print '    Artist Clustering Score: ', self.artist_cluster_score
        print '    Silhouette Score: ', self.silhouette_avg
        print '    Features considered ???'
        print '\n  ---------------------------------  \n'

    def silhouette(self):
        return silhouette_score(self.features, self.cluster_labels)

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
        return sum(self.art_dict.values())/float(len(self.art_dict.values()))

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
        self.artwork.sort(key=lambda x: x.avg_sat, reverse=True)
        for idx in xrange(0, self.n_artworks, self.n_artworks/no_pieces):
            print self.artwork[idx].primary_sat
            self.artwork[idx].show_image()
            plt.plot(self.artwork[idx].sat_bins)
            plt.show()

    def plot_art_by_value(self, no_pieces=5):
        """
        Plot art by decreasing value
        """
        self.artwork.sort(key=lambda x: x.avg_val, reverse=True)
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
    f = 'collections/test_small/'
    cluster = ClusterArt()
    #cluster.load_collection_from_directory(f)
    cluster.load_collection_from_json('data/Artwork.json', 'collections/drizl/all_small/')
    cluster.run()
    features, labels, centers = cluster.return_all()
    cluster.silhouette()
    cluster.get_exemplar_images(plot=True)
    cluster.plot_kmeans(6, 8)
