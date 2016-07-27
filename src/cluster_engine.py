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

from Art_class import Art


class ClusterArt(object):
    """
    Build and test clustering

    INPUT:
    OUTPUT:
    """
    def __init__(self):
        self.artwork = []
        self.n_clusters = 5
        self.model = KMeans(self.n_clusters)
        self.features = None
        self.cluster_fit = None
        self.cluster_labels = None
        self.exemplar_images = None

    def load_collection(self, filepath):
        for image in os.listdir(filepath):
            art = Art()
            art.load_image(filepath + image)
            self.artwork.append(art)
        print '{} images added to collection'.format(len(os.listdir(filepath)))

    def build_features(self):
        # create simple feature set of symmetry, blurriness, aspect_ratio
        self.features = np.zeros((1, 3))
        for art in self.artwork:
            row = np.array([art.symmetry, art.bluriness,
                            art.aspect_ratio]).reshape((1, 3))
            self.features = np.concatenate((self.features, row), axis=0)

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
        d = {0: 'r', 1: 'b', 2: 'g', 3: 'c', 4: 'm'}

        colors = []
        for i in self.cluster_labels:
            colors.append(d[i])
        # plot the kmeans clustering
        plt.figure(figsize=(10, 6))
        plt.xlim(self.features[:, x].min() - 0.1 * self.features[:, x].max(),
                 self.features[:, x].max() + 0.1 * self.features[:, x].max())
        plt.ylim(self.features[:, y].min() - 0.1 * self.features[:, y].max(),
                 self.features[:, y].max() + 0.1 * self.features[:, y].max())
        plt.scatter(self.features[:, x], self.features[:, y],
                    c=colors, edgecolors='face', alpha=0.3)
        plt.scatter(self.cluster_fit.cluster_centers_[:, x],
                    self.cluster_fit.cluster_centers_[:, y],
                    c=['r', 'b', 'g', 'c', 'm'], s=50)
        plt.show()

if __name__ == '__main__':
    f = 'collections/drizl/resize/test_small/'
    cluster = ClusterArt()
    cluster.load_collection(f)
    cluster.build_features()
    cluster.fit()
    cluster.predict()
    cluster.silhouette()
    cluster.get_exemplar_images(plot=False)
    cluster.plot_kmeans(0, 2)
