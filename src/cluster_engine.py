import numpy as np
import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import misc
from collections import defaultdict
from collections import Counter
from sklearn.cluster import KMeans, DBSCAN
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
        """
        Initialize a ClusterArt object.

        Input:  None
        Output: None
        """
        self.artwork = []
        self.artists = []

    def run(self, n_clusters=5):
        """
        Run the clustering engine.
        Note: Collection must of loaded first.

        Input:  n_clusters (int) number of clusters to fit the features to
        Output: None
        """
        self.n_clusters = n_clusters
        self.build_features()
        self.raw_features = self.features
        self.fit('kmeans', n_clusters)
        self.predict()
        self.score()

    def run_gridsearch(self, max_clusters):
        self.build_features()
        self.raw_features = self.features
        for n in xrange(2, max_clusters):
            self.fit('kmeans', n)
            self.predict()
            self.score()

    def load_collection_from_directory(self, img_path):
        """
        -- WARNING -- CAN TAKE A LONG TIME

        Populate the ClusterArt object with an initialized Art object for each
        image in the specified filepath. This function calculates all of the
        attributes and values for each image.

        Input:  img_path (str) path to desired images
        Output: None
        """
        # add work from directory or work from csv option
        # add functionality to ignore .DS_store file
        for idx, image in enumerate(os.listdir(img_path)):
            time.sleep(0)
            sys.stdout.write("\r-- %d %% Artwork Loaded -- " % (1.*idx/len(drizl)*100))
            sys.stdout.flush()
            if image != '.DS_Store':
                art = Art(item_id=idx)
                try:
                    art.load_image(img_path+ image)
                except ValueError:
                    print 'Wrong dimensions'
                self.artwork.append(art)
            self.n_artworks = len(self.artwork)
        print '-- {} images added to collection'.format(self.n_artworks)
        print ' -------------------------------- \n '

    def load_collection_from_json(self, catalog, img_path=None):
        """
        -- WARNING -- CAN TAKE A LONG TIME

        Populate the ClusterArt object with an initialized Art object for each
        row of a json file with references to images. Metadata about artwork is
        added to each Art object and used during feature building. This
        function calculates all of the attributes and values for each image.

        Input:  json_file (str) path to json file to be used to parse data
                img_path (str) path to desired images
        Output: None
        """
        # use the json file from cloudinary to direct how the collection
        # should be built

        df = pd.read_json(catalog, orient='records')

        item_id = 0

        print '\n   Loading collection from JSON file'
        print '   ------------------------------- '
        print '   Analyzing and Generating Features for Collection\n'

        for row in xrange(len(df)):
            time.sleep(0)
            sys.stdout.write("\r-- %d %% Artwork Loaded -- " % (1.*row/len(df)*100))
            sys.stdout.flush()
            try:
                img_name = df['results'][row]['metadata']['public_id'].replace('/', '_')
                art = Art(item_id=item_id)
                art.load_image(img_path + img_name + '.jpg', df['results'][row])
                self.artwork.append(art)
                item_id += 1
            except Exception:
                print 'Missing image file: {}'.format(img_name)

        self.n_artworks = len(self.artwork)
        print ""
        print '-- {} images added to collection'.format(self.n_artworks)
        print ' -------------------------------- \n '

    def build_features(self):
        """
        Create list of ids, artists, and an array of artwork features.

        - collection_ids (list) ids used by the recommender engine
        - artists (list) artists used to score clustering
        - features (numpy array) features with each row being a piece of art
                   and the columns being feature values.

        Input:  None
        Output: None
        """
        print "--- Building Art Features --- "
        # number of features
        self.no_features = self.make_feature_row(self.artwork[0]).shape[0]

        # get first item_id and artist
        self.collection_ids = [self.artwork[0].item_id]
        self.urls = [self.artwork[0].url]
        self.artists = [self.artwork[0].short_name]
        self.titles = [self.artwork[0].title]

        # initialize the first row of features to get the size of the array
        self.features = self.make_feature_row(self.artwork[0]).reshape(1, self.no_features)
        # loop through artworks in the ClusterArt object and pull out feature rows
        for art in self.artwork[1:len(self.artwork)]:
            self.collection_ids.append(art.item_id)  # item_id for recommender
            self.urls.append(art.url)
            self.artists.append(art.artist)  # create list of artists
            self.titles.append(art.title)
            row = self.make_feature_row(art).reshape(1, self.no_features)
            self.features = np.concatenate((self.features, row), axis=0)
        # self.fill_sizes()

    def make_feature_row(self, art):
        """
        Helper function to get the values for each art object. Pulls features
        out by Art() object attribute name. Change the `feat_names` list to
        get different features.

        Input:  art (Art object) the art object to take features from
        Output: feature row (np.array)
        """

        self.feat_names = ['primary_hue', 'avg_hue', 'hue_var', 'primary_sat',
                           'avg_sat', 'sat_var', 'primary_val', 'avg_val',
                           'val_var', 'symmetry', 'bluriness', 'aspect_ratio',
                           'retail_price', 'area', 'width', 'height']
        color_features = np.array([getattr(art, self.feat_names[0]), getattr(art, self.feat_names[1]),
                                   getattr(art, self.feat_names[2]), getattr(art, self.feat_names[3]),
                                   getattr(art, self.feat_names[4]), getattr(art, self.feat_names[5]),
                                   getattr(art, self.feat_names[6]), getattr(art, self.feat_names[7]),
                                   getattr(art, self.feat_names[8])])
        comp_features = np.array([getattr(art, self.feat_names[9]), getattr(art, self.feat_names[10]),
                                  getattr(art, self.feat_names[11])])
        meta_features = np.array([getattr(art, self.feat_names[12]), getattr(art, self.feat_names[13]),
                                  getattr(art, self.feat_names[14]), getattr(art, self.feat_names[15])])

        return np.concatenate((color_features, comp_features, meta_features))
        # np.concatenate((single_values, art.red_bins, art.grn_bins,
        # art.blue_bins, art.hue_bins, art.sat_bins,
        # art.val_bins))

    def pandas_data(self, savepath=None):
        """
        Outputs clean pandas dataframe of both the raw and scaled data.

        Input:  save (str) name of dataframe to save
        Output: None
        """
        # add artists, titles
        raw_df = pd.DataFrame(data=self.raw_features, columns=self.feat_names)
        identity = pd.DataFrame(data={'item_id': self.collection_ids,
                                      'url': self.urls,
                                      'cluster_id': self.cluster_labels,
                                      'title':self.titles})
        raw_df = pd.concat([identity, raw_df], axis=1)

        # get the scaled data
        df = pd.DataFrame(data=self.features, columns=self.feat_names)
        df = pd.concat([identity, df], axis=1)

        if savepath is not None:
            raw_df.to_csv(savepath + 'drizl_raw.csv')
            df.to_csv(savepath + 'drizl_scaled.csv')
        return df

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

    def fit(self, model, n_clusters=5):
        """
        Fits clusters to the feature set using a Kmeans model.

        Input:  n_clusters (int) number of clusters to use during clustering
        Output: None
        """
        self.n_clusters = n_clusters
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

        if model == 'kmeans':
            self.model = KMeans(self.n_clusters)
        elif model == 'DBSCAN':
            self.model = DBSCAN(eps=0.3, min_samples = 3)
        self.cluster_fit = self.model.fit(self.features)
        print ('-- Running clustering on {} piece collection --'
               .format(self.n_artworks))

    def predict(self):
        """
        Predict class labels for each piece of art.

        Input:  None
        Output: None
        """
        self.cluster_labels = self.model.predict(self.features)

    def score(self):
        """
        Build and format command line output for scoring a clustering.

        Input:  None
        Output: command line scoring output
        """
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
        print '    Features considered ', self.feat_names
        print '\n  ---------------------------------  \n'

    def silhouette(self):
        """
        Calculate the silhouette score for a certain clustering.

        Input:  None
        Output: silhouette score (None)
        """
        return silhouette_score(self.features, self.cluster_labels)

    def show_opposites(self, feature):
        """
        Show images at either range of a particular feature.

        Input:  feature (str) attribute or feature name
        Output: None
        """
        self.artwork.sort(key=lambda x: x.feature, reverse=True)
        print self.artwork[0].feature
        self.artwork[0].show_image()
        print self.artwork[-1].feature
        self.artwork[-1].show_image()

    def score_artist_clusters(self):
        """
        Builds up artist cluster scores based on how often an artist's work ends up
        in a single cluster versus many clusters.

        Input:  None
        Output: average artist cluster score (float)
        """
        # score artists in clusters
        # if an artists' work is all in one cluster give a score of 1
        # need way to penalize by the number of clusters
        # also this doesn't take into account if an artist has entirely
        # different styles in their collection
        self.art_dict = defaultdict(list)
        for idx, artist in enumerate(self.artists):
            if artist not in self.art_dict:
                self.art_dict[artist] = [self.cluster_labels[idx]]
            else:
                self.art_dict[artist].append(self.cluster_labels[idx])
        for artist in self.art_dict.keys():
            self.art_dict[artist] = Counter(self.art_dict[artist])
            self.art_dict[artist] = (1.*max(self.art_dict[artist].values()) /
                                     sum(self.art_dict[artist].values()))
        return sum(self.art_dict.values())/float(len(self.art_dict.values()))

    def calc_art_distance(self):
        dist = DistanceMetric.get_metric('euclidean')
        self.art_distances = dist.pairwise(self.features)

    def plot_by_attribute(self, attr='avg_sat', no_pieces=5):
        """
        Plot art by decreasing attribute value.
        Example Attributes: 'retail_price', 'size', 'avg_hue', 'avg_sat'

        Input:  attr (str) attribute to be used to show art by
                no_pieces (int) number of pieces to show
        Output: plot objects
        """
        self.artwork.sort(key=lambda x: getattr(x, attr), reverse=True)
        for idx in xrange(0, self.n_artworks, self.n_artworks/no_pieces):
            print self.artwork[idx]
            print getattr(self.artwork[idx], attr)
            self.artwork[idx].show_image()
            plt.show()

    def exemplar_images(self, plot=False):
        """
        Find images closest to cluster centers.

        Input:  plot (bool) show the images
        Output: plot objects
        """
        self.exemplar_images, _ = pairwise_distances_argmin_min(
                                self.cluster_fit.cluster_centers_,
                                self.features, 'euclidean')
        if plot is True:
            for c in self.exemplar_images:
                self.artwork[c].show_image()

    def more_like_this(self, clusters=None, n_images=3):
        """
        Return n_images from a cluster

        Input:  cluster (int) cluster label to take similar artwork from
                n_images (int) number of images to return from the cluster
        Output: images from cluster
        """
        for cluster in xrange(self.n_clusters):
            # maybe shuffle the artwork here
            cluster_list = [idx for idx, lab in enumerate(self.cluster_labels)
                            if lab == cluster]
            for idx in xrange(n_images):
                self.artwork[cluster_list[idx]].show_image()

    def plot_kmeans(self, x, y):
        """
        Plot the kmeans scatterplot of the Art objects by certain features (x, y).
        Cluster centers are plotted on the graph as well.

        Input:  x (int) index of feature column for independent variable
        Output: y (int) index of feature column for dependent variable
        """
        viz.plot_kmeans(self.features, self.feat_names, self.cluster_labels,
                        self.cluster_fit.cluster_centers_, x, y)


if __name__ == '__main__':
    f = 'collections/test_small/'
    cluster = ClusterArt()
    cluster.load_collection_from_json('data/Artwork.json',
                                      'collections/drizl/all_small/')
    cluster.run()
    features, labels, centers = cluster.return_all()
    cluster.silhouette()
    cluster.get_exemplar_images(plot=True)
    cluster.plot_kmeans(6, 8)
