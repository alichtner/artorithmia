# Journal for Discover-Art

## Major Task List
- Build Feature Engine
- Build Test Suite
- Build Clusterer [KMeans, Affinity, DBSCAN]
- Build Recommender
- Build Webapp
- Build API
- Calculate distances between art from the Graphlab data and show the clustering

### Mini Task List
- import json metadata for the image
- make HSV vectors
- move plotting scripts to another file
- create MongoDB for json

## Day 1
Today was spent validating some of the ideas I will be using to cluster, discover and recommend art for Drizl. The day was mainly focused on starting to build up the pipeline needed to get features for artwork. I explored different packages which can retrieve image characterists (PIL, skimage, misc, openCV, color, colorthief...). Some of the features I'd like to build are: color palette, color temperature, blurriness, symmetry, complexity.

- #### **Tasks Accomplished:**
  - started build of Art() class
    - `extract_RGB`
    - `extract_blurriness`
    - `get_palette`
  - downloaded Drizl Corpus locally
    - `get_drizl_collection`
  - explored how to calculate symmetry

## Day 2
- use scipy cosine-similarity for use in sklearn's knn
- `from scipy.spatial.distance import cosine`
- then use the `cosine` model in the sklearn knn model

- To resize images above a certain threshold and then save to a new directory with the same filename as before
  - `mogrify -path <output folder> -resize <width>x\> *.jpg`
- contacted Michael Schultheis (a math-based artist in Ballard)
  - (http://www.michaelschultheis.com/) [Michael Schultheis Artist]

#### Google Hangout with Drizl team:
  - What did you cluster on before?
  - MOMA urls are only for the thumbnails, I could webscrape but if you've done some of this work already I'd rather use that
  - Do you have a sense for what makes one piece of art different or similar to another?

- #### **Tasks Accomplished:**
  - Met with the Drizl team
  - Built first Kmeans model (pretty bad)
  - Refined feature set
  - started `cluster_engine.py` to build a cluster pipeline
  - research hsv colorspace

## Day 3
  - Started the day by getting the `cluster_engine` working. It will now take a directory of images and create a `clusterArt` object which contains a collection of `Art` objects. During this process, features are calculated for every piece. A k-means model is fit after standardizing the data. The cluster_engine will return a silhouette score of how well the clusters are formed, a k_means plot showing the clusters and their centers as well as representative images closest to the cluster centers.

  - Ran the pre-built clustering engine that Drizl had built. It uses extracted features from Turi's image_net. The script takes a very LONG time to run. It also appears as though the images are scaled to be square before running through the feature extraction matrix. Graphlab extracted 4096 features from each image.

  - more color research (http://colorizer.org/) [Colorizer and HSV]
  - **ask what the `primary_index` is in the artwork.json**
  - **Ignored Drizl Meta:**
    - it seems like these won't necessarily play a role in clustering art or helping someone discover the art that they might like
    - `home_collection`, `weight`, `ACL`, `home_collection_index`, `supplimentaryImages`, `tryout_collection`, `tryout_collection_index`, `can_commission_diff_size`, ,collection_index,published, profile,description, collection,

  - Building a method so that a subset or all of the attributes of an Art object can be built up into a feature matrix
    - I would like to be able to change features on the fly without going into the Art() class code. For example, I'd like to be able to test how it clusters using only color related methods, sometimes with only metadata related features. Perhaps it still makes sense to just output everything and then drop the columns that aren't of interest to me.

## Day 4
- run PCA on extracted features from deep learning and images

- Met with Patricia and RC from Drizl to talk about assumptions about artwork and what possible features would be most important to her. We really want to try get things like mood and emotion out of the project. Extract different color palettes from a picture (look at (http://www.kuler.com) [kuler.com]). The recommender shouldn't turn people off from the works. It's dangerous to make bad assumptions. Patricia in particularly liked the `colorfulness` metric.

### Features
#### Color_Features
  - primary_hue, secondary_hue, saturation_mean, saturation_var, value_mean, value_var, **colorfulness** (hue_hist.var() if low it is colorful, if high it is more monocolor), size_of_colored_sections
    - get `size_of_colored_sections` from taking a grayscale and doing some median filters or something to get an image with a reasonable amount of colors, then count the average size that each color seems to occupy

#### Composition
  - symmetry (vert + horiz), focus (x, y) - this might be the location of the highest detail or where the most weight of the image is located, aspect_ratio, movement (average of the direction of the edges in a picture)

#### Style
  - softness (laplacian or similar), geometricness (not sure how to get this), texture (busyness, think Bragg's law and x-ray diffraction, I should probably run this on bins of the image to see if there are areas of texture)

#### Content (Google Vision API)
  - categorical (portrait, landscape, still-life, abstract, cityscape, animal, sentiment?)

#### Meta
  - price, size, medium, framed, matted, artist, is_primary

### Bayesian recommender
- Required:
  - Image Database
  - Metrics (these could be individual features or combinations of features)
- Process:
  1. For each user generate a set of reasonable assumptions for the metrics they use to like something
  2. Pick a random image [A]
  3. Randomly choose two metrics
  4. Find pieces most similar to [A] using the two metrics [B1] and [B2]
  5. Present the user [B1] and [B2]
  6. Based on the one they like, update the prior distributions to reflect the metric they probably used to select a work
  7. Draw randomly from each metric's distributions and record the winner
  8. Use the winning metric and another metric chosen at random to find pieces most similar to [A] using the two metrics [B1_new] and [B2_new]
    - make sure not to present the same pieces to the user again
  9. Present the user [B1_new] and [B2_new]
  10. Repeat steps 6 - 9 until a single metric wins out each time

### Visualize Graphlab clusters
  - compute distance metric for each of the images with the clusters that it is in as well as between cluster centers
  - plot this

# Day 5
- Started the day playing with where are the artists among the clusters, this could be an indication of the clustering, are artists being clustered together sort of thing
  - the attribute I'm calculating is called `ClusterArt.artist_cluster_score` it is an average of the maximum number of pieces that an artist has in a single cluster divided by the total number that artist has. If they have 10 pieces and all 10 are in 1 cluster, their score will be 1.0. If they have 10 pieces and each of them is in 1 of 10 clusters, their score will be 1/10. I have also saved the dictionary used to calculate this value `ClusterArt.art_dict` where the keys are the artists and the value is the individual score for that artist. This in a way is a proxy for how versatile an artist is (assuming my clustering is good).
- Used `sklearn.metrics.DistanceMetric` to calculate the euclidean distance of vectors from my features. This is output in a rather ugly png file using the `test_clustering_engine` ipython notebook
