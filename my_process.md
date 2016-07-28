# Journal for Discover-Art

## Major Task List
- Build Feature Engine
- Build Test Suite
- Build Clusterer [KMeans, Affinity, DBSCAN]
- Build Recommender
- Build Webapp
- Build API

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
  - (http://www.michaelschultheis.com/)[Michael Schultheis Artist]

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

  - more color research (http://colorizer.org/)[Colorizer and HSV]
  - **ask what the `primary_index` is in the artwork.json**
  - **Ignored Drizl Meta:**
    - it seems like these won't necessarily play a role in clustering art or helping someone discover the art that they might like
    - `home_collection`, `weight`, `ACL`, `home_collection_index`, `supplimentaryImages`, `tryout_collection`, `tryout_collection_index`, `can_commission_diff_size`

  - Building a method so that a subset or all of the attributes of an Art object can be built up into a feature matrix
    - I would like to be able to change features on the fly without going into the Art() class code. For example, I'd like to be able to test how it clusters using only color related methods, sometimes with only metadata related features. Perhaps it still makes sense to just output everything and then drop the columns that aren't of interest to me. 
