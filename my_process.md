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

# Color Temperature

  - The hottest colour is completely red (R=255, G=0, B=0).
  - The coldest colour is completely blue (R=0, G=0, B=255).

  So the more R you have, the warmer the colour, and the more B you have, the cooler the colour. The G takes you through shades of turquoise, green, and yellow, each being increasingly warmer. When G is zero, you move in the lower left diagonal of the circle when R and B change. As G approaches 255, you cross over the upper right diagonal when R and B change.

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
- I am removing the `sold` column from the dataset since it appears to be corrupted. In some of the JSON, that key:value does not even exist. I think this is an ok assumption since, although being sold would indicate a higher value for a piece, you aren't necessarily going to want to show someone a piece if it's sold (just kidding, I added it back in and assumed anything that didn't have a value was false)

### JSON meta
- You can now load metadata from the json itself. This is fine as long as the fields are the same as the example one I have from Drizl. If the values are different I'll have to take that into consideration. The cluster engine now uses fields like price and size to cluster on as well as color values. *There are values with None in them for price and size/width/height that I have currently set to zero but they should be dealt with in some way (currently they are skewing results)*

### Webapp
- started working on the webapp today, chose a bootstrap theme and built the directories for it
- got a simple d3 visualization to work on the page just to have an example of how it should be
- for the webapp, I asked RC how to actually host the images for it and I can either do it locally or host it with a service (cloudinary or similar)

### Thinking about the overall process
- load images and metadata into a clusterArt object containing multiple art objects with calculated features (save the features)
- cluster them

### Next Steps:
- start working on the recommender portion
- create dummy variables for categorical values
- check how the scaling is working with the features
- add the final features

# Day 6
- Added functionality to show artwork images in order of decreasing attribute value
  - used `getattr(object, attr_name) => value` in the lambda expression to plot this
- Started building up labels for things that can print out next to an image
  - somewhat arbitrarily defined the primary colors - I gave the green and blue-green a larger range where they are binned which is in accordance with color theory.

- (https://blog.ytotech.com/2015/11/01/findpeaks-in-python/) used the `detect_peaks` function to find peaks
### Recommender
- I ended up building a content_based recommender using graphlab. This takes an item_id (which is just the position in the artwork list) and the features output by the cluster_engine
- At initiation I should show people images from around the cluster centers to give them the best breadth of what the corpus has to offer. That might help the recommender to converge more quickly at a reasonable level of success.
- The recommender is currently working. You just have to run `gl_recommender.py` using whatever collection library you want. It will load the cluster object and calculate a bunch of things. Afterwards it will cluster images together and put class labels on things.
- It then asks you for pieces that you like: input them like `0 45 6 2` where the integers are the `item_id`s for the pieces that you like. the script will calculate your most likely likes and then show them to you.
- **looks like you can supply weight for what is most important for you when recommending**
- https://github.com/MonsieurV/py-findpeaks/blob/master/tests/libs/detect_peaks.py source for detect_peaks

# Day 7
- Fixed multicolor and single color bug so that it used the number of hue peaks rather than the hue variance to tell if something was made using highly divergent colors rather than a fixed section of the color wheel - seems to be a bit better

## Started EC2 and EFS Instances
- From AWS console, start EFS instance
- From AWS console, start EC2 instance (used c3.8xlarge with 20GB local storage)
- Set rules so that I can access from galvanize's two IP addresses and will have to add my home address later so that it can connect from home
- EFS is elastic file system which scales as you add things
- set security groups in the EC2 and EFS so that they can talk to each other
- **ssh into AWS EC2**
  `ssh -i ~/.ssh/DiscoverArt.pem ubuntu@ec2-52-43-69-244.us-west-2.compute.amazonaws.com`
  - `sudo apt-get install nfs-common`
  `sudo mkdir efs`
  - `sudo mount -t nfs4 -o nfsvers=4.1 us-west-2b.fs-5b8272f2.efs.us-west-2.amazonaws.com:/ efs`

-**buiding up my datascience stack**
  `sudo wget <link to anaconda download .sh file>`
    - http://repo.continuum.io/archive/Anaconda2-4.1.1-Linux-x86_64.sh
  `bash anaconda....`
    - follow instructions
  - open new terminal for changes to be resolved
  `conda update conda`
  `conda install scikit-image`
  `conda update --all`
  `sudo git clone discover-art` onto the efs drive
  `scp -i <path to key> <path to file> ubuntu@.....:~` to send file to my ec2
  - change the bash_profile to have the cloudinary and graphlab key-pairs
  - to get graphlab on EC2
    - `conda update pip`
    - `pip install --upgrade --no-cache-dir https://get.graphlab.com/GraphLab-Create/2.1/your registered email address here/your product key here/GraphLab-Create-License.tar.gz`
    - `sudo apt-get install libgomp1` need to run this command since this is not installed on ubuntu by default and graphlab needs it
    - `sudo apt-get install python-matplotlib`
    - `sudo apt-get install libsm6`
    - `pip install colorutils`
    - `pip install colorthief`
    - `pip install seaborn`
    - `sudo apt-get install imagemagick`
    - `sudo apt-get install git`
    - get openCV
      - followed this blog entry http://milq.github.io/install-opencv-ubuntu-debian/


## tmux
- can use `tmux` to run things on EC2 without worrying about it closing when I shut a terminal or turn my computer off
- tmux also let's me run things on my EC2 in the background and I'm able to do other things in the terminal while it runs
- **tmux commands**
  - start tmux `tmux`
  - run process blah blah blah
  - go back to terminal `ctrl + b, then d`
  - to get back in and view output `tmux attach`

## tor -- for webscraping
- I was getting severely throttled back downloading the MOMA database.
  - https://deshmukhsuraj.wordpress.com/2015/03/08/anonymous-web-scraping-using-python-and-tor/
  - is a blog post about how to rotate you IP address through a tor network so that you don't get throttled back

## Web Gallery of Art
- this is a database of lots of artworks
- they provide a file of their archives
  - http://www.wga.hu/database/download/data_txt.zip
- wrote a script that finds the 2D works and then downloads them from the internet. These are being pulled into the EFS system using an EC2 instance

# Day 8
- the majority of art from MOMA (87%) and WGA (99%) were downloaded onto EFS last night
- switched Art class `extract_blur` to use the skimage laplacian rather than OpenCV since I can't get the OpenCV one to install on amazon
- edited the average and variance attributes of hue, saturation and color so that it uses the entire pixel range of the image, not just the histogram (which was totally wrong)
- spent a lot of time today fixing up the cluster_engine so that it can take wga data which is slightly different from the drizl data
  - this still hasn't been totally sucessful
### d3
- built a visualization which will connect up with flask app to give the cluster label and a size. Every time a like is given for an image, the probability of liking the other images can be updated by changing the sizes. This will effectively show the person traversing the corpus and help them to explore what's around.
  - still need to build the recommender into the model and build up the ability to present the image and metadata when a dot is clicked on.

# Day 9
- added function to transform the drizl urls to grab smaller sizes right away
