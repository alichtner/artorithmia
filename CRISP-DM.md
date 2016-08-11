## What is CRISP-DM?

CRISP-DM is the most comprehensive industry standard for completing a data science project from inception to deployment. I have outlined those steps with reference to my particular problem of clustering and recommending art to Drizl's users.

### 1. Business Understanding

There are significant barriers to buying art for both long-time buyers and new entrants to the art market. Traditional art buying models rely on designers, gallery proprietors or the art buyer themselves having significant domain knowledge. Labels in this field are restrictive and can impose barriers to buyers without knowledge of exactly how an art historian would classify a piece.

**The business question:** Is there a way to represent art in an agnostic way?

### 2. Data Understanding

The Drizl collection consists of approximately 700 jpeg images of art, each with metadata concerning the retail price, size, artist, medium, surface finish and image url. At the start of the project, there was no user information.

### 3. Data Cleaning / Exploratory Analysis

The data came in the form of a json with urls to Drizl's collection online and their metadata. Initially, I spent a good deal of time looking through the collection to see what sorts of work were represented. I used imageMagick's `mogrify` command to resize the images into a smaller, workable file sizes. It became clear after a quick inspection that many pieces didn't have sizes or price listed. Some didn't have titles and some didn't have their medium listed.

### 4. Modeling

A huge part of modeling for me was feature engineering. The question became what are the best features to pull out of a piece of art. After speaking with the cofounders of Drizl as well as artists in my own network, I settled on five distinct categories of features:

  - Color (hue, saturation, value...)
  - Composition (image centroid, aspect ratio...)
  - Technique (texture, medium...)
  - Metadata (price, size...)
  - Content (subject matter)

During the two weeks we had to work on this project, I implemented many algorithms to capture features in the first four categories (Content will come later). Image related modules including `openCV`, `colorthief` and `PIL` were used for featurizing the images.

#### Clustering

After pulling features from the images and scaling them with the StandardScaler, I ran different unsupervised clustering algorithms, mainly `Kmeans` and `DBSCAN` from the sklearn.cluster module. I played with the number of classes to see how that would affect the silhouette score. The highest silhouette score (0.13) occurred at k=10 classes. This silhouette score isn't particularly high but when you consider the breadth of art in the collection, it makes sense that there wouldn't be that much difference between different works. Additional features may help to bring this number up and improve the clustering.

#### Recommender

A content-similarity-based recommender from Turi was used to provide user recommendations based on incoming likes. This is essentially a knn model that takes the feature vector provided by the user likes calulates the cosine similarity between that vector and all the images in the colleciton.

*Note: A graphlab neural network derived from `image_net` data was initially built to extract features from images, but it was decided that interpretability was an important quality in any models that were built.*

### 5. Deployment

For me, the deployment of my system was very important. I wasn't interested in having a simple recommender where you type something in and get some links to things you might like. Rather, I was interested to see if I could enhance the user experience and give them some insight as to how their preferences alter the suggestions being made to them. To that end, I made an interactive webapp with a d3-based visualization that makes suggestions across the entire corpus of images that Drizl has. I believe by showcasing the information in this manner, you invite users to explore, play around, and hopefully get interested in learning more about the art and Artorithmia as a product.

### 6. Evaluation

So far, evaluation has been based on anecdotal evidence but I do see people being interested in this product and excited for what it could show. I often get questions about what the clusters means and what they contain. Future work should focus on trying to extract meaning from the way that images are clustered.
