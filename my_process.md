# Journal for Discover-Art

## Day 1
Today was spent validating some of the ideas I will be using to cluster, discover and recommend art for Drizl. The day was mainly focused on starting to build up the pipeline needed to get features for artwork. I explored different packages which can retrieve image characterists (PIL, skimage, misc, openCV, color, colorthief...). Some of the features I'd like to build are: color palette, color temperature, blurriness, symmetry, complexity.

- #### **Tasks Accomplished:**
  - started build of Art() class
    - `extract_RGB`
    - `extract_blurriness`
    - `get_palette`
  - downloaded Drizl Corpus locally
  - explored how to calculate symmetry

## Day 2
- use scipy cosine-similarity for use in sklearn's knn
- `from scipy.spatial.distance import cosine`
- then use the `cosine` model in the sklearn knn model

- To resize images above a certain threshold and then save to a new directory with the same filename as before
  - `mogrify -path <output folder> -resize <width>x\> *.jpg`
- contacted Michael Schultheis (a math-based artist in Ballard)
  - (http://www.michaelschultheis.com/)[Michael Schultheis Artist]
