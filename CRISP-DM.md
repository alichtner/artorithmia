## What is CRISP-DM?

CRISP-DM is the most comprehensive industry standard for completing a data science project from inception to deployment.

It consists of six steps.

1. Business Understanding

There are significant barriers to buying art for both long-time buyers and new entrants to the art market. Traditional art buying models rely on designers, gallery proprietors or the art buyer themselves having significant domain knowledge. Labels in this field are restrictive and can impose barriers to buyers without knowldge of exactly how an art historian would classify a piece.

**The business question:** Is there a way to represent art in an agnostic way?

2. Data Understanding

The Drizl collection consists of approximately 700 jpeg images of art, each with metadata concerning the retail price, size, artist, medium, surface finish and image url. At the start of the project, there was no user information.

3. Data Cleaning / Exploratory Analysis

A significant amount of time was spent on the feature engineering segment of this project. It was necessary to devise methods to adequately capture differences in different pieces of artwork. A number of image related modules including `openCV`, `colorthief` and `PIL` were used for featurizing the images.

A graphlab neural network derived from `image_net` data was initially built to extract features from images, but it was decided that interpretability was an important quality in any models that were built.

4. Modeling
5. Deployment
6. Evaluation

<p>There's a lot of amazing art out there, far too much to
  explore on your own. Using the power of machine learning
  and artist know-how, Artorithmia makes the process of
  finding pieces you like smarter and more fun.</p>

  <p>We start with a collection (corpus) of artwork. These are fed into
  Artorithmia's feature engine which pulls out features relating to color,
  composition, technique and more. Then, just as we can map a place on Earth
  using latitude and longitude coordinates, we can map each piece of art onto k-dimensional "taste" space.</p>

  <p>Artorithmia's recommender is like a GPS for art that you love. It triangulates works of
  art similar to the ones you like.</p>
