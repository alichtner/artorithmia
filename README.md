# Artorithmia

#### Art Clustering and Recommendation
*Capstone done in partnership with Drizl*

## Project Motivation

Applying labels to works of art is inherently subjective and often counter-productive. Whether or not a piece is considered *neo-classical* or *post-modern* has little bearing on whether or not a person will actually like or dislike the work. Additionally, labels create barriers to people who may be interested in art but aren't familiar with accepted terminologies. Ideally, we'd like to represent a piece of art using a set of unbiased metrics. **Artorithmia** attempts to do this by extracting features from unlabelled images and

## Pipeline

![project pipeline](images/pipeline.png)

A corpus of unlabelled pieces of art are fed into the featurization engine where a variety of features are pulled out of each image. The images are then clustered with the cluster engine into a k-classes. A graphlab content-similarity-based recommender is used to compute the similarities between pieces of art. Both the clustering data and recommender are deployed on a flask-based webapp using Amazon Web Services.

![tech stack](images/tech_stack.png)

## Create recommendations for users using "taste space"

## Future Development



## API Calls

- **Return Representative Images**
- **Return Similar Images**
- **Return User Recommendations**
