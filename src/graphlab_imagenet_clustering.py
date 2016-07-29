import graphlab as gl
import json
import httplib
import urllib
import os

gl.product_key.set_product_key(os.environ['GRAPHLAB_PRODUCT_KEY'])

# GRAB URLs FROM PARSE
connection = httplib.HTTPSConnection('api.parse.com', 443)
params = urllib.urlencode({"limit": 1000})
connection.connect()
connection.request('GET', '/1/classes/Artwork?%s' % params, '', {
       "X-Parse-Application-Id": "CzDFFJj9HWgZpbeH35snFqUxhta3OT7mrvEPcaPe",
       "X-Parse-REST-API-Key": "dD0Kuvi4Tzucit7Q5QDG4ncEsXCV7iJat56vZUw6"
     })
result = json.loads(connection.getresponse().read())

# TRANSFORM URLs FOR CLOUDINARY
url_list = []
ourResult = result['results']
for i in ourResult:
    j = i['image']
    if ("png" in j) or ("jpg" in j):
        if not ("pardue" in j):
            j = j.replace('png', 'jpg')
            url_list.append(j.replace('upload/'',
                            'upload/c_fill,g_center,h_256,w_256/''))

#  USING GRAPHLAB CLASSES
# Should take a few minutes - grabbing all images.
# Future iteration will save image number %APPDATA% to just grab deltas
url_sarray = gl.SArray(url_list)
image_sarray = url_sarray.apply(lambda x: gl.Image(x))

images = gl.SFrame({'image': image_sarray})


# GRAB PRETRAINED MODEL
# Takes a minute
pretrained_model = gl.load_model('http://s3.amazonaws.com/dato-datasets/deeplearning/imagenet_model_iter45')

# EXTRACT FEATURES###
# this step might take a hot minute (10-20 minutes)
images['extracted_features'] = pretrained_model.extract_features(images)


# KMEANS CLUSTERING
# Creates SFrame object using the extracted features
# Change num_clusters number to change the number of clusers
sf = gl.SFrame(images['extracted_features'])
model = graphlab.kmeans.create(sf, num_clusters=20, max_iterations=20)

# CREATE SARRAY AND ADD URLs###
# you know, just to have everything in one place
sa = gl.SArray(url_list)
model.cluster_id.add_column(sa)

# TURN TO JSON
# First parameter controls where it gets outputed - specify file path if you
# have a preference
# Save all the things
model.cluster_id.export_json('output.json', orient = 'records')
images['extracted_features'].save('../data/graphlab_extracted_features.csv', format='csv')
model.cluster_info.save('../data/graphlab_cluster_info.csv', format='csv')
model.cluster_id.save('../data/graphlab_cluster_id.csv', format='csv')
