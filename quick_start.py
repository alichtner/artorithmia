import sys
import graphlab as gl
from src.cluster_engine import ClusterArt

"""
Quick-Start script to run the clustering (and recommender) on either the shard
collection or on the full Drizl collection. Run this in ipython to get access
to the ClusterArt object and recommender methods.

Input:  argv[1]: 'shard' or <datafile used to load>
        argv[2]:  <location of images>
        argv[3]: 'rec' or 'none'
Output: None
"""

c = ClusterArt()
if sys.argv[1] == 'shard':
    c.load_collection_from_json('data/Artwork.json', 'collections/shard/')
else:
    c.load_collection_from_json(sys.argv[1], sys.argv[2])
c.run()

if sys.argv[3] == 'rec':
    lab = gl.SArray(c.collection_ids)
    data = gl.SArray(c.features)
    combined = gl.SFrame([lab, data])
    combined = combined.unpack('X2')
    rec = gl.recommender.item_content_recommender.create(combined, 'X1')
    likes = [int(val) for val in raw_input('Which pieces do you like? ').split()]
    print likes
    pred = rec.recommend_from_interactions(likes)
    for item_id in likes + [pred[0]['X1']]:
        c.artwork[item_id].show_image()
