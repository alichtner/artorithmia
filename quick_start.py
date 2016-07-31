from src.cluster_engine import ClusterArt
import sys

c = ClusterArt()
if sys.argv[1] == 'shard':
    c.load_collection_from_json('data/Artwork.json', 'collections/shard/')
else:
    c.load_collection_from_json('data/Artwork.json', 'collections/drizl/all_small/')
c.run()
