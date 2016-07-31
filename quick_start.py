from src.cluster_engine import ClusterArt

c = ClusterArt()
c.load_collection_from_json('data/Artwork.json', 'collections/shard/')
c.run()
