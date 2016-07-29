from sklearn.neighbors import DistanceMetric

def calc_art_distance(art_features):
    dist = DistanceMetric.get_metric('euclidean')
    return dist.pairwise(art_features)
