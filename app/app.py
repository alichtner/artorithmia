from flask import Flask, request, render_template, session
from cluster_engine import ClusterArt
import pandas as pd
import numpy as np


app = Flask(__name__)

# Home page
@app.route('/')
def index():
    session['art'] = range(len(c.artwork))
    session['likes'] = []
    session['dislikes'] = []

    items = [] # list of dictionaries

    collection_size = len(c.artwork)
    n_clusters = c.n_clusters
    labels = c.cluster_labels
    radius = np.random.choice([1,2,3,4,5,6,7,8], size=collection_size)
    urls = c.urls

    data = [{"id_": i, "url": urls[i].encode("utf-8"), "radius": radius[i], "cluster": labels[i]} for i in range(len(c.artwork))]
    hero_id = np.random.choice(session['art'])
    hero = data[hero_id]['url']


    item = dict(hero=hero, hero_id=hero_id, collection_size=collection_size, n_clusters=n_clusters, data=data)
    items.append(item)

    return render_template('index.html', items = items)

@app.route('/likes/<int:id>')
def likes(id):
    session['likes'].append(id)
    recommend()
    return 'itkalsjdflkajsdlf'

@app.route('/dislikes/<int:id>')
def dislikes(id):
    session['dislikes'].append(id)

@app.route('/recommend')
def recommend():
    pass



if __name__ == '__main__':
    c = ClusterArt()
    c.load_collection_from_json('app/data/Artwork.json', 'collections/shard/', 'drizl')
    c.run(10)
    app.secret_key = 'Hjadfjlji1909389283'
    app.run(host='0.0.0.0', port=8080, debug=True)
