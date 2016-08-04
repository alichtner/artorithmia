from flask import Flask, request, render_template, session
import graphlab as gl
import pandas as pd
import numpy as np


app = Flask(__name__)

# Home page
@app.route('/')
def index():
    session['art'] = range(len(df))
    session['likes'] = []
    session['dislikes'] = []

    items = [] # list of dictionaries

    collection_size = len(df)
    n_clusters = 5
    pred_radius = np.random.choice([1,2,3,4,5,6,7,8], size=collection_size)

    data = [{"id_": art.item_id, "art_title": art.title, "url": art.url,
             "radius": pred_radius[i], "cluster": art.cluster_id}
            for i, art in df.iterrows()]

    hero_id = np.random.choice(session['art'])
    hero = data[hero_id]['url']
    hero_title = data[hero_id]['art_title']
    print data[0]

    item = dict(hero=hero, hero_id=hero_id, hero_title=hero_title, collection_size=collection_size, n_clusters=n_clusters, data=data)
    items.append(item)

    return render_template('index.html', items = items)

@app.route('/likes/<int:id>')
def likes(id):
    session['likes'].append(id)
    print session['likes']
    recommend()
    return 'itkalsjdflkajsdlf'

@app.route('/dislikes/<int:id>')
def dislikes(id):
    session['dislikes'].append(id)

@app.route('/recommend')
def recommend():
    pass




if __name__ == '__main__':
    df = pd.read_csv('data/august_3.csv', index_col=0)
    df['title'] = df['title'].fillna('none')

    gl.load_model('app/data/recommender')   # not sure why this needs the 'app'
    app.secret_key = 'Hjadfjlji1909389283'
    app.run(host='0.0.0.0', port=8080, debug=True)
