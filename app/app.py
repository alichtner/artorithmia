from flask import Flask, request, render_template, session, jsonify
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
    return recommend(json=False)

@app.route('/likes/<int:id>')
def likes(id):
    session['likes'].append(id)
    print session['likes']
    return recommend(json=True)

@app.route('/dislikes/<int:id>')
def dislikes(id):
    session['dislikes'].append(id)

@app.route('/recommend')
def recommend(json=True):
    items = [] # list of dictionaries

    collection_size = len(df)
    n_clusters = 5
    if len(session['likes']) == 0:
        pred_radius = np.random.choice([4,5,6], size=len(df))
    else:
        results = rec.recommend_from_interactions(session['likes'], k=len(df))
        print results['score']
        print 'length of df', len(df)
        print 'length predictions', len(results['score'])
        pred_radius = results['score']
        print 'here i am'

# build up the dictionary for each circle to represent the artwork
    data = [{"id_": art.item_id, "art_title": art.title, "url": art.url,
             "radius": pred_radius[i], "cluster": art.cluster_id,
             "retail_price": art.retail_price, "medium": art.medium,
             "art_width": art.width, "art_height": art.height}
            for i, art in df.iterrows()]


    hero_id = np.random.choice(session['art'])
    hero = data[hero_id]['url']
    hero_title = data[hero_id]['art_title']
    hero_retail_price = data[hero_id]['retail_price']
    hero_medium = data[hero_id]['medium']
    hero_width = data[hero_id]['art_width']
    hero_height = data[hero_id]['art_height']


    item = dict(hero=hero, hero_id=hero_id, hero_title=hero_title,
                hero_retail_price=hero_retail_price, hero_medium=hero_medium,
                hero_width=hero_width, hero_height=hero_height,
                collection_size=collection_size, n_clusters=n_clusters,
                data=data)
    items.append(item)
    if json:
        return jsonify(items=items)
    else:
        return render_template('index.html', items = items)


    # if len(session['likes'])  == 0:
    #     return np.random.choice([4], size=len(df))
    #
    # else:
    #     results = rec.recommend_from_interactions(session['likes'], k=len(df))
        #return results['score']


if __name__ == '__main__':
    df = pd.read_csv('data/drizl_raw.csv', index_col=0)
    df['title'] = df['title'].fillna('Untitled')
    df['medium'] = df['medium'].fillna('Unknown')
    df['width'] = np.round(df['width'] / 12, 1)
    df['height'] = np.round(df['height'] / 12, 1)

    rec = gl.load_model('app/data/recommender')   # not sure why this needs the 'app'
    app.secret_key = 'Hjadfjlji1909389283'
    app.run(host='0.0.0.0', port=8080, debug=True)
