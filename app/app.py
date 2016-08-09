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
    return recommend(json=False)

@app.route('/likes/<int:id>')
def likes(id):
    session['likes'].append(id)
    print 'likes', session['likes']
    return recommend(json=True)

@app.route('/recommend')
def recommend(json=True):
    print 'entering the recommend'
    items = [] # list of dictionaries

    collection_size = len(df)
    n_clusters = 5
    # when no likes have been given
    if len(session['likes']) == 0:
        df['radius'] = 1
        pred_radius = df['radius']
    else:
        results = rec.recommend_from_interactions(session['likes'], k=len(df)).sort('item_id', ascending=True)
        # need to append to the original weights to make sure that I always have sizes for the dots
        rec_df = results['item_id', 'score'].to_dataframe()
        merged = df.merge(rec_df, how='outer')
        merged['score'] = merged['score'].fillna(value=0)
        merged['radius'] = np.where(merged['score'] < merged['score'].mean(),
                                    merged['radius'] - .25,
                                    merged['radius'] + .5)
        merged['radius'] = np.where(merged['score'] == merged['score'].max(),
                                    merged['radius'] + 1,
                                    merged['radius'])

        # set the max and min limits for node size
        merged['radius'] = np.where(merged['radius'] < .5,
                                    .5, merged['radius'])
        merged['radius'] = np.where(merged['radius'] >= 4, 4, merged['radius'])
        pred_radius = merged['radius']
        df['radius'] = pred_radius

# build up the dictionary for each circle to represent the artwork
    print 'build dict'
    data = [{"id_": art.item_id, "art_title": art.title, "url": art.url,
             "radius": pred_radius[i], "cluster": art.cluster_id,
             "retail_price": art.retail_price, "medium": art.medium,
             "art_width": art.width, "art_height": art.height}
            for i, art in df.iterrows()]

    # initialize the starting image to show on the page
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

if __name__ == '__main__':
    df = pd.read_csv('data/drizl_raw.csv', index_col=0)
    df['title'] = df['title'].fillna('Untitled')
    df['medium'] = df['medium'].fillna('Unknown')
    df['width'] = np.round(df['width'] / 12, 1)
    df['height'] = np.round(df['height'] / 12, 1)

    rec = gl.load_model('data/recommender')   # not sure why this needs the 'app'
    app.secret_key = 'Hjadfjlji1909389283'
    app.run(host='0.0.0.0', port=8080, threaded=True)
