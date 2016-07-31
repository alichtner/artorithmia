from flask import Flask, request, render_template
from cluster_engine import ClusterArt


app = Flask(__name__)


# Home page
@app.route('/')
def index():
    items = "../static/img/bird.jpg"
    items = "../static/img/shard/" + c.artwork[0].short_name + '.jpg'
    return render_template('index.html', items = items)


@app.route('/cluster_centers', methods=['POST'])
def upload_centers():

    return "HERE ARE IMAGES!!!"

if __name__ == '__main__':
    c = ClusterArt()
    c.load_collection_from_json('app/data/Artwork.json', 'collections/shard/')
    c.run()
    app.run(host='0.0.0.0', port=8080, debug=True)
