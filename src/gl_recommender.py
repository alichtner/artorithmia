import graphlab as gl
import pandas as pd

# load in the data
df = pd.read_csv('data/august_3.csv', index_col=0)
# drop columns
df.drop(['cluster_id', 'title', 'url'], axis=1, inplace=True)
data = gl.SFrame(df)

# build the item_content recommender
rec = gl.recommender.item_content_recommender.create(data, 'item_id')
rec.save('recommender')
