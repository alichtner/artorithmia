import os
import pandas as pd
import cloudinary as cloud
from PIL import Image
import urllib2 as urllib
import io

if __name__ == '__main__':
    name = os.environ['CLOUDINARY_CLOUD_NAME']
    key = os.environ['CLOUDINARY_API_KEY']
    secret = os.environ['CLOUDINARY_API_SECRET']

    cloud.config(cloud_name=name, api_key=key, api_secret=secret)
    drizl = pd.read_json('data/Artwork.json', orient='records')

    for i in xrange(len(drizl)):
        try:
            url = drizl['results'][i]['image']
            name = drizl['results'][i]['metadata']['public_id']
            fname = 'collections/drizl/' + name.replace('/', '_') + '.jpg'
            print 'Downloading: {} at {}'.format(name, url)
            fd = urllib.urlopen(url)
            image_file = io.BytesIO(fd.read())
            im = Image.open(image_file)
            im.save(fname)
        except:
            pass
