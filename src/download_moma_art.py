
import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib

# read in the


df = pd.read_csv('../data/moma_collection.csv')
df = df[df['Department'].isin(['Media and Performance Art', 'Fluxus Collection', 'Film', 'Architecture & Design - Image Archive'])==False]

for i in xrange(0,10):
    time.sleep(0)
    sys.stdout.write("\r-- %d of %d Works -- %% Artwork Downloaded -- " % (i, len(df), 1.*i/len(df)*100))
    sys.stdout.flush()
    url = df['URL'][i]
    fname = 'collections/moma/moma_' + str(df['ObjectID'][i]) + '.jpg'
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'lxml')
    try:
        image = soup.findAll('img', attrs={'class':'sov-hero__image-container__image'})[0].get('srcset').split(',')[1].split()[0]
        print image
        url = 'http://www.moma.org' + image
        uopen = urllib.urlopen(url)
        stream = uopen.read()
        file = open(fname,'w')
        file.write(stream)
        file.close()
    except IndexError:
        print 'No image URL'
