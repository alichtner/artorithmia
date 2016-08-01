import urllib2
import requests
import pandas as pd
from bs4 import BeautifulSoup

def get_moma_collection():
    """
    Download a CSV of the MOMA collection.
    """
    url = 'https://media.githubusercontent.com/media/MuseumofModernArt/collection/master/Artworks.csv'
    response = urllib2.urlopen(url)
    html = response.read()

    with open('data/moma_collection.csv', 'wb') as f:
        f.write(html)

def download_moma_collection(csv_loc='data/moma_collection.csv', 'location='collections/moma/', res=1)
    # read in the moma csv and filter it for 2D works of art
    df = pd.read_csv(csv_loc)
    df = df[df['Department'].isin(['Media and Performance Art', 'Fluxus Collection', 'Film', 'Architecture & Design - Image Archive'])==False]

    # loop over each row in the dataframe and grab the image associated with that row
    for i in xrange(len(df)):
        # stdout status update
        time.sleep(0)
        sys.stdout.write("\r-- %d %% Artwork Downloaded -- " % (1.*idx/len(df)*100))
        sys.stdout.flush()
        # grab the URL
        url = df['URL'][i]
        html = requests.get(url)
        soup = BeautifulSoup(html.content, 'lxml')
        # get the image URL, the index value [res] right before the final split tells the script which resolution image to take 1 ~ 640px width
        image = soup.findAll('img', attrs={'class':'sov-hero__image-container__image'})[0].get('srcset').split(',')[res].split()[0]
        url = 'http://www.moma.org' + image
        uopen = urllib.urlopen(url)
        stream = uopen.read()
        # save the file
        fname = location + 'moma_' + str(df['ObjectID'][i]) + '.jpg'
        file = open(fname,'w')
        file.write(stream)
        file.close()


if __name__ == '__main__':
    get_moma_collection()
    #download_moma_collection()
