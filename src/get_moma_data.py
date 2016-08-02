import urllib2
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import sys
import socks
import socket

def get_moma_collection():
    """
    Download a CSV of the MOMA collection.
    """
    url = 'https://media.githubusercontent.com/media/MuseumofModernArt/collection/master/Artworks.csv'
    response = urllib2.urlopen(url)
    html = response.read()
    print ' -- Downloading records for the MOMA collection -- '

    with open('data/moma_collection.csv', 'wb') as f:
        f.write(html)


def download_moma_collection(csv_loc='data/moma_collection.csv',
                             location='collections/moma/',start=596, res=1):
    # read in the moma csv and filter it for 2D works of art

    df = pd.read_csv(csv_loc)
    df = df[df['Department'].isin(['Media and Performance Art',
                                   'Fluxus Collection', 'Film',
                                   'Architecture & Design - Image Archive']) == False]
    df = df.reset_index()

    # create output message for user
    output = """
              \n\r -- Downloading MOMA collection --
              \r  ----------------------------
              \r    Download Location --> {}
              \r  ----------------------------
              \r    Total Works: {}\n"""
    print output.format(location, len(df))

    # for each row in the dataframe grab the image associated with that row
    for i in xrange(start, len(df)):
        # stdout status update
        time.sleep(0)
        sys.stdout.write("\r  -- %d of %d Works -- %d %% Artwork Downloaded -- " % (i, len(df), 1.*i/len(df)*100))
        sys.stdout.flush()

        # grab the URL
        if type(df['URL'][i]) is str:
            try:
                url = df['URL'][i]
                html = requests.get(url)
                soup = BeautifulSoup(html.content, 'lxml')
                # get the image URL, the index value [res] right before the final
                # split tells the script which resolution image to take 1 ~ 640px width
                image = soup.findAll('img', attrs={'class':'sov-hero__image-container__image'})[0].get('srcset').split(',')[res].split()[0]
                image_url = 'http://www.moma.org' + image
                uopen = urllib2.urlopen(image_url)
                stream = uopen.read()

                # save the file
                fname = location + 'moma_' + str(df['ObjectID'][i]) + '.jpg'
                file = open(fname, 'w')
                file.write(stream)
                file.close()
            except IndexError:
                print 'No image URL'
    print '\n\n    Download was successful!!!\n'

if __name__ == '__main__':
    socks.setdefaultproxy(proxy_type=socks.PROXY_TYPE_SOCKS5, addr="127.0.0.1", port=9050)
    #get_moma_collection()
    download_moma_collection(start=56259)
