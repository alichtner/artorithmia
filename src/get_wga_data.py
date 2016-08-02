import urllib2
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import sys



def get_wga_database():
    """
    Download a CSV of the MOMA collection.
    """
    url = 'http://www.wga.hu/database/download/data_txt.zip'
    response = urllib2.urlopen(url)
#    cat_file = response.read()
    print ' -- Downloading records for the WGA collection -- '

#    with open('data/wga_collection.csv', 'wb') as f:
#        f.write(cat_file)

    zipfile = ZipFile(StringIO(response.read()))
    with open('data/wga_collection.csv', 'wb') as f:
        f.write(zipfile)


def download_wga_collection(csv_loc='data/wga_collection.csv',location='collections/wga/', start=0, res='detail'):
    """
    res = 'detail' or 'art'
    """
    df = pd.read_csv('data/catalog.csv', delimiter=';')
    df = df[df['FORM'].isin(['painting','graphics']) == True]
    df = df.reset_index()

    # create output message for user
    output = """
              \n\r -- Downloading MOMA collection --
              \r  ----------------------------
              \r    Download Location --> {}
              \r  ----------------------------
              \r    Total Works: {}\n"""
    print output.format(location, len(df))


    for i in xrange(start, len(df)):
        time.sleep(0)
        sys.stdout.write("\r  -- %d of %d Works -- %d %% Artwork Downloaded -- " % (i, len(df), 1.*i/len(df)*100))
        sys.stdout.flush()
        # grab the URL
        if type(df['URL'][i]) is str:
            try:
                url = df['URL'][i]
                html = requests.get(url)
                soup = BeautifulSoup(html.content, 'lxml')
                s = url.split('/')[-3:]
                name = s[1] + '/' + s[2].split('.')[0]
                image_url = 'http://www.wga.hu/' + res + '/' + s[0] + '/' + name + '.jpg'
                uopen = urllib2.urlopen(image_url)
                stream = uopen.read()
                fname = location + 'wga_' + name.replace('/', '_') + '.jpg'
                with open(fname, 'w') as f:
                    f.write(stream)
            except IndexError:
                print 'No image URL'
    print '\n\n    Download was successful!!!\n'

if __name__ == '__main__':
    download_wga_collection(start=0)
