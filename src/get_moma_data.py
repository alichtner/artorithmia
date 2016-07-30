import urllib2

def get_moma_collection():
    """
    Download a CSV of the MOMA collection.
    """
    url = 'https://media.githubusercontent.com/media/MuseumofModernArt/collection/master/Artworks.csv'
    response = urllib2.urlopen(url)
    html = response.read()

    with open('../data/moma_collection.csv', 'wb') as f:
        f.write(html)

if __name__ == '__main__':
    get_moma_collection()
