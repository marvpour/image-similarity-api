import pandas as pd
import requests

PREFIX = 'https://www.valentino.com/'

def download_images(app):

    headers = {'User-Agent': 
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 \
            (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}


    iamge_list = pd.read_csv('/'.join([app.config['IN_DIR'], 'valentino_images.csv']))[-10:]['image'].tolist()

    for link in iamge_list:
        name = link.split(PREFIX)[1].replace('/', '____')
        r = requests.get(link, headers = headers, timeout=(20,1000))
        open(f"{app.config['IN_DIR']}/images/{name}", 'wb').write(r.content)