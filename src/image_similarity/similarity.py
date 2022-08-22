import pandas as pd
import requests
import os
from tqdm import tqdm
import cv2
import numpy as np
from src.helper_service import create_dir_safely
from src.image_similarity.download import download_images
import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from scipy.spatial import distance
import matplotlib.pyplot as plt
from flask import current_app as app
import warnings
from src.image_similarity.download import PREFIX
warnings.filterwarnings("ignore")

metric = 'cosine'

def get_input_image(url):
    image_array = []
    headers = {'User-Agent': 
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 \
            (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

    r = requests.get(url, headers = headers)
    downloaded_image_path = '/'.join([app.config['IN_DIR'], 'downloaded_image/'])
    create_dir_safely(downloaded_image_path)
    open(downloaded_image_path + 'downloaded_image.jpg', 'wb').write(r.content)

    img = load_img(downloaded_image_path + 'downloaded_image.jpg')
    img = img_to_array(img)
    img = cv2.resize(img, (224,224))

    image_array.append(np.array(img))

    image = predict_model(np.array(image_array), is_input=True)
    return image

def read_image(file_array):
    data_path = '/'.join([app.config['IN_DIR'], 'data.npy'])

    if os.path.exists(data_path):
        app.logger.info('Data exists, loading...')
        data = np.load(data_path)

    else:
        app.logger.info('Saving data...')
        image_array = []
        for path in tqdm(file_array):
            if path != '.DS_Store':
                img = load_img('/'.join([app.config['IN_DIR'], 'images', path]))
                img = img_to_array(img)
                img = cv2.resize(img, (224,224))
                image_array.append(np.array(img))
        data = np.array(image_array)
        np.save('/'.join([app.config['IN_DIR'], 'data']), data)
    return data

def mlmodel():
    model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

    IMAGE_SHAPE = (224, 224)

    layer = hub.KerasLayer(model_url, input_shape=IMAGE_SHAPE+(3,))
    model = tf.keras.Sequential([layer])
    
    return model

def calculate_image_distance(emb, image):
    all_distances = []
    for i in range(len(emb)):
        cosineDistance = distance.cdist([image[0]], [emb[i]], metric)[0]
        all_distances.append(cosineDistance[0])
    return all_distances

def find_most_similar_images(all_distances, n, image_dir_path):
    most_ten_similar = []
    sorted_values = sorted(all_distances, reverse = False)[:n]

    most_ten_similar = [all_distances.index(i) for i in sorted_values]
    all_names = []
    for i in most_ten_similar:
        name = PREFIX + image_dir_path[i]
        all_names.append(name.replace('____', '/'))
    return all_names

def draw_simialr_images(data, most_ten_similar):
    pass
       

def predict_model(data, is_input=False):
    model = mlmodel()
    if is_input:
        return model.predict(data)
    else:
        embedding_path = '/'.join([app.config['IN_DIR'], 'embbedding'])
        if os.path.exists(embedding_path + '.npy'):
            app.logger.info('Embedding exists, loading...')
            emb = np.load(embedding_path + '.npy')
        else:
            emb = model.predict(data)
            np.save(embedding_path, emb)
        return emb

def image_similarity(image_dir_path, input_image_url):
    app.logger.info('Reading images...')
    image_dir_path = os.listdir(image_dir_path)
    data = read_image(image_dir_path)

    app.logger.info('Predicting...')
    emb = predict_model(data)
    
    app.logger.info('preparing input image...')
    input_image = get_input_image(input_image_url)
    
    app.logger.info('Calculating distance...')
    distance = calculate_image_distance(emb, input_image)
    
    app.logger.info('Finding most similar images...')
    similar_images = find_most_similar_images(distance, 10, image_dir_path)
    return similar_images

def show_similar_images(input_image_url):

    image_path = f"{app.config['IN_DIR']}/images"

    create_dir_safely(image_path)

    if not os.listdir(image_path):
        app.logger.info('Downloading images...')
        download_images(app)
    else:
        app.logger.info('Input images found. Calculating the similarity...')
        return image_similarity(image_path, input_image_url)