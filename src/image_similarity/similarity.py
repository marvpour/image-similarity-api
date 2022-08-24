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
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from keras.models import load_model
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
from scipy.spatial import distance
import matplotlib.pyplot as plt
from flask import current_app as app
import warnings
from src.image_similarity.download import PREFIX
warnings.filterwarnings("ignore")

metric = 'cosine'

def get_input_image(url, model):
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

    image = model.predict(np.array(image_array))
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

def tf_hub_model():
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
    most_n_similar = []
    sorted_values = sorted(all_distances, reverse = False)[:n]

    most_n_similar = [all_distances.index(i) for i in sorted_values]

    all_names = []
    for i in most_n_similar:
        name = PREFIX + image_dir_path[i]
        all_names.append(name.replace('____', '/'))
    return all_names

def split_data(X,y):
    return train_test_split(X, y, test_size=0.33, random_state=42)

def remove_invalid_images(image_dir_path):
    file_path = os.listdir(image_dir_path)
    try:
        for filename in file_path:
            path = image_dir_path + f"/{filename}"
            img=Image.open(path)
            img.verify()
        return False
    except:
        app.logger.info(f'{path} is an invalid image. removing the file...')
        os.remove(path)
        return True

def get_labels(path):
    labels = [label[:2].replace('va', '10') for label in path]

    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    labels = to_categorical(labels, num_classes=len(set(labels)))
    return labels


def vgg_model(data, labels):
    
    ## Loading VGG16 base model
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=data[0].shape)
    base_model.trainable = False ## Not trainable weights


    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(150, activation='relu')
    dense_layer_2 = layers.Dense(70, activation='relu')
    prediction_layer = layers.Dense(25, activation='softmax')


    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])

    model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )


    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)

    model.fit(data, labels, epochs=2, validation_split=0.2, batch_size=256, callbacks=[es])

    model.save(f"{app.config['IN_DIR']}/model_saved.h5")

def predict_model(data, labels, method='tensorflow-hub'):

    if method == 'VGG-TL':
        model_path = '/'.join([app.config['IN_DIR'], 'model.h5'])

        if os.path.exists(model_path):
            app.logger.info('VGG model exists, loading...')
            model = load_model(model_path)

        else:
            app.logger.info('Transfer learning VGG model...')
            model = vgg_model(data, labels)

    elif method == 'tensorflow-hub':
        app.logger.info('Loading tensorflow-hub model...')
        model = tf_hub_model()


    embedding_path = '/'.join([app.config['IN_DIR'], 'embbedding'])
    if os.path.exists(embedding_path + '.npy'):
        app.logger.info('Embedding exists, loading...')
        emb = np.load(embedding_path + '.npy')
    else:
        app.logger.info('Predicting the data...')
        emb = model.predict(data)
        np.save(embedding_path, emb)
    return model, emb

def image_similarity(image_dir_path, input_image_url, model_name):
    app.logger.info('Validating downloaded images...')
    is_valid = True
    while is_valid:
        is_valid = remove_invalid_images(image_dir_path)
    valid_images_path = os.listdir(image_dir_path)

    app.logger.info('Reading images...')
    data = read_image(valid_images_path)

    app.logger.info('Defining labels...')
    labels = get_labels(valid_images_path)

    app.logger.info('Vectorizing images using ML...')
    model, emb = predict_model(data, labels, method=model_name)
    
    app.logger.info('preparing input image...')
    input_image = get_input_image(input_image_url, model)
    
    app.logger.info('Calculating distance...')
    distance = calculate_image_distance(emb, input_image)
    
    app.logger.info('Finding most similar images...')
    similar_images = find_most_similar_images(distance, 10, valid_images_path)
    return similar_images

def show_similar_images(input_image_url, model_name):

    image_path = f"{app.config['IN_DIR']}/images"

    create_dir_safely(image_path)

    if not os.listdir(image_path):
        app.logger.info('Downloading images...')
        download_images(app)
    else:
        app.logger.info('Input images found. Calculating the similarity...')
        return image_similarity(image_path, input_image_url, model_name)