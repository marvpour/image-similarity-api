import logging
import os
import time
from waitress import serve
from flask import Flask, request, jsonify, make_response, url_for, redirect
from src.config import configure_app
from src.helper_service import clean_dir_safely
from src.image_similarity.similarity import show_similar_images
from src.helper_service import create_dir_safely
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, instance_relative_config=True)

def json_response(obj, response_code=200):
    return make_response(jsonify(obj), response_code)

@app.route('/success/', methods=['GET'])
def success():
    return 'Welcome to onboarding server!'

@app.route('/get-similar-images', methods=['POST'])
def app_runner():
    input_image_url = ''
    model = 'tensorflow-hub'
    body = request.get_json()

    if body is not None:
        input_image_url = body.get('input_image_url', input_image_url)
        model_name = body.get('model', model)


    if model_name not in ['VGG-TL', 'tensorflow-hub']:
        raise(f'{model_name} model has not been implemented yet! Please choose from tensorflow-hub or VGG-TL' )
    print(model_name)

    return '\n'.join(show_similar_images(input_image_url, model_name))
    

def main():
    configure_app(app)

    # make dirs if they don't exist
    if not os.path.exists(app.config['LOGGING_DIR']):
        os.makedirs(app.config['LOGGING_DIR'])

    logging_file_name = os.path.join(app.config['LOGGING_DIR'], 'app.log')

    host = app.config['HOST']
    port = app.config['PORT']

    # enable logging
    logging.Formatter.converter = time.gmtime
    handler = logging.FileHandler(logging_file_name)
    handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))

    if app.config['DEBUG']:
        handler.setLevel(logging.DEBUG)
        app.logger.addHandler(handler)

        if app.config['RELOAD']:
            app.run(host=host, port=port)
        else:
            app.run(host=host, port=port, use_reloader=False)

    else:
        logger = logging.getLogger()
        logger.addHandler(handler)
        app.logger.setLevel(logging.INFO)

        # Using waitress in production.
        # Note that the way waitress works is it spawns a certain number of threads (by default 4) and then can only
        # handle that many requests at once. Any more requests will get queued and will run once a prior request
        # finishes.
        serve(app, host=host, port=port)
