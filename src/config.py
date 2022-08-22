import os


class BaseConfig(object):
    DEBUG = True
    # If DEBUG = True and reload to false, app will not reload on save but show logs
    # If DEBUG = False, reload will be superseded
    RELOAD = True
    TESTING = True
    HOST = 'localhost'
    PORT = '5562'
    TMP_DIR = 'tmp'
    LOGGING_DIR = 'logs'
    IN_DIR = 'in'
    OUT_DIR = 'out'

config = {
    "development": "src.config.BaseConfig"
}


def configure_app(app):
    config_name = os.getenv('FLASK_CONFIGURATION', 'development')
    app.config.from_object(config[config_name])  # object-based default configuration
    app.config.from_pyfile('config.py', silent=True)  # instance-folders configuration
