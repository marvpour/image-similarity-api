import os
import shutil
from flask import current_app as app


def create_dir_safely(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def clean_dir_safely(dir_path: str):
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                app.logger.error(f'Failed to delete {file_path}: {e}')
