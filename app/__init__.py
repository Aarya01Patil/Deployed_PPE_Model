import os
import shutil
import secrets
from flask import Flask
from flask_cors import CORS

app = Flask(__name__, static_folder='../static', template_folder='templates')
app.secret_key = secrets.token_hex(16)
CORS(app, resources={r"/*": {"origins": "*", "expose_headers": "ETag", "methods": ["GET", "HEAD", "OPTIONS"], "max_age": 3000}})

def clear_folder(folder_name):
    folder_path = os.path.join(app.static_folder, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

for folder in ['results', 'uploads']:
    clear_folder(folder)

from app import views