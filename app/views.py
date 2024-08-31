import os
import boto3
from botocore.exceptions import ClientError
from flask import render_template, request, redirect, send_file, send_from_directory, url_for, flash
from app import app
from scripts.inference import perform_inference, unload_models
from werkzeug.utils import secure_filename
import threading
import logging
from botocore.config import Config

logging.basicConfig(level=logging.INFO)

s3_config = Config(
    signature_version='s3v4',
    region_name=os.environ.get('AWS_REGION', 'eu-north-1')
)

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    config=s3_config
)
BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ppedetectionbucket')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

processing_status_dict = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file_to_s3(file, filename):
    try:
        s3_client.upload_fileobj(file, BUCKET_NAME, filename)
        return True
    except ClientError as e:
        logging.error(f"Error uploading to S3: {e}")
        return False

def download_file_from_s3(filename, local_path):
    try:
        with open(local_path, 'wb') as f:
            s3_client.download_fileobj(BUCKET_NAME, filename, f)
        return True
    except ClientError as e:
        logging.error(f"Error downloading from S3: {e}")
        return False

def process_file_async(input_filename, output_filename, session_id):
    local_input_path = f"/tmp/{input_filename}"
    local_output_path = f"/tmp/{output_filename}"
    
    try:
        if download_file_from_s3(input_filename, local_input_path):
            perform_inference(local_input_path, local_output_path)
            processing_status_dict[session_id] = 'completed'
        else:
            processing_status_dict[session_id] = 'error'
    except Exception as e:
        logging.error(f"Error processing {input_filename}: {e}")
        processing_status_dict[session_id] = 'error'
    finally:
        if os.path.exists(local_input_path):
            os.remove(local_input_path)
        unload_models()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if upload_file_to_s3(file, filename):
                output_filename = f"processed_{filename}"
                session_id = request.cookies.get('session_id', filename)
                processing_status_dict[session_id] = 'processing'
                threading.Thread(target=process_file_async, args=(filename, output_filename, session_id)).start()
                flash('File uploaded successfully. Processing...', 'success')
                return redirect(url_for('show_result', filename=output_filename, session_id=session_id))
            else:
                flash('Error uploading file', 'error')
                return redirect(request.url)
    return render_template('index.html')

@app.route('/result/<filename>')
def show_result(filename):
    session_id = request.args.get('session_id', filename)
    processing_status = processing_status_dict.get(session_id, 'processing')
    
    if processing_status == 'completed':
        file_path = os.path.join('/tmp', filename) 
        if os.path.exists(file_path):
            file_type = 'video' if filename.lower().endswith(('.mp4', '.avi', '.mov')) else 'image'
            return render_template('result.html', filename=filename, processing_status=processing_status, file_type=file_type)
        else:
            flash('File not found', 'error')
            return redirect(url_for('upload_file'))
    elif processing_status == 'error':
        flash('An error occurred while processing the file', 'error')
        return redirect(url_for('upload_file'))
    
    return render_template('result.html', processing_status=processing_status, filename=filename)

@app.route('/files/<filename>')
def serve_file(filename):
    return send_from_directory('/tmp', filename)