import os
import boto3
from botocore.exceptions import ClientError
from flask import Response, render_template, request, redirect, send_file, send_from_directory, url_for, flash 
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

@app.route('/result/<filename>')
def show_result(filename):
    session_id = request.args.get('session_id', filename)
    processing_status = processing_status_dict.get(session_id, 'processing')
    
    logging.debug(f"Showing result for file: {filename}")
    logging.debug(f"Processing status: {processing_status}")
    
    if processing_status == 'completed':
        file_path = os.path.join('/tmp', filename)
        logging.debug(f"Checking file path: {file_path}")
        if os.path.exists(file_path):
            logging.debug(f"File exists: {file_path}")
            file_type = 'video' if filename.lower().endswith(('.mp4', '.avi', '.mov')) else 'image'
            return render_template('result.html', filename=filename, processing_status=processing_status, file_type=file_type)
        else:
            logging.error(f"File not found: {file_path}")
            flash('File not found', 'error')
            return redirect(url_for('upload_file'))
    elif processing_status == 'error':
        logging.error(f"Processing error for file: {filename}")
        flash('An error occurred while processing the file', 'error')
        return redirect(url_for('upload_file'))
    
    return render_template('result.html', processing_status=processing_status, filename=filename)

@app.route('/files/<filename>')
def serve_file(filename):
    logging.debug(f"Attempting to serve file: {filename}")
    file_path = os.path.join('/tmp', filename)
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        range_header = request.headers.get('Range')
        if range_header:
            byte1, byte2 = 0, None
            match = re.search(r'(\d+)-(\d*)', range_header)
            if match:
                groups = match.groups()
                if groups[0]:
                    byte1 = int(groups[0])
                if groups[1]:
                    byte2 = int(groups[1])
            byte2 = byte2 if byte2 else file_size - 1
            length = byte2 - byte1 + 1
            with open(file_path, 'rb') as f:
                f.seek(byte1)
                data = f.read(length)
            rv = Response(data, 206, mimetype='video/mp4',
                          content_type='video/mp4', direct_passthrough=True)
            rv.headers.add('Content-Range', f'bytes {byte1}-{byte2}/{file_size}')
            return rv
        else:
            return send_from_directory('/tmp', filename, mimetype='video/mp4')
    else:
        logging.error(f"File not found: {file_path}")
        return "File not found", 404
