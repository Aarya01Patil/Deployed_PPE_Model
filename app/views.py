import os
import boto3
from botocore.exceptions import ClientError
from flask import render_template, request, redirect, url_for, flash, session
from app import app
from scripts.inference import perform_inference
from werkzeug.utils import secure_filename
import threading

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
)
BUCKET_NAME = 'ppedetectionbucket'  

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file_to_s3(file, filename):
    try:
        s3_client.upload_fileobj(file, BUCKET_NAME, filename)
        return True
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        return False

def download_file_from_s3(filename, local_path):
    try:
        s3_client.download_file(BUCKET_NAME, filename, local_path)
        return True
    except ClientError as e:
        print(f"Error downloading from S3: {e}")
        return False

def process_file_async(input_filename, output_filename):
    local_input_path = f"/tmp/{input_filename}"
    local_output_path = f"/tmp/{output_filename}"
    
    if download_file_from_s3(input_filename, local_input_path):
        try:
            perform_inference(local_input_path, local_output_path)
            with open(local_output_path, 'rb') as f:
                upload_file_to_s3(f, output_filename)
        except Exception as e:
            print(f"Error processing {input_filename}: {e}")
        finally:
            os.remove(local_input_path)
            os.remove(local_output_path)

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
                threading.Thread(target=process_file_async, args=(filename, output_filename)).start()
                flash('File uploaded successfully. Processing...', 'success')
                session['last_processed'] = output_filename
                return redirect(url_for('show_result', filename=output_filename))
            else:
                flash('Error uploading file', 'error')
                return redirect(request.url)
    return render_template('index.html')

@app.route('/result/<filename>')
def show_result(filename):
    if session.get('last_processed') == filename:
        session.pop('last_processed', None)
        try:
            url = s3_client.generate_presigned_url('get_object',
                                                   Params={'Bucket': BUCKET_NAME,
                                                           'Key': filename},
                                                   ExpiresIn=3600)  # URL expires in 1 hour
            return render_template('result.html', result_url=url)
        except ClientError as e:
            print(f"Error generating presigned URL: {e}")
            flash('Error retrieving processed file', 'error')
    return redirect(url_for('upload_file'))