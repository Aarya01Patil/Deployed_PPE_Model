import os
import boto3
from botocore.exceptions import ClientError
from flask import render_template, request, redirect, send_file, url_for, flash , Response
from app import app
from scripts.inference import perform_inference, unload_models
from werkzeug.utils import secure_filename
import threading
import logging
from botocore.config import Config
from io import BytesIO
from flask_cors import cross_origin

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

def generate_presigned_url(bucket_name, object_name, expiration=3600):
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        logging.error(e)
        return None
    return response

def upload_file_to_s3(file, filename):
    try:
        content_type = 'video/mp4' if filename.lower().endswith(('.mp4', '.avi', '.mov')) else 'image/jpeg'
        s3_client.upload_fileobj(
            file, 
            BUCKET_NAME, 
            filename,
            ExtraArgs={'ContentType': content_type}
        )
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
            
            # Check if the output file exists and has content
            if os.path.exists(local_output_path) and os.path.getsize(local_output_path) > 0:
                with open(local_output_path, 'rb') as f:
                    s3_client.upload_fileobj(
                        f, 
                        BUCKET_NAME, 
                        output_filename,
                        ExtraArgs={'ContentType': 'video/mp4'}
                    )
                logging.info(f"Successfully uploaded processed file {output_filename} to S3")
                processing_status_dict[session_id] = 'completed'
            else:
                logging.error(f"Processed file {local_output_path} is empty or does not exist")
                processing_status_dict[session_id] = 'error'
        else:
            processing_status_dict[session_id] = 'error'
    except Exception as e:
        logging.error(f"Error processing {input_filename}: {e}")
        processing_status_dict[session_id] = 'error'
    finally:
        if os.path.exists(local_input_path):
            os.remove(local_input_path)
        if os.path.exists(local_output_path):
            os.remove(local_output_path)
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
        file_type = 'video' if filename.lower().endswith(('.mp4', '.avi', '.mov')) else 'image'
        video_url = url_for('stream_file', filename=filename, _external=True)
        
        return render_template('result.html', video_url=video_url, processing_status=processing_status, file_type=file_type, filename=filename)
    elif processing_status == 'error':
        flash('An error occurred while processing the file', 'error')
        return redirect(url_for('upload_file'))
    
    return render_template('result.html', processing_status=processing_status, filename=filename)

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file = s3_client.get_object(Bucket=BUCKET_NAME, Key=filename)
        return send_file(
            BytesIO(file['Body'].read()),
            download_name=filename,  
            as_attachment=True,
            mimetype='video/mp4' if filename.lower().endswith(('.mp4', '.avi', '.mov')) else 'image/jpeg'
        )
    except ClientError as e:
        logging.error(f"Error downloading file {filename}: {e}")
        flash('Error downloading file', 'error')
        return redirect(url_for('upload_file'))
    
@app.route('/stream/<filename>')
@cross_origin()
def stream_file(filename):
    try:
        file_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=filename)
        file_size = int(file_obj['ContentLength'])

        range_header = request.headers.get('Range', None)
        if range_header:
            byte1, byte2 = 0, None
            match = re.search(r'(\d+)-(\d*)', range_header)
            groups = match.groups()

            if groups[0]: byte1 = int(groups[0])
            if groups[1]: byte2 = int(groups[1])

            if byte2 is None:
                byte2 = file_size - 1
            else:
                byte2 = min(byte2, file_size - 1)

            length = byte2 - byte1 + 1

            file_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=filename, Range=f'bytes={byte1}-{byte2}')

            resp = Response(
                file_obj['Body'].iter_chunks(chunk_size=8192),
                status=206,
                mimetype='video/mp4',
                content_type='video/mp4',
                direct_passthrough=True,
            )
            resp.headers.add('Content-Range', f'bytes {byte1}-{byte2}/{file_size}')
            resp.headers.add('Accept-Ranges', 'bytes')
            resp.headers.add('Content-Length', str(length))
        else:
            resp = Response(
                file_obj['Body'].iter_chunks(chunk_size=8192),
                status=200,
                mimetype='video/mp4',
                content_type='video/mp4',
                direct_passthrough=True,
            )
            resp.headers.add('Content-Length', str(file_size))

        resp.headers.add('Content-Disposition', f'inline; filename="{filename}"')
        resp.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
        resp.headers.add('Pragma', 'no-cache')
        resp.headers.add('Expires', '0')
        return resp
    except ClientError as e:
        logging.error(e)
        return Response(status=404)