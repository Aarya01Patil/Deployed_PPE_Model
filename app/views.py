import os
from flask import render_template, request, redirect, url_for, send_from_directory, flash , session
from app import app
from scripts.inference import perform_inference
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static', 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"processed_{filename}")
            file.save(input_path)
            
            try:
                perform_inference(input_path, output_path)
                flash('File processed successfully', 'success')
                session['last_processed'] = f"processed_{filename}"  
                return redirect(url_for('show_result', filename=f"processed_{filename}"))
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(url_for('upload_file'))
    return render_template('index.html')

@app.route('/result/<filename>')
def show_result(filename):
    if session.get('last_processed') == filename:
        session.pop('last_processed', None)
        return render_template('result.html', result_file=filename)
    return redirect(url_for('upload_file'))

@app.route('/static/results/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)