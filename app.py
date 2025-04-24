from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from rag_agent import process_query
from pdf_assistant import save_pdf_to_vector_db, temp_query, permanent_query
from deepfake_detector import detect_deepfake
app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Replace with secure key in production
CORS(app)

UPLOAD_FOLDER = 'uploads'
DB_FOLDER = 'db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
@app.route('/')
@app.route('/index')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/research')
def research():
    return send_from_directory('static', 'research-assistant.html')

@app.route('/pdf')
def pdf_research():
    return send_from_directory('static', 'pdf-research-assistant.html')

@app.route('/deepfake')
def deepfake():
    return send_from_directory('static', 'image-detector.html')

@app.route('/image-detector.html')
def serve_deepfake_html():
    return send_from_directory(app.static_folder, 'image-detector.html')
@app.route('/deepfake')
def serve_deepfake():
    return send_from_directory(app.static_folder, 'image-detector.html')
@app.route('/api/deepfake/detect', methods=['POST'])
def deepfake_detect():
    if 'image' not in request.files:
        return jsonify({'error': 'Image is required'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename, {'png', 'jpg', 'jpeg'}):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            result = detect_deepfake(filepath)
            os.remove(filepath)
            return jsonify(result)
        except Exception as e:
            os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/rag/query', methods=['POST'])
def rag_query():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    try:
        response = process_query(query)
        return jsonify({'response': response})
    except Exception as e:
        app.logger.error(f"Error in rag_query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            session['last_uploaded_pdf'] = file_path
            app.logger.info(f"PDF uploaded: {file_path}")
            return jsonify({'message': 'PDF uploaded successfully'})
        except Exception as e:
            app.logger.error(f"Error saving PDF: {str(e)}")
            return jsonify({'error': f"Failed to save PDF: {str(e)}"}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/rag/pdf-query', methods=['POST'])
def pdf_query():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    storage_type = session.get('storage_type', 'temporary')
    try:
        app.logger.info(f"Querying with storage_type: {storage_type}")
        if storage_type == 'temporary':
            response = temp_query(query)
        else:
            response = permanent_query(query)
        return jsonify({'response': response})
    except Exception as e:
        app.logger.error(f"Error in pdf_query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/storage/<store_type>', methods=['POST'])
def set_storage(store_type):
    if store_type not in ['temporary', 'permanent']:
        return jsonify({'error': 'Invalid storage type'}), 400
    last_uploaded_pdf = session.get('last_uploaded_pdf')
    if not last_uploaded_pdf or not os.path.exists(last_uploaded_pdf):
        return jsonify({'error': 'No PDF uploaded to process'}), 400
    try:
        app.logger.info(f"Saving PDF to {store_type} storage: {last_uploaded_pdf}")
        save_pdf_to_vector_db(last_uploaded_pdf, store_type)
        session['storage_type'] = store_type
        return jsonify({'message': f'Storage set to {store_type}'})
    except Exception as e:
        app.logger.error(f"Error setting storage to {store_type}: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)