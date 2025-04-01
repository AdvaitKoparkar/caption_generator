from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
import requests
from functools import lru_cache
import hashlib
import torch
import logging
from config_loader import config
from caption_models import db, update_caption_selection, CaptionGeneration, Caption

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///captions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create database tables
with app.app_context():
    db.create_all()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_hash(file_path):
    """
    Generate a hash for the uploaded file to use as cache key.
    """
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

@lru_cache(maxsize=100)
def process_image(file_hash, user_suggestion):
    """
    Process image and generate descriptions with caching.
    """
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_hash}.jpg")
    try:
        # Make request to FastAPI service
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'suggestion': user_suggestion} if user_suggestion else {}
            git_api_url = config.get_git_api_url()
            logger.info(f"=== Starting request to GiT API ===")
            logger.info(f"API URL: {git_api_url}")
            logger.info(f"User suggestion: {user_suggestion}")
            
            response = requests.post(
                f"{git_api_url}/generate",
                files=files,
                data=data
            )
            logger.info(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"Error response from GiT API: {response.text}")
                raise requests.exceptions.RequestException(f"GiT API returned status code {response.status_code}")
                
            response_data = response.json()
            logger.info(f"Generated captions: {response_data['captions']}")
            logger.info("=== GiT API request complete ===")
            return response_data['captions'], response_data['image_data']
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making request to GiT API: {e}")
        raise
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and generate captions."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save file temporarily
            file_hash = hashlib.md5(file.read()).hexdigest()
            file.seek(0)  # Reset file pointer
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_hash}.jpg")
            file.save(file_path)
            
            # Get user suggestion if provided
            user_suggestion = request.form.get('suggestion', '')
            
            # Process image and get captions
            captions_txt, image_data = process_image(file_hash, user_suggestion)
            
            # Save captions to database
            caption_generation = CaptionGeneration(
                algorithm=config.algorithm,
            )
            db.session.add(caption_generation)
            db.session.commit()
            
            captions = []
            for caption_text in captions_txt:
                caption = Caption(
                    text=caption_text,
                    image_hash=file_hash,
                    user_suggestion=user_suggestion,
                    generation_id=caption_generation.id
                )
                captions.append(caption)
                db.session.add(caption)

            db.session.commit()
            
            return jsonify({
                'captions': [{'id': c.id, 'text': c.text} for c in captions],
                'image_data': image_data
            })
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/select-caption', methods=['POST'])
def select_caption():
    """
    Handle caption selection.
    """
    data = request.get_json()
    if not data or 'caption_id' not in data:
        return jsonify({'error': 'No caption ID provided'}), 400
    
    try:
        update_caption_selection(data['caption_id'])
        logger.info(f"Updated caption selection for ID: {data['caption_id']}")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error updating caption selection: {e}")
        return jsonify({'error': str(e)}), 500

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 5 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=300'
    return response

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
