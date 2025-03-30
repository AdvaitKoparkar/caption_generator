from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
from models import enhance_description

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Open the image using PIL
                image = Image.open(filepath)
                
                # Generate enhanced descriptions using GiT and Llama 2
                descriptions = enhance_description(image)
                
                # Clean up the uploaded file
                os.remove(filepath)
                
                return jsonify({'descriptions': descriptions})
                
            except Exception as e:
                # Clean up the uploaded file in case of error
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': str(e)}), 500
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
