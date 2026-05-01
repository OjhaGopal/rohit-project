from flask import request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from app import app
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import logging
import tempfile

# --- Config ---
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB

CLASSES = [
    'acanthosis-nigricans', 'acne', 'acne-scars', 'alopecia-areata',
    'dry', 'melasma', 'oily', 'vitiligo', 'warts'
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- Load ONNX model once at startup ---
def _load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'skin-model.onnx')
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return session

session = _load_model()
input_name = session.get_inputs()[0].name

# --- Helpers ---
def _allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _preprocess(img):
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)        # HWC -> CHW
    return arr[np.newaxis, ...]          # add batch dim

def _predict(img):
    tensor = _preprocess(img)
    outputs = session.run(None, {input_name: tensor})[0][0]
    probs = np.exp(outputs) / np.sum(np.exp(outputs))  # softmax
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx])

# --- Routes ---
@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def home_page():
    if request.method == 'GET':
        return redirect(url_for('welcome'))

    if 'file' not in request.files:
        return render_template('result.html', res='No file uploaded. Please try again.')

    f = request.files['file']
    filename = secure_filename(f.filename)

    if not filename or not _allowed(filename):
        return render_template('result.html', res='Invalid file type. Please upload a JPG, PNG, or BMP image.')

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        img = Image.open(tmp_path).convert('RGB')
        label, confidence = _predict(img)
        res = f'{label}  ({confidence:.1%} confidence)'
    except Exception:
        logging.exception('Error during prediction')
        res = 'Could not process the image. Please try a different file.'
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return render_template('result.html', res=res)

@app.errorhandler(413)
def file_too_large(_):
    return render_template('result.html', res='File too large. Maximum size is 5 MB.'), 413
