from flask import request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from app import app
import cv2
import numpy as np
from PIL import Image
import os
import logging
import tempfile

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

CLASSES = [
    'acanthosis-nigricans', 'acne', 'acne-scars', 'alopecia-areata',
    'dry', 'melasma', 'oily', 'vitiligo', 'warts'
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'skin-model.onnx')
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

net = _load_model()

def _allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _predict(img):
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    net.setInput(arr)
    out = net.forward()[0]
    probs = np.exp(out) / np.sum(np.exp(out))
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx])

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
