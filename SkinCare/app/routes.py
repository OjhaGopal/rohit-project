from flask import request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from app import app
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
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

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Load model once at startup ---
def _load_model():
    net = models.efficientnet_b0(weights=None)
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(1280, len(CLASSES))
    )
    weights_path = os.path.join(os.path.dirname(__file__), '..', 'skin-model-v2-best.pth')
    net.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    net.eval()
    return net

model = _load_model()

# --- Helpers ---
def _allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _predict(img):
    tensor = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        idx = probs.argmax().item()
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
