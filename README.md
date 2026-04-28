# Skin Neuro — Skin Disease Classifier

A Flask web application that uses a deep learning model (EfficientNet-B0) to classify skin conditions from uploaded or camera-captured images.

## Features

- Detect 9 skin conditions from a photo
- Upload an image from your device or use your camera directly
- Confidence score shown with every prediction
- Responsive UI that works on desktop and mobile

## Supported Conditions

| Condition | Description |
|---|---|
| Acne | Clogged hair follicles causing pimples |
| Acne Scars | Residual marks from inflamed acne |
| Acanthosis Nigricans | Dark, velvety skin patches in body folds |
| Alopecia Areata | Autoimmune hair loss condition |
| Dry Skin | Eczema and atopic dermatitis |
| Melasma | Brown/blue-gray pigmentation patches |
| Oily Skin | Excess sebum production |
| Vitiligo | Loss of skin pigment in patches |
| Warts | Viral skin infection (HPV) |

## Model

- Architecture: EfficientNet-B0 (pretrained on ImageNet, fine-tuned)
- Training set: 2,271 images across 9 classes
- Input: 224×224 RGB with ImageNet normalization
- Augmentation: random crop, flip, rotation, colour jitter
- Class imbalance handled with WeightedRandomSampler
- Optimizer: Adam + CosineAnnealingLR over 30 epochs
- Weights file: `SkinCare/skin-model-v2-best.pth`

## Project Structure

```
SkinCare-main/
├── SkinCare/
│   ├── app/
│   │   ├── __init__.py          # Flask app factory
│   │   ├── routes.py            # Routes and prediction logic
│   │   ├── static/
│   │   │   ├── css/             # Stylesheets
│   │   │   ├── js/main.js       # Frontend logic + camera
│   │   │   └── images/          # UI assets
│   │   └── templates/
│   │       ├── index.html       # Main page
│   │       └── result.html      # Prediction result page
│   ├── Datasets/                # Training images (9 classes)
│   ├── Training.ipynb           # Model training notebook
│   ├── Inference.ipynb          # Inference testing notebook
│   ├── skin-model-v2-best.pth   # Trained model weights
│   ├── run.py                   # Flask entry point
│   └── requirements.txt         # Python dependencies
```

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/SkinCare.git
cd SkinCare-main/SkinCare

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python3 run.py
```

Open `http://127.0.0.1:5000` in your browser.

## Usage

1. Navigate to the **Check Your Skin** section
2. Either **Upload Photo** (JPG, PNG, BMP — max 5 MB) or click **Use Camera** to take a live photo
3. Click **Analyse Skin**
4. The predicted condition and confidence score are shown on the result page

## Requirements

```
Flask==3.0.3
Werkzeug==3.0.3
torch==2.11.0
torchvision==0.26.0
Pillow==12.2.0
```

## Team

- Piyush Ranjan
- Ritika Kumari
- Mansi Kumari
- Loknath S Shetty
