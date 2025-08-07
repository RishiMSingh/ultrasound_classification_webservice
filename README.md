# ultrasound_classification_webservice

## CRL Ultrasound Classification API

A FastAPI-based backend for predicting **Crown-Rump Length (CRL) ultrasound image quality** using a pre-trained deep learning model. This API receives an image and returns a predicted label (`good` or `bad`) along with a probability score.

---

## Features

- Fast inference with a `.keras` model
- Accepts JPEG/PNG uploads via HTTP POST
- Returns JSON with prediction and probability
- Easily deployable on Render, EC2, Docker, or locally

---

## Model Info

- Model: ResNet50-based CNN
- Format: `.keras` (Keras model file)
- Input Shape: `(224, 224, 3)`
- Output: Binary class (`good`, `bad`)

---

## Project Structure

ultrasound-api/
├── main.py # FastAPI application
├── resnet50_ultrasound_model.keras # Pretrained ultrasound model
├── requirements.txt # Python dependencies
└── README.md


---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ultrasound-api.git
cd ultrasound-api

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt



