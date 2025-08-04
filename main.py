from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI(
    title="CRL Ultrasound Classifier API",
    description="API for predicting ultrasound image quality and providing CRL classification.",
    version="1.0.0",
    docs_url="/docs",       
    redoc_url="/redoc",          
)

# Allow frontend (Streamlit) access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your .keras model
model = load_model("resnet50_ultrasound_model.keras")

# Class names
class_names = ['bad', 'good']

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)
    img_resized = cv2.resize(img_array, (224, 224))

    test_image = image.img_to_array(img_resized)
    test_image = np.expand_dims(test_image, axis=0)

    # Prediction logic
    result = model.predict(test_image)
    probability = tf.sigmoid(result).numpy()[0][0]
    predicted_class = "good" if probability > 0.5 else "bad"

    if predicted_class == "bad":
        probability = 1 - probability

    return {
        "prediction": predicted_class,
        "probability": float(probability),
    }
