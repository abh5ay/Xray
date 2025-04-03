from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "covid_classification_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = ['COVID19', 'NORMAL', 'PNEUMONIA']

def preprocess_image(image):
    img_size = (150, 150)
    image = image.resize(img_size)  # Resize
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    image = preprocess_image(image)
    
    predictions = model.predict(image)
    predicted_class = CLASS_LABELS[np.argmax(predictions)]
    
    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
