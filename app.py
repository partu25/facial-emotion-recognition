import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load the lightweight TFLite model instead of the massive H5 model
MODEL_PATH = 'model_optimal.tflite'
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Emotion labels (standard FER2013 alphabetical order)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        file = request.files['image'].read()
        npimg = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return jsonify({'emotion': 'No Face Detected', 'confidence': 0})
        
        # Take the first face found
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Preprocess for model
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=-1) # Add channel dimension (48, 48, 1)
        roi = np.expand_dims(roi, axis=0)  # Add batch dimension (1, 48, 48, 1)

        
        # Predict using TFLite interpreter
        interpreter.set_tensor(input_details[0]['index'], roi)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]
        
        label = EMOTIONS[preds.argmax()]
        confidence = float(preds.max())
        
        return jsonify({
            'emotion': label,
            'confidence': round(confidence * 100, 2),
            'box': [int(x), int(y), int(w), int(h)]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
