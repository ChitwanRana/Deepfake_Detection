from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import os

app = Flask(__name__)

# Load the model architecture from JSON
json_file = open('model/Resnet_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
deepfake_model = model_from_json(loaded_model_json)

# Load weights into the model
deepfake_model.load_weights("model/Resnet_weights.weights.h5")
print("Loaded model from disk")

# Dictionary to map predictions
deepfake_dict = {0: "Fake", 1: "Real"}

# Function to process the uploaded image or video and predict the class (Fake/Real)
def process_media(media_path):
    result = ""
    if media_path.endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(media_path)
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in num_faces:
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                emotion_prediction = deepfake_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                result = deepfake_dict[maxindex]
                break  # Only predict for the first face (if multiple faces are found)
        cap.release()

    else:
        frame = cv2.imread(media_path)
        frame = cv2.resize(frame, (1280, 720))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in num_faces:
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            emotion_prediction = deepfake_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            result = deepfake_dict[maxindex]
            break  # Only predict for the first face (if multiple faces are found)

    return result

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    # Save the file to a temporary location
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Process the file and get prediction result
    result = process_media(file_path)

    # Return the result as a JSON response
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
