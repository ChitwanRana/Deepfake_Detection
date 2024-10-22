import cv2
import numpy as np
from tensorflow.keras.models import model_from_json # type: ignore
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Dictionary to map predictions
deepfake_dict = {0: "Fake", 1: "Real"}

# Load the model architecture from JSON
json_file = open('model/Resnet_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
deepfake_model = model_from_json(loaded_model_json)

# Load weights into the model
deepfake_model.load_weights("model/Resnet_weights.weights.h5")
print("Loaded model from disk")

# Function to process the uploaded image and predict the class (Fake/Real)
def process_image(image_path):
    # Read the image
    frame = cv2.imread(image_path)
    
    # Resize the image if needed
    frame = cv2.resize(frame, (1280, 720))
    
    # Load Haar Cascade for face detection
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    
    # Convert the image to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    # Loop through the detected faces
    for (x, y, w, h) in num_faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        
        # Crop and resize the detected face region
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict whether the face is Fake or Real
        emotion_prediction = deepfake_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))  # Get the index of the maximum confidence score

        # Put the prediction (Fake/Real) on the image
        cv2.putText(frame, deepfake_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the image with the prediction
    cv2.imshow('Deepfake Detection', frame)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

# GUI for selecting an image file
def upload_image():
    Tk().withdraw()  # Hide the root Tkinter window
    file_path = askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        process_image(file_path)

# Run the image upload function
upload_image()
