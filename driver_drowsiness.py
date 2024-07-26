import cv2
import numpy as np
import tensorflow as tf
import winsound

# Load the trained model
model = tf.keras.models.load_model('drowsiness_model.h5')

# Preprocess input images
def preprocess_image(image):
    img_size = 224
    img_resized = cv2.resize(image, (img_size, img_size))  # Resize to match model's expected sizing
    img_normalized = img_resized / 255.0  # Normalize pixel values to range [0, 1]
    return img_normalized.reshape(1, img_size, img_size, 3)  # Reshape for model input

# Function to detect drowsiness
def detect_drowsiness(eye_image):
    preprocessed_image = preprocess_image(eye_image)
    prediction = model.predict(preprocessed_image)
    return prediction[0][0]  # Return the probability of eye being closed

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize variables for alarm
alarm_active = False
ALARM_DURATION = 2  # Duration of alarm sound in seconds
EYE_CLOSED_THRESHOLD = 0.5  # Threshold for eye closed probability
closed_frames = 0  # Counter for consecutive frames where eyes are closed

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale for eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected eyes and pass them to the drowsiness detection function
    for (ex, ey, ew, eh) in eyes:
        eye_image = frame[ey:ey+eh, ex:ex+ew]  # Extract the eye region from the frame
        # Perform drowsiness detection for each eye
        eye_closed_prob = detect_drowsiness(eye_image)
        print("Eye Closed Probability:", eye_closed_prob)
        
        # Draw rectangle around the detected eye
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Check if eye is closed
        if eye_closed_prob < EYE_CLOSED_THRESHOLD:
            closed_frames += 1
        else:
            closed_frames = 0
        
        # Activate alarm if eyes are closed for a long duration
        if closed_frames >= ALARM_DURATION * 30:  # Convert duration to number of frames
            if not alarm_active:
                winsound.Beep(1000, 5000)  # Play alarm sound
                alarm_active = True
        else:
            if alarm_active:
                winsound.Beep(1000, 5000)  # Play alarm sound
            alarm_active = False
    
    # Display the captured frame with detected eyes
    cv2.imshow('Webcam', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
