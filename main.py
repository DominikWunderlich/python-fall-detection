import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model for fall detection
model = load_model('fall_detection_model.h5')
categories = ['fall', 'not fallen']  # Change categories to match fall detection

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and resize it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray, (128, 128)) / 255.0
    reshaped_frame = np.reshape(resized_frame, (1, 128, 128, 1))  # Model expects this shape

    # Predict if a person is falling or not fallen
    prediction = model.predict(reshaped_frame)
    status = categories[np.argmax(prediction)]  # Get the predicted label

    # Overlay the prediction on the frame
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the prediction
    cv2.imshow('Fall Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
