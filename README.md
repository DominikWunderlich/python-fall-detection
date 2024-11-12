# Fall Detection System

This project is a real-time fall detection system that uses computer vision and deep learning techniques to determine whether a person is falling or not. It uses a Convolutional Neural Network (CNN) model trained on images to classify the posture of a person.

## Key Features
- **Real-time Detection**: Uses a webcam to capture video and detect whether a person is falling or not.
- **Deep Learning Model**: A CNN model trained with images to classify two categories: "fall" and "not fallen."
- **Live Prediction**: Displays the prediction on the video feed in real-time.

## Requirements
- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- scikit-learn

## Setup

1. **Install Dependencies**:
   Create a virtual environment and install the required libraries:
   ```bash
   pip install -r requirements.txt

2. **Run Application**:
    ```bash
    python main.py

2. **Model**:
The model used for classification is trained using images of people in different postures (falling and not fallen). The model is saved as fall_detection_model.h5 and is loaded during runtime.

