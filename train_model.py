import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load preprocessed data
data = np.load('preprocessed_data.npz') # load the preprocessed data
images = data['images']
labels = data['labels']

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.15, random_state=42)


def create_model():
    # build the model. We will use a simple CNN model with 2 Convolutional layers, 2 MaxPooling layers,
    # 1 Flatten layer, 1 Dense layer, and 1 Dropout layer
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax') # Two output classes: fall and not fallen
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

datagen = ImageDataGenerator(rescale=1./255)

model = create_model() # create the model
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)
model.save('fall_detection_model.h5') # save the model
print("Model training complete. Model saved as 'fall_detection_model.h5'.")