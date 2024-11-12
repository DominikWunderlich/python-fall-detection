import cv2
import numpy as np
import os

# Paths to your dataset folders
data_dir = '~/dev/python-fall-detection/Data/images/train'
categories = ['fall', 'not fallen']
img_size = 128

# List to store processed images and labels
images = []
labels = []

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return image

for label, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            processed_img = preprocess_image(img_path)
            images.append(processed_img)
            labels.append(label) # label is the index of the category in the categories list
        except Exception as e:
            print(f'Error processing image: {img_path}: {e}')

# Convert the lists to numpy arrays
images = np.array(images).reshape(-1, img_size, img_size, 1) # add the channel dimension
labels = np.array(labels)

# Save arrays in an npz file
np.savez_compressed('preprocessed_data.npz', images=images, labels=labels)

print("Data preprocessing completed! Saved to preprocessed_data.npz.")
