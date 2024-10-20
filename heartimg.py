#importing
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from joblib import dump, load
import os
import re
import cv2
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

# Set parameters
IMAGE_SIZE = (128, 128)  # Resize images to this size
BATCH_SIZE = 42
EPOCHS = 10

# Load dataset
def load_images(image_dir):
    images = []
    labels = []
    for label in ['Normal Person', 'Myocardial Infarction Patients']:
        folder = os.path.join(image_dir, label)
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).resize(IMAGE_SIZE)
            img_array = np.array(img) / 255.0  # Normalize the image
            images.append(img_array)
            labels.append(0 if label == 'Normal Person' else 1)
    return np.array(images), np.array(labels)

# Load data
image_dir = r'C:\Users\admin\Downloads\hackathon\hackathon\heart_images'  # Path to the dataset directory
X, y = load_images(image_dir)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_save_path="heartimg.keras"
#acts like chechpoint preventing our model from not working
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy', save_best_only=True)

# Train the model
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test),callbacks=[model_checkpoint])

# Save the model
model.save('heartimg.h5')

# Function to predict new images
def predict_image(image_path):
    img = Image.open(image_path).resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return "Myocardial Infarction" if prediction[0][0] > 0.5 else "Normal"

# Example of predicting an image
input_image_path = input("Enter the image path of the ECG:")
result = predict_image(input_image_path)
print(result)
