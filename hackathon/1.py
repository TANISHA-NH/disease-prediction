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

from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('heartimg.h5')
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('heartimg.h5')

# Now the model is ready to use for predictions


# Set parameters
IMAGE_SIZE = (128, 128)  # Resize images to this size
BATCH_SIZE = 42
EPOCHS = 10


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