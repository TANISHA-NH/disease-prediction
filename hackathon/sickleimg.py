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
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import load_model

def train_anemia_cnn_model(positive_img_dir, negative_img_dir, model_save_path="cnn_model.h5", epochs=20):
    # EDA to understand how to select predictive features
    sample_image = os.path.join(positive_img_dir, "1.jpg")
    image = plt.imread(sample_image)
    print(f"Sample positive image shape: {image.shape}")
    plt.imshow(image)
    plt.title("Sample Positive Image")
    plt.show()

    neg_image = plt.imread(os.path.join(negative_img_dir, "1.jpg"))
    plt.imshow(neg_image)
    plt.title("Sample Negative Image")
    plt.show()

    # Total number of images
    negative_imgs = len(os.listdir(negative_img_dir))
    positive_imgs = len(os.listdir(positive_img_dir))
    print("Total images =", (negative_imgs + positive_imgs))

    # Distribution of images
    fig, ax = plt.subplots()

    presence = ['positive images', 'negative images']
    counts = [positive_imgs, negative_imgs]
    bar_labels = ['positive images', 'negative images']
    bar_colors = ['tab:red', 'tab:blue']

    ax.bar(presence, counts, label=bar_labels, color=bar_colors)

    ax.set_ylabel('Count')
    ax.set_title('Sickle Cell Data Distribution')
    ax.legend()
    plt.show()

    # Dataset creation
    Images = []
    Labels = []

    # Append positive images
    for file_dir, _, files in os.walk(positive_img_dir):
        for file in files:
            img_file = os.path.join(file_dir, file)
            img = cv2.imread(img_file)
            img = cv2.resize(img, (255, 255))
            Images.append(img)
            Labels.append(1)

    # Append negative images
    for file_dir, _, files in os.walk(negative_img_dir):
        for file in files:
            img_file = os.path.join(file_dir, file)
            img = cv2.imread(img_file)
            img = cv2.resize(img, (255, 255))
            Images.append(img)
            Labels.append(0)

    Images = np.array(Images)
    print(f"Total dataset shape: {Images.shape}")

    # Display sample images
    plt.imshow(Images[0])
    plt.title("Positive Image Sample")
    plt.show()

    plt.imshow(Images[-1])
    plt.title("Negative Image Sample")
    plt.show()

    # Train the model
    X = Images / 255  # Scale the data
    y = Labels

    train_X, test_X, train_y, test_y = train_test_split(X, np.array(y), stratify=np.array(y), random_state=42, shuffle=True, test_size=0.2)

    print(f"Train shapes: {train_X.shape}, Test shapes: {test_X.shape}")
    print(np.unique(train_y, return_counts=True), np.unique(test_y, return_counts=True))

    # CNN training
    img_width, img_height = 255, 255
    input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()

    # Compile and train the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Add a callback to avoid overfitting
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy', save_best_only=True)

    history = model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=epochs, callbacks=[model_checkpoint])

    # Plot the training history
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # Set the y-axis limits
    plt.show()
    # If you want to get predictions or evaluate further
    # y_pred = model.predict(train_X)
    # y_pred = np.round(y_pred)
    # get_score(y_pred, train_y)
    # get_score(predictions, test_y)
    model.save("cnn_model.h5")  # Save the Keras model in .h5 format

# To load the model back later
# from tensorflow.keras.models import load_model
# model = load_model("cnn_model.h5")

# Function to load the model and make predictions on a new image
def predict_image(image_path, model_path=r"C:\Users\admin\Downloads\hackathon\hackathon\cnn_model.h5"):
    # Load the trained model
    model = load_model(model_path)
    
    # Preprocess the image for the model
    img_width, img_height = 255, 255  # Image dimensions used during training
    img = Image.open(image_path)  # Load the image
    img = img.resize((img_width, img_height))  # Resize image to match the input shape
    img = np.array(img)  # Convert image to a numpy array
    img = img / 255.0  # Normalize the image (same as training data)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 255, 255, 3)

    # Make a prediction
    prediction = model.predict(img)
    
    # Since it's binary classification, we use a threshold of 0.5
    if prediction[0] > 0.5:
        return "Positive prediction: Sickle cell detected"
    else:
        return "Negative prediction: No sickle cell detected"

# Example of taking input and predicting
image_path = input("Enter the path to the image you want to predict: ")
result = predict_image(image_path)
print(result)