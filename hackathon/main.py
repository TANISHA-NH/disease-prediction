import numpy as np
import pandas as pd
import os
import re
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from joblib import dump, load

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# TensorFlow and Keras imports
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import load_model



#training of the heart ,diabetes as well as anemia using the bloodtest reports stored in the output.csv file, which is, make the models into classification models
def blood_models():
    def anemia():
        # Load the dataset
        data = pd.read_csv(r'C:\Users\admin\Downloads\hackathon\hackathon\output.csv')
        feature_columns = ["Hemoglobin","Red Blood Cells","Hematocrit","Mean Corpuscular Volume","Mean Corpuscular Hemoglobin Concentration","Platelets"]
        #features and target variable 
        X = data[feature_columns]  
        y = data['Anemia']  
        # split it (data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        #model making and training and prediction to get the accuracy
        model = LogisticRegression(random_state=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        #print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")
        cm = confusion_matrix(y_test, y_pred)
        #print("Confusion Matrix:",cm)

        # Save the model
        dump(model, 'Anemia.joblib')

    def diabetes():
        #load dataset
        data = pd.read_csv(r"C:\Users\admin\Downloads\hackathon\hackathon\output.csv")
        #take the features and target variable
        x = data[["Glucose", "Cholesterol", "HbA1c", "Triglycerides", "LDL Cholesterol", "HDL Cholesterol"]]
        y = data[["Diabetes"]]
        #model making ,data splitting and model training 
        model = LogisticRegression()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)
        model.fit(x_train, y_train)
        #data prediction
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1]  # Probability of the positive class
        print(y_pred_proba)
        #report of its
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        #print(f'Accuracy: {accuracy:.2f}')
        #print('Confusion Matrix:\n', conf_matrix)
        #print('Classification Report:\n', class_report)
        #print('Predicted Probabilities:\n', y_pred_proba)
        dump(model, "Diabetes.joblib")

    def heart():
        #data loading
        data = pd.read_csv(r"C:\Users\admin\Downloads\hackathon\hackathon\output.csv")
        #feaures and target variable
        X = data[["Glucose", "Cholesterol", "Systolic Blood Pressure", 
                   "Diastolic Blood Pressure", "Triglycerides", "HbA1c", 
                   "Heart Rate", "C-reactive Protein", "BMI"]]
        Y = data["Heart Disease"]
        #data splitting and model training 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        #model prediction
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        #print(f"Accuracy: {accuracy:.2f}")
        #print("Confusion Matrix:",confusion)
        #print("Classification Report:",report)
        dump(model, "heart.joblib")
    # calling the subfucntion to get the trained models
    anemia()
    diabetes()
    heart()

#calls all the funcitions and allows it to train every blood test model 
#blood_models()

#training for the imaging models 
#they will be converted into a single funciton with multiple subfunctions ,so hope it works :) (im dying from the lack of sleep)
def image_models():
    #this is for heart attack / myocardial infarction
    def heart_training():
        #reuqire d parameters for training
        IMAGE_SIZE = (128, 128)  # Resize images 
        BATCH_SIZE = 42
        EPOCHS = 10
        #custom funcitons to check if the image is positive or negative b4 adding it to the dataframe used for prediciton
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
        image_dir = r'C:\Users\admin\Downloads\hackathon\hackathon\heart_images'
        X, y = load_images(image_dir)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        # Build the convolutional neural network (CNN)
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
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model_save_path = "heartimg.keras"
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy', save_best_only=True)
        # Train the model
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test), callbacks=[model_checkpoint])
        # Save the trained model
        model.save('heartimg.h5')
    #this is for anemia ,sickle cell anemia in particular 
    def anemia_training():
        def train_anemia_cnn_model(positive_img_dir, negative_img_dir, model_save_path="cnn_model.h5", epochs=20):
            # Create dataset
            Images = []
            Labels = []
            # Load positive images
            for file_dir, _, files in os.walk(positive_img_dir):
                for file in files:
                    img_file = os.path.join(file_dir, file)
                    img = cv2.imread(img_file)
                    img = cv2.resize(img, (255, 255))
                    Images.append(img)
                    Labels.append(1)
            # Load negative images
            for file_dir, _, files in os.walk(negative_img_dir):
                for file in files:
                    img_file = os.path.join(file_dir, file)
                    img = cv2.imread(img_file)
                    img = cv2.resize(img, (255, 255))
                    Images.append(img)
                    Labels.append(0)

            Images = np.array(Images)
           # print(f"Total dataset shape: {Images.shape}")

            # Prepare data for training
            X = Images / 255  
            y = Labels
            #train the model
            train_X, test_X, train_y, test_y = train_test_split(X, np.array(y), stratify=np.array(y), random_state=42, shuffle=True, test_size=0.2)
            #print(f"Train shapes: {train_X.shape}, Test shapes: {test_X.shape}")

            # CNN model creation
            model = Sequential()
            model.add(Conv2D(32, (2, 2), input_shape=(255, 255, 3)))
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
            #understand the model
            #model.summary()

            # Compile and train the model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy', save_best_only=True)
            history = model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=epochs, callbacks=[model_checkpoint])

            # Plot training history
            pd.DataFrame(history.history).plot(figsize=(8, 5))
            plt.grid(True)
            plt.gca().set_ylim(0, 1)  # Set the y-axis limits
            #plt.show()

            model.save("cnn_model.h5")  # Save the Keras model in .h5 format

        #train it
        positive_img_dir = r'C:\Users\admin\Downloads\hackathon\hackathon\positive_images'
        negative_img_dir = r'C:\Users\admin\Downloads\hackathon\hackathon\negative_images'
        train_anemia_cnn_model(positive_img_dir, negative_img_dir)
    #train both models to be ready for use
    heart_training()
    anemia_training()

#train the models by executing main function
#image_models()


#genome model training
#this is genome test
# Global variables to hold the encoders and scaler
label_encoders = {}
scaler = None
def genome_model():
    global label_encoders, scaler
    # Load
    df1 = pd.read_csv(r'C:\Users\admin\Downloads\hackathon\hackathon\PheGenI.csv')
    # feture extrraction
    categorical_columns = ['SNP rs', 'Context', 'Gene', 'Chromosome']
    numerical_columns = ['Location', 'P-Value']
    # Drop waste columns
    df1 = df1.drop(['No.', 'Source', 'Study Name', 'Analysis ID', 'PubMed', 'Gene ID', 'Study ID', 'Gene 2', 'Gene ID 2'], axis=1)
    
    # Convert string to integers/numbers
    for col in categorical_columns:
        le = LabelEncoder()
        df1[col] = le.fit_transform(df1[col])
        label_encoders[col] = le

    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df1[categorical_columns] = imputer.fit_transform(df1[categorical_columns])
    df1[numerical_columns] = imputer.fit_transform(df1[numerical_columns])

    # Define keywords to filter by required disease/trait
    keywords = ["heart", "anemia", "diabetes"]

    # Function to check if any keyword is in the unique value
    def filter_values(values, keywords):
        return [value for value in values if any(keyword.lower() in value.lower() for keyword in keywords)]

    # Get the filtered values
    unique_values = df1['Trait'].unique()
    filtered_values = filter_values(unique_values, keywords)

    # Get only the rows from the original dataset where 'Trait' matches the filtered unique values
    filtered_rows = df1[df1['Trait'].isin(filtered_values)]
    filtered_rows = filtered_rows[filtered_rows['Trait'] != 'Heart Function Tests']

    # Dropping rows with any remaining NaN values (if any)
    df = filtered_rows.dropna()

    # Define features and target variable
    X = df.drop(columns=['Trait'])
    y = df['Trait']

    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=1)

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the model
    dump(model, r'C:\Users\admin\Downloads\hackathon\hackathon\GenomePredictor.pkl')
#this below fucntion will train the model
#genome_model()

#TRAINING ALL MODELS BEFORE PREEDCTION
def trainer():
    blood_models()
    image_models()
    genome_model()
#below function will cause all the models to be trained
#trainer()

def choice():
    print("Would you like to proceed with further testing?")
    a = input("[Y]es or [N]o: ").strip().lower() 
    return a[0]

#prediction function for genome testing
#the same one can be reused for every disease
def genome_test():
    a = input("Enter the values ('SNP rs', 'Context', 'Gene', 'Chromosome', 'Location', 'P-Value') separated by '|': ")
    ch = a.split("|")
    model = load(r"C:\Users\admin\Downloads\hackathon\hackathon\GenomePredictor.pkl")

    # Ensure the input_data is a DataFrame
    input_df = pd.DataFrame([ch], columns=['SNP rs', 'Context', 'Gene', 'Chromosome', 'Location', 'P-Value'])
    
    categorical_columns = ['SNP rs', 'Context', 'Gene', 'Chromosome']
    numerical_columns = ['Location', 'P-Value']
    
    # Encode categorical columns using the fitted LabelEncoders
    for col in categorical_columns:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Standardize the input features using the fitted scaler
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Get prediction probabilities
    probabilities = model.predict_proba(input_scaled)
    
    # Get the index of the predicted class
    predicted_index = model.classes_.tolist().index(prediction[0])
    
    # Get the probability of the predicted class
    predicted_probability = probabilities[0][predicted_index]
    print(f"Predicted Disease Trait: {prediction[0]}")
    print(f"Probability: {predicted_probability}")

# Prediction functions for imaging models
#Prediction fucntions for anemia in particular
def predict_heart_image():
    image_path = input("Enter the image path of the ECG:")
    model = keras.models.load_model('heartimg.h5')  # Load the trained heart model
    img = Image.open(image_path).resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.6:
        print("The prediction of the imaging shows that you suffer from Myocardial Infarction.")
        print("We recommend further genetic testing to evaluate the potential inheritance of identified conditions to your children.")
        ch=choice()
        if ch == 'y':genome_test()
    else:
        print("The prediction of the imaging shows that you have a healthy heart.")
        
        
#Prediction fucntions for anemia in particular
def predict_anemia_image():
    image_path = input("Enter the path of the microspoic blood image:")
    model = keras.models.load_model("cnn_model.h5")  # Load the trained anemia model
    img = Image.open(image_path).resize((255, 255))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    if prediction[0] > 0.6:
        print("The prediction of the imaging shows that Sickle Cell is detected.")
        print("We recommend further genetic testing to evaluate the potential inheritance of identified conditions to your children.")
        ch=choice()
        if ch == 'y':genome_test()
    else:
        print("The prediction of the imaging shows that there is no Sickle Cell detected.")



#this is the functions to use the blood models and predict if they have that particular disease or not
#this is particularly for heart attack/myocardial infarction
def heart_prediction():
    #load
    model=load(r"C:\Users\admin\Downloads\hackathon\hackathon\bloodtestmodels\heart.joblib")
    #input for fellow
    user_input = input("Enter the Blood Test Results")
    #make data useable
    input_values = user_input.split(",")
    dats = [float(value.strip()) for value in input_values]
    a = [float(dats[0]), float(dats[1]), float(dats[12]), float(dats[13]), 
              float(dats[14]), float(dats[15]), float(dats[20]), float(dats[23]),float(dats[11])]
    b = pd.DataFrame([a], columns=["Glucose", "Cholesterol", "Systolic Blood Pressure","Diastolic Blood Pressure", "Triglycerides","HbA1c", "Heart Rate", "C-reactive Protein", "BMI"])  
    # Make prediction
    prediction = model.predict(b)
    if prediction==0:
        print("The prediction of the blood tests shows that the patient does not suffer from any heart conditions.")
    else:
        print("The prediction of the blood tests shows that the patient suffers from heart conditions and requires imaging.")
        ch=choice()
        if ch == 'y':predict_heart_image()

#this is particularly  diabetes prediction
def diabetes_prediction():
    #load
    model = load(r"C:\Users\admin\Downloads\hackathon\hackathon\bloodtestmodels\Diabetes.joblib")
    #input
    user_input = input("Enter the Blood Test Results")
    #make input predictable
    input_values = user_input.split(",")
    input_values = [float(value.strip()) for value in input_values]
    a,b,c,d,e,f=input_values[0],input_values[1],input_values[15],input_values[14],input_values[16],input_values[17]
    new_data = pd.DataFrame([[a,b,c,d,e,f]], columns=["Glucose", "Cholesterol", "HbA1c", "Triglycerides", "LDL Cholesterol", "HDL Cholesterol"])
    #prediction taking place 
    prediction = model.predict(new_data)
    if(prediction==0):
        print("The prediction of the blood tests shows that the patient does not suffer from diabetes.")
    else:
        print("The prediction of the blood tests shows that the patient suffers from diabetes")
        print("We recommend further genetic testing to evaluate the potential inheritance of identified conditions to your children.")
        ch=choice()
        if ch == 'y':genome_test()

#this function is particurlarly for anemia prediction 
def anemia_predictor():
    # Load
    model = load(r"C:\Users\admin\Downloads\hackathon\hackathon\bloodtestmodels\Anemia.joblib")
    # input
    user_input = input("Enter the Blood Test Results")   
    # convert data into a useful form
    input_values = user_input.split(",")
    dats = [float(value.strip()) for value in input_values]
    features = [dats[2], dats[5], dats[6], dats[7], dats[9], dats[3]]
    dataframe = pd.DataFrame([features], columns=["Hemoglobin", "Red Blood Cells", "Hematocrit", "Mean Corpuscular Volume", "Mean Corpuscular Hemoglobin Concentration", "Platelets"])
    # Making predictions using the Logistic Regression model
    prediction = model.predict(dataframe)    
    if prediction == 0:
        print("The prediction of the blood tests shows that the patient does not suffer from anemia.")
    else:
        print("The prediction of the blood tests shows that the patient suffers from anemia and requires imaging.")
        ch=choice()
        if ch == 'y':predict_anemia_image()

#main function
def main():
    # Greet the user
    print("Welcome to WhiteHat Hospital!")

    #options to the user
    print("Please select any one of the curated test we have:")
    print("1. Diabetes")
    print("2. Heart Diseases")
    print("3. Anemia")
    choice = input("Enter the number corresponding to your choice (1-3): ")

    if choice == '1':
        print("You selected Diabetes Blood Test.")
        diabetes_prediction()  

    elif choice == '2':
        print("You selected Heart Diseases Blood Test.")
        heart_prediction()  

    elif choice == '3':
        print("You selected Anemia Blood Test.")
        anemia_predictor()
        
        
    else:
        # Handle invalid input
        print("Invalid choice. Please select a valid option.")


main()