from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from joblib import dump, load

def anemia_predictor():
    # Load the saved Logistic Regression model
    model = load(r"C:\Users\admin\Downloads\hackathon\hackathon\bloodtestmodels\Anemia.joblib")
    
    # Prompt the user to input the required particulars
    user_input = input("Enter the particulars(Glucose, Cholesterol, Hemoglobin, Platelets, White Blood Cells, Red Blood Cells, Hematocrit, Mean Corpuscular Volume, Mean Corpuscular Hemoglobin Concentration, Insulin, BMI, Systolic Blood Pressure, Diastolic Blood Pressure, Triglycerides, HbA1c, LDL Cholesterol, HDL Cholesterol, ALT, AST, Heart Rate, Creatinine, Troponin):")
    input_values = user_input.split(",")
    
    # Convert the input values to a list of floats
    dats = [float(value.strip()) for value in input_values]
    
    # Selecting features for anemia prediction (excluding C-reactive Protein and Mean Corpuscular Hemoglobin)
    features = [dats[2], dats[5], dats[6], dats[7], dats[9], dats[3]]  # Hemoglobin, Red Blood Cells, Hematocrit, Mean Corpuscular Volume, Mean Corpuscular Hemoglobin Concentration, Platelets
    
    # Creating a DataFrame with the selected features
    dataframe = pd.DataFrame([features], columns=["Hemoglobin", "Red Blood Cells", "Hematocrit", "Mean Corpuscular Volume", "Mean Corpuscular Hemoglobin Concentration", "Platelets"])
    
    # Making predictions using the Logistic Regression model
    prediction = model.predict(dataframe)
    
    # Displaying the result based on the prediction
    if prediction == 0:
        print("Patient does not suffer from Anemia")
    else:
        print("Patient suffers from Anemia and would require further testing")



