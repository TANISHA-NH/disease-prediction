import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load

def heart_prediction():
    model=load(r"C:\Users\admin\Downloads\hackathon\hackathon\bloodtestmodels\heart.joblib")
    user_input = input("Enter the particulars(Glucose,Cholesterol,Hemoglobin,Platelets,White Blood Cells,Red Blood Cells,Hematocrit,Mean Corpuscular Volume,Mean Corpuscular Hemoglobin,Mean Corpuscular Hemoglobin Concentration,Insulin,BMI,Systolic Blood Pressure,Diastolic Blood Pressure,Triglycerides,HbA1c,LDL Cholesterol,HDL Cholesterol,ALT,AST,Heart Rate,Creatinine,Troponin,C-reactive Protein)")
    input_values = user_input.split(",")
    dats = [float(value.strip()) for value in input_values]

    a = [float(dats[0]), float(dats[1]), float(dats[12]), float(dats[13]), 
              float(dats[14]), float(dats[15]), float(dats[20]), float(dats[23]),float(dats[11])]

    b = pd.DataFrame([a], columns=["Glucose", "Cholesterol", "Systolic Blood Pressure", 
                                        "Diastolic Blood Pressure", "Triglycerides", 
                                        "HbA1c", "Heart Rate", "C-reactive Protein", "BMI"])
        
# Make prediction
    prediction = model.predict(b)
    if prediction==0:print("Patient does not suffer from any heart conditions")
    else:
        print("Patient suffers from Heart Conditions and would require imaging")
        
