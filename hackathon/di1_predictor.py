import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump, load

def diabetes_prediction():
    model = load(r"C:\Users\admin\Downloads\hackathon\hackathon\bloodtestmodels\Diabetes.joblib")

    user_input = input("Enter the particulars(Glucose,Cholesterol,Hemoglobin,Platelets,White Blood Cells,Red Blood Cells,Hematocrit,Mean Corpuscular Volume,Mean Corpuscular Hemoglobin,Mean Corpuscular Hemoglobin Concentration,Insulin,BMI,Systolic Blood Pressure,Diastolic Blood Pressure,Triglycerides,HbA1c,LDL Cholesterol,HDL Cholesterol,ALT,AST,Heart Rate,Creatinine,Troponin,C-reactive Protein)")

    input_values = user_input.split(",")

    input_values = [float(value.strip()) for value in input_values]
    a,b,c,d,e,f=input_values[0],input_values[1],input_values[15],input_values[14],input_values[16],input_values[17]
    new_data = pd.DataFrame([[a,b,c,d,e,f]], columns=["Glucose", "Cholesterol", "HbA1c", "Triglycerides", "LDL Cholesterol", "HDL Cholesterol"])

    prediction = model.predict(new_data)
    if(prediction==0):print("Not Diabetic")
    else:print("you are diabetic")


