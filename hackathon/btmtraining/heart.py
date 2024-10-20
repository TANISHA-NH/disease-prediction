import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load


data = pd.read_csv(r"C:\Users\admin\Downloads\hackathon\hackathon\output.csv")

X = data[["Glucose", "Cholesterol", "Systolic Blood Pressure", 
           "Diastolic Blood Pressure", "Triglycerides", "HbA1c", 
           "Heart Rate", "C-reactive Protein", "BMI"]]
Y = data["Heart Disease"]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
    
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(report)
dump(model,"heart.joblib")
#inputs = input("Enter values separated by commas: ")
#dats = inputs.split(',')

#a = [float(dats[0]), float(dats[1]), float(dats[12]), float(dats[13]), 
#             float(dats[14]), float(dats[15]), float(dats[20]), float(dats[23]), 
#             float(dats[11])]

#b = pd.DataFrame([a], columns=["Glucose", "Cholesterol", "Systolic Blood Pressure", 
#                                        "Diastolic Blood Pressure", "Triglycerides", 
#                                        "HbA1c", "Heart Rate", "C-reactive Protein", "BMI"])
        
# Make prediction
#prediction = model.predict(b)
#print("\nPrediction:", prediction)