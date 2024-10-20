import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump,load
data=pd.read_csv(r"C:\Users\admin\Downloads\hackathon\hackathon\output.csv")
x=data[["Glucose","Cholesterol","HbA1c","Triglycerides","LDL Cholesterol","HDL Cholesterol"]]
y=data[["Diabetes"]]
model=LogisticRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)
model.fit(x_train,y_train)


# Step 6: Make Predictions
y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:, 1]  # Probability of the positive class
print(y_pred_proba)

# Step 7: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)
print('Predicted Probabilities:\n', y_pred_proba)
dump(model,"Diabetes.joblib")