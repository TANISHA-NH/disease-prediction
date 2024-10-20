from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from joblib import dump, load

# Load the dataset
data = pd.read_csv(r'C:\Users\admin\Downloads\hackathon\hackathon\output.csv')

feature_columns = [
    "Hemoglobin",
    "Red Blood Cells",
    "Hematocrit",
    "Mean Corpuscular Volume",
    "Mean Corpuscular Hemoglobin Concentration",
    "Platelets"
]

X = data[feature_columns]  
y = data['Anemia']  

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(random_state=5)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Save the model
dump(model, 'Anemia.joblib')
print("Model saved as Anemia.joblib")
