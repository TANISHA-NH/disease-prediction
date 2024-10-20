import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#this is genome test, i dont know how to keep it working without this strucutre
# Global variables to hold the encoders and scaler
label_encoders = {}
scaler = None
def process_data():
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
process_data()


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

genome_test()