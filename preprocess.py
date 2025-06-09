import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import numpy as np

def preprocess_data(file_path='diabetes.csv'):
    """
    Loads, preprocesses the Pima Indians Diabetes dataset,
    handles '0' values in specific columns as missing,
    scales numerical features, and splits it into training and testing sets.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} records from {file_path}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None, None, None, None, None

    # Features and target column
    features = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    target = 'Outcome'

    # Check if all required columns exist
    missing_columns = [col for col in features + [target] if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None, None, None, None, None, None

    X = df[features]
    y = df[target]

    # Check for any non-numeric values
    try:
        X = X.astype(float)
        y = y.astype(int)
    except ValueError as e:
        print(f"Error: Non-numeric values found in data: {e}")
        return None, None, None, None, None, None

    # Replace '0' values with NaN in specific columns where 0 is not a valid measurement
    # For Glucose, BloodPressure, SkinThickness, Insulin, BMI, 0 indicates missing data
    cols_to_impute_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    X[cols_to_impute_zeros] = X[cols_to_impute_zeros].replace(0, np.nan)

    # Impute missing values (NaNs) using the median strategy
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed_df = pd.DataFrame(X_imputed, columns=features) # Convert back to DataFrame

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed_df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y # Stratify to maintain class balance
    )

    # Save the scaler and imputer to be used in gui_app.py
    try:
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(imputer, 'imputer.pkl') # Save imputer as well for consistent preprocessing
        print("Scaler and imputer saved successfully.")
    except Exception as e:
        print(f"Warning: Could not save scaler/imputer files: {e}")

    print("Data preprocessing complete.")
    return X_train, X_test, y_train, y_test, scaler, imputer

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler_obj, imputer_obj = preprocess_data()
    if X_train is not None:
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"Scaler saved as 'scaler.pkl'")
        print(f"Imputer saved as 'imputer.pkl'")