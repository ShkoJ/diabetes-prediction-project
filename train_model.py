import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from preprocess import preprocess_data # Import the preprocessing function

def train_and_save_model():
    """
    Trains a Logistic Regression model on the preprocessed Pima Indians Diabetes data and saves it.
    """
    print("Starting model training...")

    # Get preprocessed data from preprocess.py
    # We don't need the scaler or imputer objects here, just the processed data
    result = preprocess_data('diabetes.csv')
    
    if result[0] is None:  # Check if preprocessing failed
        print("Model training aborted due to data preprocessing issues.")
        return
    
    X_train, X_test, y_train, y_test, _, _ = result

    # Validate the data
    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: No training or test data available.")
        return

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Initialize and train the Logistic Regression model
    model = LogisticRegression(solver='liblinear', random_state=42) # 'liblinear' is good for small datasets
    
    try:
        model.fit(X_train, y_train)
        print("Model training completed successfully.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    # Predict on the test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of positive class

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Save the trained model
    try:
        model_filename = 'model.pkl'
        joblib.dump(model, model_filename)
        print(f"\nModel saved successfully as '{model_filename}'")
        print("You can now run gui_app.py to use the model!")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == '__main__':
    train_and_save_model()