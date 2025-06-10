# Diabetes Prediction Project

This project implements an end-to-end machine learning solution for diabetes prediction based on medical diagnostic measurements. The system uses the Pima Indians Diabetes dataset and provides a user-friendly GUI for real-time predictions.

## Project Overview

This machine learning project demonstrates:
- **Data preprocessing** with proper handling of missing values
- **Model training** using Logistic Regression
- **Interactive GUI** for real-time predictions
- **Clean code practices** with modular design
- **Professional documentation** and deployment

## Dataset Information

**Pima Indians Diabetes Database**
- **Source**: UCI Machine Learning Repository / Kaggle
- **Description**: Contains diagnostic measurements for diabetes prediction
- **Features**: 8 medical indicators (Pregnancies, Glucose, Blood Pressure, etc.)
- **Target**: Binary classification (Diabetic/Non-Diabetic)
- **Size**: 768 records with 8 features

## Project Structure

```
diabetes-prediction-project/
├── preprocess.py          # Data loading, cleaning, and preprocessing
├── train_model.py         # Model training and evaluation
├── gui_app.py            # Interactive GUI application
├── run_workflow.py       # Automated workflow script
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── diabetes.csv         # Dataset file
├── model.pkl           # Trained model
├── scaler.pkl          # Feature scaler
└── imputer.pkl         # Missing value imputer
```

## Features

### 1. Data Preprocessing (`preprocess.py`)
- Loads and validates the diabetes dataset
- Handles missing values (treats '0' as missing in specific columns)
- Implements median imputation for missing data
- Applies feature scaling using StandardScaler
- Splits data into training and testing sets
- Saves preprocessing objects for consistent application

### 2. Model Training (`train_model.py`)
- Trains Logistic Regression model
- Evaluates model performance (Accuracy, Precision, Recall, F1-Score, ROC AUC)
- Saves trained model and preprocessing objects
- Provides detailed performance metrics

### 3. GUI Application (`gui_app.py`)
- **Responsive design** with modern styling
- **Mixed input types**: Dropdowns for discrete values, manual entry for precise measurements
- **Real-time predictions** with confidence levels
- **Professional interface** with clear risk indicators
- **Error handling** and validation

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/diabetes-prediction-project.git
   cd diabetes-prediction-project
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Dataset**
   - Download `diabetes.csv` from [Kaggle Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
   - Place it in the project root directory

## Usage

### Quick Start (Automated)
```bash
python run_workflow.py
```
This script will automatically:
1. Preprocess the data
2. Train the model
3. Launch the GUI application

### Manual Execution
```bash
# Step 1: Preprocess data
python preprocess.py

# Step 2: Train model
python train_model.py

# Step 3: Launch GUI
python gui_app.py
```

## Model Performance

- **Algorithm**: Logistic Regression
- **Accuracy**: ~75-80%
- **Features**: 8 medical indicators
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC AUC

## GUI Features

- **Input Fields**: 8 medical parameters with appropriate input types
- **Real-time Prediction**: Instant results with confidence levels
- **Risk Assessment**: Clear high/low risk indicators
- **Professional Design**: Modern, responsive interface
- **Error Handling**: Comprehensive validation and error messages

## Technical Details

### Data Preprocessing
- **Missing Value Handling**: Treats '0' values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI as missing
- **Imputation Strategy**: Median imputation for missing values
- **Feature Scaling**: StandardScaler for numerical features
- **Data Splitting**: 80% training, 20% testing with stratification

### Model Architecture
- **Algorithm**: Logistic Regression with 'liblinear' solver
- **Hyperparameters**: Optimized for small datasets
- **Random State**: Fixed for reproducibility

### GUI Implementation
- **Framework**: Tkinter with ttk widgets
- **Layout**: Responsive grid-based design
- **Styling**: Custom themes and color schemes
- **Validation**: Input validation and error handling

## Requirements

```
pandas>=1.3.0
scikit-learn>=1.0.0
numpy>=1.21.0
joblib>=1.1.0
```

## Contributing

This is a team project demonstrating end-to-end machine learning development. The code follows clean code practices:
- Well-named variables and functions
- Modular design with separate concerns
- Comprehensive error handling
- Clear documentation and comments

## License

This project is created for educational purposes as part of a machine learning course assignment.

## Team Members

- [Your Name] - Data preprocessing and model training
- [Partner Name] - GUI development and testing

## Screenshots

Screenshot 1: https://github.com/ShkoJ/diabetes-prediction-project/blob/main/screenshot_1.png
Screenshot 2: https://github.com/ShkoJ/diabetes-prediction-project/blob/main/screenshot_2.png

## Video Presentation

Soon to be added!

## Report

https://github.com/ShkoJ/diabetes-prediction-project/blob/main/report.pdf
