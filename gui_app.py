import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd
import os

class DiabetesPredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("Diabetes Prediction System")
        master.geometry("800x700")
        master.resizable(True, True)
        
        # Configure grid weights for responsiveness
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)
        
        # Set theme and styling
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2E86AB')
        style.configure('Subtitle.TLabel', font=('Arial', 10), foreground='#666666')
        style.configure('Header.TLabel', font=('Arial', 11, 'bold'), foreground='#2E86AB')
        style.configure('Result.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Success.TLabel', foreground='#28a745')
        style.configure('Warning.TLabel', foreground='#dc3545')
        
        # Check if required files exist
        required_files = ['model.pkl', 'scaler.pkl', 'imputer.pkl']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            error_msg = f"Missing required files: {', '.join(missing_files)}\n\nPlease run the following commands in order:\n1. python preprocess.py\n2. python train_model.py"
            messagebox.showerror("Missing Files", error_msg)
            master.destroy()
            return

        # Load the trained model, scaler, and imputer
        try:
            self.model = joblib.load('model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.imputer = joblib.load('imputer.pkl')
            print("All model files loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model files: {str(e)}\nPlease ensure you've run train_model.py first.")
            master.destroy()
            return

        self.create_widgets()

    def create_widgets(self):
        # Main container with grid weights for responsiveness
        main_frame = ttk.Frame(self.master, padding="20")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(1, weight=1)  # Input section can expand
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Title and Description
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        title_frame.grid_columnconfigure(0, weight=1)
        
        title_label = ttk.Label(title_frame, text="Diabetes Prediction System", style='Title.TLabel')
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        description_text = """This system uses machine learning to predict diabetes risk based on medical indicators. 
The model was trained on the Pima Indians Diabetes dataset and analyzes 8 key health parameters 
to provide a risk assessment with confidence levels."""
        
        desc_label = ttk.Label(title_frame, text=description_text, style='Subtitle.TLabel', 
                              wraplength=750, justify="center")
        desc_label.grid(row=1, column=0)
        
        # Input Section
        input_frame = ttk.LabelFrame(main_frame, text="Patient Health Information", padding="15")
        input_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 20))
        
        # Configure grid weights for input frame
        for i in range(4):
            input_frame.grid_columnconfigure(i, weight=1)
        
        # Define input fields with mix of dropdowns and manual entry
        self.input_fields_details = {
            'Pregnancies': {
                'label': 'Number of Pregnancies:',
                'type': 'dropdown',
                'values': list(range(0, 18)),
                'default': 1,
                'description': 'Number of times pregnant',
                'row': 0,
                'col': 0
            },
            'Glucose': {
                'label': 'Glucose Level (mg/dL):',
                'type': 'entry',
                'default': 120,
                'description': 'Plasma glucose concentration',
                'row': 0,
                'col': 1
            },
            'BloodPressure': {
                'label': 'Blood Pressure (mmHg):',
                'type': 'entry',
                'default': 70,
                'description': 'Diastolic blood pressure',
                'row': 0,
                'col': 2
            },
            'SkinThickness': {
                'label': 'Skin Thickness (mm):',
                'type': 'entry',
                'default': 20,
                'description': 'Triceps skin fold thickness',
                'row': 0,
                'col': 3
            },
            'Insulin': {
                'label': 'Insulin Level (mu U/ml):',
                'type': 'entry',
                'default': 80,
                'description': '2-Hour serum insulin',
                'row': 1,
                'col': 0
            },
            'BMI': {
                'label': 'BMI (Body Mass Index):',
                'type': 'entry',
                'default': 25.0,
                'description': 'Body mass index',
                'row': 1,
                'col': 1
            },
            'DiabetesPedigreeFunction': {
                'label': 'Diabetes Pedigree Function:',
                'type': 'entry',
                'default': 0.5,
                'description': 'Diabetes pedigree function',
                'row': 1,
                'col': 2
            },
            'Age': {
                'label': 'Age (years):',
                'type': 'dropdown',
                'values': list(range(21, 82)),
                'default': 30,
                'description': 'Age in years',
                'row': 1,
                'col': 3
            }
        }

        self.entries = {}
        
        for key, details in self.input_fields_details.items():
            # Create frame for each field
            field_frame = ttk.Frame(input_frame)
            field_frame.grid(row=details['row'], column=details['col'], padx=10, pady=10, sticky="nsew")
            field_frame.grid_columnconfigure(0, weight=1)
            
            # Label
            label = ttk.Label(field_frame, text=details['label'], style='Header.TLabel')
            label.grid(row=0, column=0, sticky="w", pady=(0, 5))
            
            # Description
            desc = ttk.Label(field_frame, text=details['description'], style='Subtitle.TLabel')
            desc.grid(row=1, column=0, sticky="w", pady=(0, 5))
            
            # Input widget
            if details['type'] == 'dropdown':
                var = tk.StringVar(value=str(details['default']))
                combo = ttk.Combobox(field_frame, textvariable=var, values=details['values'], 
                                   state="readonly", width=15)
                combo.grid(row=2, column=0, sticky="ew")
                self.entries[key] = var
            else:
                entry = ttk.Entry(field_frame, width=15)
                entry.grid(row=2, column=0, sticky="ew")
                entry.insert(0, str(details['default']))
                self.entries[key] = entry

        # Buttons Frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=20)
        
        # Prediction Button
        predict_button = ttk.Button(button_frame, text="🔍 Predict Diabetes Risk", 
                                  command=self.predict_diabetes, style='Accent.TButton')
        predict_button.pack(side="left", padx=(0, 10))
        
        # Clear button
        clear_button = ttk.Button(button_frame, text="🗑️ Clear All Fields", 
                                command=self.clear_fields)
        clear_button.pack(side="left")

        # Results Section
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="15")
        results_frame.grid(row=3, column=0, sticky="ew", pady=(0, 20))
        results_frame.grid_columnconfigure(0, weight=1)
        
        self.result_label = ttk.Label(results_frame, text="Enter patient data and click 'Predict Diabetes Risk'", 
                                     style='Result.TLabel')
        self.result_label.grid(row=0, column=0, pady=10)
        
        self.proba_label = ttk.Label(results_frame, text="", style='Subtitle.TLabel')
        self.proba_label.grid(row=1, column=0, pady=5)
        
        # Risk Level Indicator
        self.risk_label = ttk.Label(results_frame, text="", font=('Arial', 12))
        self.risk_label.grid(row=2, column=0, pady=10)
        
        # Information Section
        info_frame = ttk.LabelFrame(main_frame, text="System Information", padding="10")
        info_frame.grid(row=4, column=0, sticky="ew")
        info_frame.grid_columnconfigure(0, weight=1)
        
        info_text = """• Model: Logistic Regression trained on Pima Indians Diabetes dataset
• Features: 8 medical indicators analyzed
• Accuracy: ~75-80% (varies based on data quality)
• Note: This is a screening tool and should not replace professional medical diagnosis"""
        
        info_label = ttk.Label(info_frame, text=info_text, style='Subtitle.TLabel', 
                              wraplength=750, justify="left")
        info_label.grid(row=0, column=0)

    def clear_fields(self):
        for key, details in self.input_fields_details.items():
            if details['type'] == 'dropdown':
                self.entries[key].set(str(details['default']))
            else:
                self.entries[key].delete(0, tk.END)
                self.entries[key].insert(0, str(details['default']))
        
        self.result_label.config(text="Enter patient data and click 'Predict Diabetes Risk'")
        self.proba_label.config(text="")
        self.risk_label.config(text="")

    def predict_diabetes(self):
        try:
            raw_input = {}
            for key in self.input_fields_details.keys():
                value = self.entries[key].get()
                # Convert to float, handling potential errors
                try:
                    raw_input[key] = float(value)
                except ValueError:
                    messagebox.showerror("Invalid Input", 
                                       f"Please enter a valid number for {self.input_fields_details[key]['label']}.")
                    return

            # Create a DataFrame from raw input
            input_df = pd.DataFrame([raw_input], columns=list(self.input_fields_details.keys()))

            # Apply the SAME preprocessing steps as done during training
            # 1. Replace 0 with NaN for specific columns
            cols_to_impute_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            input_df[cols_to_impute_zeros] = input_df[cols_to_impute_zeros].replace(0, np.nan)

            # 2. Impute missing values (NaNs)
            imputed_input = self.imputer.transform(input_df)

            # 3. Scale numerical features
            scaled_input = self.scaler.transform(imputed_input)

            # Make prediction
            prediction = self.model.predict(scaled_input)[0]
            prediction_proba = self.model.predict_proba(scaled_input)[0]

            # Determine result and styling
            if prediction == 1:
                result_text = "⚠️ HIGH RISK - Diabetic"
                risk_text = "This patient shows indicators of diabetes risk"
                self.result_label.config(text=result_text, style='Warning.TLabel')
                self.risk_label.config(text=risk_text, foreground='#dc3545')
            else:
                result_text = "✅ LOW RISK - Non-Diabetic"
                risk_text = "This patient shows low diabetes risk indicators"
                self.result_label.config(text=result_text, style='Success.TLabel')
                self.risk_label.config(text=risk_text, foreground='#28a745')

            # Format probability display
            non_diabetic_prob = prediction_proba[0] * 100
            diabetic_prob = prediction_proba[1] * 100
            
            proba_text = f"Confidence: {non_diabetic_prob:.1f}% Non-Diabetic | {diabetic_prob:.1f}% Diabetic"
            self.proba_label.config(text=proba_text)

        except ValueError as ve:
            messagebox.showerror("Invalid Input", f"Error: {ve}\nPlease ensure all fields have valid numerical values.")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")
            print(f"Error details: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = DiabetesPredictorApp(root)
    root.mainloop()