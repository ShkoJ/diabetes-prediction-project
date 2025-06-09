#!/usr/bin/env python3
"""
Workflow script to run the complete diabetes prediction pipeline.
This script will:
1. Preprocess the data
2. Train the model
3. Launch the GUI application
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Step: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("SUCCESS!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("ERROR!")
        print(f"Command failed with return code {e.returncode}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error:")
            print(e.stderr)
        return False

def main():
    print("Diabetes Prediction Model - Complete Workflow")
    print("This script will run the entire pipeline for you.")
    
    # Check if diabetes.csv exists
    if not os.path.exists('diabetes.csv'):
        print("ERROR: diabetes.csv not found in current directory!")
        print("Please ensure you have the diabetes.csv file in the same directory as this script.")
        return
    
    # Step 1: Preprocess data
    if not run_command("python preprocess.py", "Data Preprocessing"):
        print("Preprocessing failed. Please check the error messages above.")
        return
    
    # Step 2: Train model
    if not run_command("python train_model.py", "Model Training"):
        print("Model training failed. Please check the error messages above.")
        return
    
    # Step 3: Check if all required files exist
    required_files = ['model.pkl', 'scaler.pkl', 'imputer.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"ERROR: Missing required files after training: {missing_files}")
        return
    
    print("\n" + "="*50)
    print("SUCCESS! All files created successfully.")
    print("="*50)
    
    # Step 4: Launch GUI
    print("\nLaunching GUI application...")
    print("Close the GUI window when you're done testing.")
    
    try:
        subprocess.run("python gui_app.py", shell=True)
    except KeyboardInterrupt:
        print("\nGUI closed by user.")
    except Exception as e:
        print(f"Error launching GUI: {e}")

if __name__ == "__main__":
    main() 