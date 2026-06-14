"""
run_pipeline.py
---------------
One-shot script to:
  1. Run feature engineering on raw data
  2. Train all models
  3. Print a summary

Usage:
    python run_pipeline.py --data "data/Car Sell Dataset.csv"
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Add src folder path explicitly to scope
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Step Imports from local src folder
from src.feature_engineering import run_feature_engineering
from src.train_models import train_pipeline
from src.pca_visualization import generate_anomaly_insights
from src.inference import execute_inference

def main():
    print("====================================================")
    print("   LAUNCHING UNIFIED CAR SELLING VALUE PIPELINE     ")
    print("====================================================\n")
    
    # Step 1: Feature Engineering
    run_feature_engineering()
    print("-" * 50)
    
    # Step 2: Training & Preservation
    train_pipeline()
    print("-" * 50)
    
    # Step 3: Anomaly Analysis and 3D Visual Plot Mapping
    generate_anomaly_insights()
    print("-" * 50)
    
    # Step 4: Out-of-Sample Validation Diagnostics
    print("[-] Running validations on test split...")
    
    # Now mapping the labels path explicitly so target-aware deviations map correctly
    test_predictions = execute_inference(
        new_data_path=ROOT_DIR / "data" / "X_test_engineered.csv",
        labels_path=ROOT_DIR / "data" / "y_test.csv"
    )
    
    # Save the evaluation records for downstream reference
    test_predictions.to_csv(ROOT_DIR / "data" / "results.csv", index=False)
    
    y_test = pd.read_csv(ROOT_DIR / "data" / "y_test.csv").iloc[:, 0]
    y_pred = test_predictions["Predicted_Price_INR"].values
    
    # Evaluation Calculations
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\n====================================================")
    print("         PRODUCTION SYSTEM MODEL PERFORMANCE         ")
    print("====================================================")
    print(f" -> Explained Variance (R2 Score): {r2:.4f}")
    print(f" -> Mean Absolute Error (MAE)   : INR {mae:,.2f}")
    print(f" -> Root Mean Squared Error(RMSE): INR {rmse:,.2f}")
    print("====================================================\n")
    print("[+] End-to-end processing pipeline execution successful.")

if __name__ == "__main__":
    main()