import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# Dynamic project root mapping (up one level from src/)
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "Car Sell Dataset.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data"

def run_feature_engineering(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT_DIR):
    """Performs deterministic feature engineering and safe dataset splits."""
    os.makedirs(output_dir, exist_ok=True)
    print("[-] Reading raw dataset...")
    df = pd.read_csv(input_path)
    
    # 1. Temporal Engineering
    current_year = datetime.now().year
    df["Car_Age"] = current_year - df["Year"]
    df["KM_Per_Year"] = df["Kilometers"] / df["Car_Age"].replace(0, 1)
    df = df.drop(columns=["Year"])
    
    # 2. Structural/Ordinal Feature Mapping
    df["Owner"] = df["Owner"].map({"1st": 1, "2nd": 2, "3rd+": 3})
    df["Accidental"] = df["Accidental"].map({"No": 0, "Yes": 1})
    df["Transmission"] = df["Transmission"].map({"Manual": 0, "Automatic": 1})
    
    # 3. Features & Target Splitting
    X = df.drop(columns=["Price"])
    y = df["Price"]
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Save out intermediate splits
    X_train.to_csv(output_dir / "X_train_engineered.csv", index=False)
    X_test.to_csv(output_dir / "X_test_engineered.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)
    print(f"[+] Feature Engineering complete. Splits preserved in '{output_dir}'.")

if __name__ == "__main__":
    run_feature_engineering()