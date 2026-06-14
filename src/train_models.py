import os
import pickle
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

def train_pipeline(data_dir=DATA_DIR, output_dir=MODEL_DIR):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load intermediate splits
    X_train = pd.read_csv(data_dir / "X_train_engineered.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").iloc[:, 0]
    
    # Configure exact tracking column domains
    high_card_cols = ["Brand", "Model Name", "Model Variant", "State"]
    low_card_cols = ["Fuel Type", "Car Type"]
    
    # Feature Engineering Architecture Setup
    preprocessor = ColumnTransformer(
        transformers=[
            ("target", TargetEncoder(min_samples_leaf=20, smoothing=10), high_card_cols),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_card_cols)
        ],
        remainder="passthrough"
    )
    
    # Combined Pipeline Layer (Encoding -> Standardization)
    feature_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler())
    ])
    
    # Base Estimator Core
    xgb_pipeline = Pipeline(steps=[
        ('features', feature_pipeline),
        ('regressor', XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42))
    ])
    
    # Secure Native Target Transformer (Fixes scaling errors)
    final_price_model = TransformedTargetRegressor(
        regressor=xgb_pipeline,
        transformer=StandardScaler()
    )
    
    print("[-] Training Price Regressor...")
    final_price_model.fit(X_train, y_train)
    
    # Extract structural state-weights for downstream anomaly models
    print("[-] Extracting normalized feature spaces for anomaly models...")
    X_train_scaled = feature_pipeline.fit_transform(X_train, y_train)
    
    print("[-] Fitting Outlier Engines...")
    iso_forest = IsolationForest(contamination=0.03, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train_scaled)
    
    lof = LocalOutlierFactor(n_neighbors=15, contamination=0.03, n_jobs=-1, novelty=True)
    lof.fit(X_train_scaled)
    
    # Export artifacts to disk safely
    print("[+] Exporting model binaries...")
    with open(output_dir / "price_regressor.pkl", "wb") as f:
        pickle.dump(final_price_model, f)
    with open(output_dir / "iso_forest.pkl", "wb") as f:
        pickle.dump(iso_forest, f)
    with open(output_dir / "lof.pkl", "wb") as f:
        pickle.dump(lof, f)
        
    print("[+] Training complete.")

if __name__ == "__main__":
    train_pipeline()