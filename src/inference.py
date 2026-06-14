import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"

def execute_inference(new_data_path=None, labels_path=None, model_dir=MODEL_DIR):
    """Applies validation inferences against stored operational models for a dataset."""
    if new_data_path is None:
        new_data_path = ROOT_DIR / "data" / "X_test_engineered.csv"
        
    X_new = pd.read_csv(new_data_path)
    
    # Map target labels if provided or implicitly detected
    if labels_path is None and "X_test" in str(new_data_path):
        labels_path = ROOT_DIR / "data" / "y_test.csv"
        
    if labels_path and Path(labels_path).exists():
        y_actual = pd.read_csv(labels_path).iloc[:, 0]
        X_new["Price"] = y_actual.values
        
    with open(model_dir / "price_regressor.pkl", "rb") as f:
        price_model = pickle.load(f)
    with open(model_dir / "iso_forest.pkl", "rb") as f:
        iso_model = pickle.load(f)
    with open(model_dir / "lof.pkl", "rb") as f:
        lof_model = pickle.load(f)
        
    print("[-] Running structured predictions...")
    
    # Isolate features safely to prevent column mismatch with the trained estimators
    X_features = X_new.drop(columns=["Price"]) if "Price" in X_new.columns else X_new.copy()
    
    predicted_prices = price_model.predict(X_features)
    
    feature_transformer = price_model.regressor_.named_steps['features']
    X_new_scaled = feature_transformer.transform(X_features)
    
    iso_flags = (iso_model.predict(X_new_scaled) == -1).astype(int)
    lof_flags = (lof_model.predict(X_new_scaled) == -1).astype(int)
    consensus_fraud = ((iso_flags + lof_flags) >= 1).astype(int)
    
    results_df = X_new.copy()
    results_df["Predicted_Price_INR"] = np.round(predicted_prices, 2)
    results_df["Is_Fraudulent_Deviance"] = consensus_fraud
    
    # Calculate continuous fraud score based on residual deviations
    if "Price" in results_df.columns:
        price_deviation_pct = ((results_df["Price"] - results_df["Predicted_Price_INR"]) / results_df["Predicted_Price_INR"]) * 100
        deviation_contrib = np.minimum(100, np.abs(price_deviation_pct) * 2)
    else:
        price_deviation_pct = 0.0
        deviation_contrib = 0.0
        
    results_df["fraud_score"] = (0.40 * iso_flags * 100) + (0.35 * deviation_contrib) + (0.25 * lof_flags * 100)
    results_df["is_anomaly"] = consensus_fraud
    results_df["price_deviation_pct"] = price_deviation_pct
    
    return results_df

def analyze_single_listing(user_features, output_dir=MODEL_DIR):
    """Ingests data from single interface mappings, processes features and runs predictions."""
    with open(output_dir / "price_regressor.pkl", "rb") as f:
        price_model = pickle.load(f)
    with open(output_dir / "iso_forest.pkl", "rb") as f:
        iso_model = pickle.load(f)
    with open(output_dir / "lof.pkl", "rb") as f:
        lof_model = pickle.load(f)

    raw_df = pd.DataFrame([user_features])

    current_year = datetime.now().year
    raw_df["Car_Age"] = current_year - raw_df["Year"]
    raw_df["KM_Per_Year"] = raw_df["Kilometers"] / raw_df["Car_Age"].replace(0, 1)
    raw_df = raw_df.drop(columns=["Year"])

    raw_df["Owner"] = raw_df["Owner"].map({"1st": 1, "2nd": 2, "3rd+": 3})
    raw_df["Accidental"] = raw_df["Accidental"].map({"No": 0, "Yes": 1})
    raw_df["Transmission"] = raw_df["Transmission"].map({"Manual": 0, "Automatic": 1})

    # Cache target and separate features prior to pipeline execution
    listed_val = float(user_features.get("Price", 0))
    if "Price" in raw_df.columns:
        raw_df = raw_df.drop(columns=["Price"])

    predicted_val = price_model.predict(raw_df)[0]

    feature_transformer = price_model.regressor_.named_steps['features']
    scaled_coordinates = feature_transformer.transform(raw_df)

    iforest_vote = iso_model.predict(scaled_coordinates)[0]
    lof_vote = lof_model.predict(scaled_coordinates)[0]

    price_deviation_pct = ((listed_val - predicted_val) / predicted_val) * 100

    iforest_contrib = 100 if iforest_vote == -1 else 15
    lof_contrib = 100 if lof_vote == -1 else 20
    deviation_contrib = min(100, abs(price_deviation_pct) * 2)

    composite_fraud_score = (0.40 * iforest_contrib) + (0.35 * deviation_contrib) + (0.25 * lof_contrib)

    if composite_fraud_score < 35:
        risk_lvl = "Low"
    elif composite_fraud_score < 60:
        risk_lvl = "Medium"
    elif composite_fraud_score < 80:
        risk_lvl = "High"
    else:
        risk_lvl = "Very High"

    deviance_types = []
    insights = []
    if price_deviation_pct < -35:
        deviance_types.append("Severe Underpricing")
        insights.append("Market listing is severely underpriced compared to fair evaluation metrics. Inspect vehicle title records for severe accidents or structural discrepancies.")
    elif price_deviation_pct > 35:
        deviance_types.append("Severe Overpricing")
        insights.append("The asking value significantly exceeds realistic territorial baselines for this variant trim.")
        
    if raw_df["KM_Per_Year"].values[0] < 2000 and raw_df["Car_Age"].values[0] > 6:
        deviance_types.append("Suspicious Odometer Reading")
        insights.append("The recorded odometer values display abnormally low mileage accumulation for the vehicle's structural operational life. Check service logs for tampering indicators.")

    if not deviance_types:
        deviance_types.append("Standard Performance Matrix")
        insights.append("The statistical properties of this asset match default structural patterns within the territory.")

    return {
        "fraud_score": composite_fraud_score,
        "risk_level": risk_lvl,
        "anomaly_type": ", ".join(deviance_types),
        "predicted_price": predicted_val,
        "listed_price": listed_val,
        "price_gap": listed_val - predicted_val,
        "price_deviation_pct": price_deviation_pct,
        "signals": {
            "Global Isolation Matrix": iforest_contrib,
            "Local Distance Deviance": lof_contrib,
            "Target Value Delta": deviation_contrib
        },
        "votes": {
            "Isolation Forest Engine": "FLAGGED" if iforest_vote == -1 else "Normal Cluster State",
            "Local Outlier Factor Tracker": "FLAGGED" if lof_vote == -1 else "Normal Cluster State"
        },
        "insights": insights
    }

if __name__ == "__main__":
    predictions = execute_inference()
    print("\n[+] Verification inference preview:")
    print(predictions[["Brand", "Model Name", "Predicted_Price_INR", "Is_Fraudulent_Deviance"]].head())