import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.feature_engineering import engineer_features
from src.dbscan_utils import DBSCANNoveltyDetector  # noqa: F401 -- required so pickle can resolve dbscan.pkl

MODEL_DIR = ROOT_DIR / "models"

# Single source of truth for ensemble weights and the consensus rule. Both
# execute_inference() (batch) and analyze_single_listing() (the live
# Streamlit tool) call the SAME _compute_fraud_score() helper below -- that's
# what fixes the original bug where the two paths silently used different
# scoring formulas (a baseline of 15/20 points for "unflagged" listings in
# the single-listing path, vs. 0 in the batch path) and could disagree on the
# fraud score for the exact same car.
WEIGHTS = {
    "iso_forest": 0.45,
    "price_deviation": 0.30,
    "lof": 0.15,
    "dbscan": 0.10,
}

# Require at least 2 of the 3 independent anomaly detectors to agree before
# calling something "consensus fraud". The old logic was `(iso + lof) >= 1`,
# i.e. a single model's vote was enough -- with both models independently
# tuned to ~3% contamination, that's a weak bar that inflates false positives.
CONSENSUS_VOTES_REQUIRED = 2

ARTIFACT_NAMES = ["feature_pipeline", "y_scaler", "price_regressor", "iso_forest", "lof", "dbscan"]


def _load_artifacts(model_dir=MODEL_DIR):
    """Loads every model artifact directly and flatly -- no reaching into a
    fitted estimator's internal attributes (e.g. the old
    `price_model.regressor_.named_steps['features']` pattern), which was
    fragile and broke if sklearn's internal pipeline structure ever changed."""
    artifacts = {}
    for name in ARTIFACT_NAMES:
        path = model_dir / f"{name}.pkl"
        with open(path, "rb") as f:
            artifacts[name] = pickle.load(f)
    return artifacts


def _compute_fraud_score(iso_flag, lof_flag, dbscan_flag, price_deviation_pct):
    """The one and only fraud-score formula in the system."""
    deviation_contrib = np.minimum(100, np.abs(price_deviation_pct) * 2)
    score = (
        WEIGHTS["iso_forest"] * iso_flag * 100
        + WEIGHTS["lof"] * lof_flag * 100
        + WEIGHTS["dbscan"] * dbscan_flag * 100
        + WEIGHTS["price_deviation"] * deviation_contrib
    )
    return score, deviation_contrib


def _risk_level(score):
    if score < 35:
        return "Low"
    if score < 60:
        return "Medium"
    if score < 80:
        return "High"
    return "Very High"


def execute_inference(new_data_path=None, labels_path=None, model_dir=MODEL_DIR):
    """Applies the trained ensemble to a batch of listings."""
    if new_data_path is None:
        new_data_path = ROOT_DIR / "data" / "X_test_engineered.csv"

    X_new = pd.read_csv(new_data_path)

    if labels_path is None and "X_test" in str(new_data_path):
        labels_path = ROOT_DIR / "data" / "y_test.csv"

    has_price = False
    if labels_path and Path(labels_path).exists():
        y_actual = pd.read_csv(labels_path).iloc[:, 0]
        X_new["Price"] = y_actual.values
        has_price = True

    artifacts = _load_artifacts(model_dir)
    X_features = X_new.drop(columns=["Price"]) if "Price" in X_new.columns else X_new.copy()

    print("[-] Running structured predictions...")
    X_scaled = artifacts["feature_pipeline"].transform(X_features)

    predicted_scaled = artifacts["price_regressor"].predict(X_scaled)
    predicted_prices = artifacts["y_scaler"].inverse_transform(
        predicted_scaled.reshape(-1, 1)
    ).ravel()

    iso_flags = (artifacts["iso_forest"].predict(X_scaled) == -1).astype(int)
    lof_flags = (artifacts["lof"].predict(X_scaled) == -1).astype(int)
    dbscan_flags = (artifacts["dbscan"].predict(X_scaled) == -1).astype(int)

    vote_count = iso_flags + lof_flags + dbscan_flags
    consensus_fraud = (vote_count >= CONSENSUS_VOTES_REQUIRED).astype(int)

    if has_price:
        price_deviation_pct = (
            (X_new["Price"].values - predicted_prices) / predicted_prices
        ) * 100
    else:
        price_deviation_pct = np.zeros(len(X_new))

    fraud_score, _ = _compute_fraud_score(iso_flags, lof_flags, dbscan_flags, price_deviation_pct)

    results_df = X_new.copy()
    results_df["Predicted_Price_INR"] = np.round(predicted_prices, 2)
    results_df["iso_anomaly"] = iso_flags
    results_df["lof_anomaly"] = lof_flags
    results_df["dbscan_anomaly"] = dbscan_flags
    results_df["vote_count"] = vote_count
    results_df["Is_Fraudulent_Deviance"] = consensus_fraud
    results_df["is_anomaly"] = consensus_fraud
    results_df["fraud_score"] = fraud_score
    results_df["price_deviation_pct"] = price_deviation_pct

    return results_df


def analyze_single_listing(user_features, model_dir=MODEL_DIR):
    """Live, single-listing path used by the Streamlit app. Uses the exact
    same artifacts, the exact same engineer_features() function, and the
    exact same _compute_fraud_score()/consensus rule as execute_inference()
    above -- there is no separate logic left to drift out of sync."""
    artifacts = _load_artifacts(model_dir)

    listed_price = float(user_features.get("Price", 0))
    raw_df = pd.DataFrame([{k: v for k, v in user_features.items() if k != "Price"}])
    feature_df = engineer_features(raw_df)

    X_scaled = artifacts["feature_pipeline"].transform(feature_df)

    predicted_scaled = artifacts["price_regressor"].predict(X_scaled)
    predicted_price = artifacts["y_scaler"].inverse_transform(
        predicted_scaled.reshape(-1, 1)
    ).ravel()[0]

    iso_flag = int(artifacts["iso_forest"].predict(X_scaled)[0] == -1)
    lof_flag = int(artifacts["lof"].predict(X_scaled)[0] == -1)
    dbscan_flag = int(artifacts["dbscan"].predict(X_scaled)[0] == -1)
    vote_count = iso_flag + lof_flag + dbscan_flag

    price_deviation_pct = ((listed_price - predicted_price) / predicted_price) * 100
    fraud_score, deviation_contrib = _compute_fraud_score(
        iso_flag, lof_flag, dbscan_flag, price_deviation_pct
    )
    risk_lvl = _risk_level(fraud_score)

    deviance_types = []
    insights = []

    if price_deviation_pct < -35:
        deviance_types.append("Severe Underpricing")
        insights.append(
            "Market listing is severely underpriced compared to fair evaluation "
            "metrics. Inspect vehicle title records for severe accidents or "
            "structural discrepancies."
        )
    elif price_deviation_pct > 35:
        deviance_types.append("Severe Overpricing")
        insights.append(
            "The asking value significantly exceeds realistic territorial "
            "baselines for this variant trim."
        )

    if feature_df["KM_Per_Year"].values[0] < 2000 and feature_df["Car_Age"].values[0] > 6:
        deviance_types.append("Suspicious Odometer Reading")
        insights.append(
            "The recorded odometer values display abnormally low mileage "
            "accumulation for the vehicle's structural operational life. Check "
            "service logs for tampering indicators."
        )

    if vote_count >= CONSENSUS_VOTES_REQUIRED:
        deviance_types.append("Consensus Structural Anomaly")
        insights.append(
            f"{vote_count} of 3 anomaly-detection models (Isolation Forest, LOF, "
            "DBSCAN) independently flagged this listing's feature profile as "
            "atypical for the broader market."
        )

    if not deviance_types:
        deviance_types.append("Standard Performance Matrix")
        insights.append(
            "The statistical properties of this asset match default structural "
            "patterns within the territory."
        )

    return {
        "fraud_score": fraud_score,
        "risk_level": risk_lvl,
        "anomaly_type": ", ".join(deviance_types),
        "predicted_price": predicted_price,
        "listed_price": listed_price,
        "price_gap": listed_price - predicted_price,
        "price_deviation_pct": price_deviation_pct,
        "vote_count": vote_count,
        "signals": {
            "Isolation Forest": WEIGHTS["iso_forest"] * iso_flag * 100,
            "Local Outlier Factor": WEIGHTS["lof"] * lof_flag * 100,
            "DBSCAN": WEIGHTS["dbscan"] * dbscan_flag * 100,
            "Price Deviation": WEIGHTS["price_deviation"] * deviation_contrib,
        },
        "votes": {
            "Isolation Forest": "FLAGGED" if iso_flag else "Normal Cluster State",
            "Local Outlier Factor": "FLAGGED" if lof_flag else "Normal Cluster State",
            "DBSCAN": "FLAGGED" if dbscan_flag else "Normal Cluster State",
        },
        "insights": insights,
    }


if __name__ == "__main__":
    predictions = execute_inference()
    print("\n[+] Verification inference preview:")
    print(
        predictions[
            ["Brand", "Model Name", "Predicted_Price_INR", "vote_count", "Is_Fraudulent_Deviance"]
        ].head()
    )
