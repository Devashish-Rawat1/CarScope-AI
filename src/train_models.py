import os
import sys
import pickle
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dbscan_utils import DBSCANNoveltyDetector

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

# Named constants instead of magic numbers, kept here as the single source of
# truth -- inference.py and pca_visualization.py read the *fitted* model's own
# .eps/.min_samples (via DBSCANNoveltyDetector) rather than re-declaring these,
# so training and inference can never disagree about what was actually fit.
ISO_CONTAMINATION = 0.03
LOF_N_NEIGHBORS = 15
LOF_CONTAMINATION = 0.03
DBSCAN_EPS = 1.5
DBSCAN_MIN_SAMPLES = 3


def train_pipeline(data_dir=DATA_DIR, output_dir=MODEL_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # Load intermediate splits
    X_train = pd.read_csv(data_dir / "X_train_engineered.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").iloc[:, 0]

    # Configure exact tracking column domains
    high_card_cols = ["Brand", "Model Name", "Model Variant", "State"]
    low_card_cols = ["Fuel Type", "Car Type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("target", TargetEncoder(min_samples_leaf=20, smoothing=10), high_card_cols),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_card_cols),
        ],
        remainder="passthrough",
    )

    # This is the ONLY copy of the feature pipeline in the whole system. It's
    # fit once here and saved as its own artifact (feature_pipeline.pkl) below.
    # Previously, inference reached into
    # `price_model.regressor_.named_steps['features']` to recover this --
    # fragile, version-sensitive, and broke if the price model's internal
    # structure ever changed. Saving it standalone removes that coupling
    # entirely: every consumer just loads feature_pipeline.pkl directly.
    feature_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
    ])

    print("[-] Fitting shared feature pipeline...")
    X_train_scaled = feature_pipeline.fit_transform(X_train, y_train)

    # Target scaling is now explicit and saved on its own too, for the same
    # reason -- no more digging through a TransformedTargetRegressor's
    # internals to invert the prediction back to rupees.
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    print("[-] Training Price Regressor...")
    price_model = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    price_model.fit(X_train_scaled, y_train_scaled)

    print("[-] Fitting Isolation Forest...")
    iso_forest = IsolationForest(contamination=ISO_CONTAMINATION, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train_scaled)

    print("[-] Fitting Local Outlier Factor...")
    lof = LocalOutlierFactor(
        n_neighbors=LOF_N_NEIGHBORS, contamination=LOF_CONTAMINATION, n_jobs=-1, novelty=True
    )
    lof.fit(X_train_scaled)

    print("[-] Fitting DBSCAN (with nearest-core-point lookup for live inference)...")
    dbscan_model = DBSCANNoveltyDetector(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
    dbscan_model.fit(X_train_scaled)
    noise_frac = dbscan_model.noise_fraction_
    print(f"[i] DBSCAN flagged {noise_frac:.2%} of training points as noise "
          f"(eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}).")
    if noise_frac > 0.15:
        print("[!] That's a high noise fraction relative to the ~3% contamination used "
              "for Isolation Forest/LOF -- DBSCAN may end up agreeing with almost "
              "everything. Consider raising eps or min_samples and refitting.")
    elif noise_frac < 0.005:
        print("[!] That's a very low noise fraction -- DBSCAN may barely ever flag "
              "anything. Consider lowering eps or min_samples and refitting.")

    # Export artifacts to disk safely
    print("[+] Exporting model binaries...")
    artifacts = {
        "feature_pipeline.pkl": feature_pipeline,
        "y_scaler.pkl": y_scaler,
        "price_regressor.pkl": price_model,
        "iso_forest.pkl": iso_forest,
        "lof.pkl": lof,
        "dbscan.pkl": dbscan_model,
    }
    for filename, obj in artifacts.items():
        with open(output_dir / filename, "wb") as f:
            pickle.dump(obj, f)

    print("[+] Training complete.")


if __name__ == "__main__":
    train_pipeline()
