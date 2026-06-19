import os
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dbscan_utils import DBSCANNoveltyDetector  # noqa: F401 -- required so pickle can resolve dbscan.pkl

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
PLOTS_DIR = ROOT_DIR / "plots"

CONSENSUS_VOTES_REQUIRED = 2


def generate_anomaly_insights(data_dir=DATA_DIR, model_dir=MODEL_DIR, plots_dir=PLOTS_DIR):
    os.makedirs(plots_dir, exist_ok=True)

    X_train = pd.read_csv(data_dir / "X_train_engineered.csv")

    # Load the standalone feature pipeline directly -- no more reaching into
    # `regressor.regressor_.named_steps['features']` to recover it.
    with open(model_dir / "feature_pipeline.pkl", "rb") as f:
        feature_pipeline = pickle.load(f)
    with open(model_dir / "iso_forest.pkl", "rb") as f:
        iso_model = pickle.load(f)
    with open(model_dir / "lof.pkl", "rb") as f:
        lof_model = pickle.load(f)
    with open(model_dir / "dbscan.pkl", "rb") as f:
        dbscan_model = pickle.load(f)

    X_scaled = feature_pipeline.transform(X_train)

    iso_anomalies = (iso_model.predict(X_scaled) == -1).astype(int)
    lof_anomalies = (lof_model.predict(X_scaled) == -1).astype(int)
    dbscan_anomalies = (dbscan_model.predict(X_scaled) == -1).astype(int)

    vote_count = iso_anomalies + lof_anomalies + dbscan_anomalies
    consensus = (vote_count >= CONSENSUS_VOTES_REQUIRED).astype(int)

    print("[-] Reducing dimensionality to 3D PCA space...")
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pca_output_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
    pca_output_df["iso_anomaly"] = iso_anomalies
    pca_output_df["lof_anomaly"] = lof_anomalies
    pca_output_df["dbscan_anomaly"] = dbscan_anomalies
    pca_output_df["vote_count"] = vote_count
    pca_output_df["consensus_anomaly"] = consensus
    pca_output_df["fraud_score"] = consensus * 100
    pca_output_df.to_csv(data_dir / "pca_results.csv", index=False)

    normal_mask = (consensus == 0)
    anomaly_mask = (consensus == 1)

    print("[-] Generating static 3D cluster fallback plot...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], X_pca[normal_mask, 2],
               c='#1f77b4', alpha=0.2, s=15, label="Normal Operations")
    ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], X_pca[anomaly_mask, 2],
               c='#d62728', alpha=0.8, s=40, marker='x',
               label="Consensus Fraud Deviance (\u22652 of 3 votes)")

    ax.set_title("Vehicle Market Structures - 3D PCA Mapping", fontsize=14, pad=15)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.legend(loc="upper left")

    output_plot = plots_dir / "consensus_anomaly_3d.png"
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    plt.close()
    print(f"[+] Dimensionality artifact preserved at: '{output_plot}'")
    print(f"[i] Vote distribution: {np.bincount(vote_count, minlength=4).tolist()} "
          f"(index = number of models flagging the listing, 0-3)")


if __name__ == "__main__":
    generate_anomaly_insights()
