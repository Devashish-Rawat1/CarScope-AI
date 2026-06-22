# CarScope AI: India Used Car Fraud Detector

CarScope AI is an end-to-end machine learning pipeline and interactive web application designed to determine fair market valuations and detect pricing and structural anomalies within the Indian used car market.

It utilizes a robust **Ensemble Anomaly Detection Framework** combining **Supervised Target Residuals (XGBoost)** with three complementary **Unsupervised Distance, Density, and Cluster-Membership Models** — Isolation Forest, Local Outlier Factor, and DBSCAN — to flag suspicious vehicle listings in real time.

**Live Demo:** [carscope-ai.streamlit.app](https://carscope-ai-hmrqvpiry8ewehhz3kvevw.streamlit.app/)

---

## Features

### 🔹 End-to-End ML Pipeline

Fully automated data ingestion, temporal feature engineering, model training, anomaly detection, and evaluation scoring.

### 🔹 Multi-Algorithm Consensus Detection

Flags a listing as high-confidence fraud only when at least **2 of 3** independent unsupervised detectors (Isolation Forest, LOF, DBSCAN) agree it's structurally atypical. A separate, weighted price-deviation signal from the XGBoost residual contributes continuously to the final 0–100 fraud score, rather than acting as a fourth binary vote.

### 🔹 Interactive 3D PCA Visualization

Visualizes how Isolation Forest, LOF, and DBSCAN each separate normal vehicle listings from anomalous outliers in a reduced 3D feature space, alongside the combined consensus vote.

### 🔹 Real-Time Listing Evaluation

A modern dark-themed Streamlit application allowing users to input vehicle specifications and instantly receive an **Anomaly Index Rating** with actionable insights — using the exact same scoring logic, feature engineering, and trained artifacts as the batch pipeline, so a listing can never score differently depending on which entry point evaluates it.

---

## Project Structure

```text
 CarScope-AI
├── run_pipeline.py              # Master execution script
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
│
├── app/
│   └── streamlit_app.py        # Interactive Streamlit web application
│
├── src/
│   ├── feature_engineering.py  # Shared feature construction (used by training AND inference)
│   ├── train_models.py         # XGBoost, Isolation Forest, LOF, and DBSCAN training
│   ├── dbscan_utils.py         # DBSCAN wrapper with nearest-core-point inference for new data
│   ├── pca_visualization.py    # PCA computation and visualization generation
│   └── inference.py            # Batch and real-time anomaly inference logic
│
├── data/                       # Raw datasets and processed outputs
├── models/                     # Serialized artifacts (.pkl): feature_pipeline, y_scaler,
│                                #   price_regressor, iso_forest, lof, dbscan
├── plots/                      # Generated visualizations and figures
├── notebooks/                  # EDA and experimentation notebooks
└── hyperparameters/            # Tuned model configurations
```

---

## Installation & Setup

### 1️. Clone the Repository

```bash
git clone https://github.com/yourusername/CarScope-AI.git
cd CarScope-AI
```

### 2. Create a Virtual Environment (Recommended)

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage Guide

### Step 1: Prepare the Dataset

Place your dataset inside the `data/` directory and ensure it is named exactly:

```text
Car Sell Dataset.csv
```

---

### Step 2: Run the Training Pipeline

Execute the master pipeline to:

* Perform feature engineering
* Train the price regressor and all three anomaly detectors (Isolation Forest, LOF, DBSCAN)
* Generate PCA coordinates and consensus vote counts
* Save trained model artifacts
* Run validation diagnostics

```bash
python run_pipeline.py
```

> **Note:** This automatically populates the `models/`, `data/`, and `plots/` directories with production-ready artifacts. If you're re-running after an architecture change to the models, delete the existing contents of `models/` first so stale artifacts can't be mixed with a new pipeline version.

---

### Step 3: Launch the Web Application

After the pipeline completes successfully:

```bash
streamlit run app/streamlit_app.py
```

Open the provided local URL in your browser to access the application.

---

## Algorithmic Methodology

### XGBoost Regressor

Acts as the fair-market valuation engine.

Features are processed through a single shared pipeline:

* Target Encoding for high-cardinality variables (Brand, Model, Variant, State)
* One-Hot Encoding for low-cardinality variables (Fuel Type, Car Type)
* Standard Scaling for the full numerical feature space

Listings with significant deviations between actual and predicted prices are flagged as suspicious. Because the dataset's `Model Variant` field captures only coarse trim tiers rather than granular specs, this model is intentionally positioned as a **price-deviation signal generator for the anomaly ensemble**, not a standalone precision pricing engine — see [Model Performance](#model-performance) below.

---

### Isolation Forest

A global anomaly detection model that identifies structurally unusual vehicles based on feature isolation.

Examples include:

* Unrealistic mileage-to-age ratios
* Rare combinations of vehicle attributes
* Abnormal market positioning

---

### Local Outlier Factor (LOF)

A density-based anomaly detector that compares a listing against its nearest neighbors.

This helps identify vehicles that appear unusual relative to similar:

* Brands
* Variants
* Fuel types
* Geographic regions

---

### DBSCAN (with Nearest-Core-Point Inference)

A density-based clustering model that asks a different question than the other two: *does this listing belong to any natural market cluster at all?* Cars that don't fit any recognizable grouping — wrong price for their cluster, wrong mileage for their cluster, or simply isolated in feature space — are labeled noise.

sklearn's DBSCAN has no native way to evaluate a brand-new listing it wasn't fit on. `dbscan_utils.py` solves this with a standard technique: after fitting DBSCAN once on the training feature space, it builds a fast nearest-neighbor index over only the *core points* of each real cluster. A new listing is treated as belonging to a cluster (normal) if it falls within `eps` of any core point, and as noise (anomaly) otherwise — the same definition of cluster membership DBSCAN itself uses, just made usable for live, single-listing inference.

---

## Ensemble Anomaly Framework

CarScope AI combines four complementary signals into a single 0–100 fraud score:

| Component                       | Role                                              | Weight |
| -------------------------------- | -------------------------------------------------- | -----: |
| Isolation Forest                 | Global structural anomaly (binary vote)             |  45% |
| XGBoost Price Deviation           | Continuous signal: \|listed − predicted\| / predicted |  30% |
| Local Outlier Factor              | Local density anomaly (binary vote)                  |  15% |
| DBSCAN                            | Cluster-membership anomaly (binary vote)              |  10% |

Separately, a listing is marked **Consensus Fraud Deviance** only when at least **2 of the 3** unsupervised detectors (Isolation Forest, LOF, DBSCAN) independently vote it anomalous. This majority-vote rule — rather than flagging on a single model's vote — is what keeps the false-positive rate manageable while still catching genuinely structurally unusual listings.

---

## Model Performance

Results from an actual training run on the full dataset (~141K listings, 80/20 train/test split):

| Metric                          | Value             |
| -------------------------------- | ------------------ |
| R² Score (price model)            | 0.3273              |
| MAE (price model)                 | ₹2,97,917.50         |
| RMSE (price model)                | ₹3,63,570.49         |
| DBSCAN noise fraction (training)  | 1.46%                |

An R² in the 0.27–0.35 range is the expected ceiling for this dataset, not a modeling shortfall — `Model Variant` only captures coarse trim tiers (e.g. "VXI", "ZXI") rather than the granular spec sheet that actually drives used-car pricing. Rather than over-fit to chase a misleadingly high R², the price model is deliberately kept as a **residual generator**: its prediction error is the most informative single signal for the anomaly ensemble, which is the model's actual job here.

**Consensus vote distribution from the same run** (out of ~112.7K training listings):

| Votes Received (of 3 unsupervised detectors) | Listings | % of Training Set |
| ---------------------------------------------- | --------: | -----------------: |
| 0 (normal)                                      |  106,708 |             94.66% |
| 1                                                |    4,558 |              4.04% |
| 2                                                |    1,242 |              1.10% |
| 3 (flagged by all three)                         |      215 |              0.19% |
| **Consensus Fraud (≥2 votes)**                   | **1,457** |          **1.29%** |

---

## Outputs Generated

After running the pipeline, the following artifacts are generated:

* Trained XGBoost price prediction model (`price_regressor.pkl`) and its standalone feature/target scaling pipeline (`feature_pipeline.pkl`, `y_scaler.pkl`)
* Trained Isolation Forest model
* Trained Local Outlier Factor model
* Trained DBSCAN model with nearest-core-point inference wrapper
* PCA-transformed feature space with per-model anomaly flags and consensus vote counts
* Anomaly labels and weighted fraud scores
* Interactive Streamlit dashboard
* Diagnostic visualizations

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Category Encoders
* Plotly
* Matplotlib
* Streamlit
* Pickle (model serialization)

---

## Use Cases

* Used Car Marketplace Monitoring
* Vehicle Price Fraud Detection
* Suspicious Listing Investigation
* Dealer Quality Auditing
* Market Pricing Intelligence
* Automotive Risk Analytics

---

## 👨‍💻 Author

**Devashish Rawat**

Computer Science Student | Machine Learning Enthusiast | Data Science Practitioner

If you found this project useful, consider giving it a ⭐ on GitHub.
