# CarScope AI: India Used Car Fraud Detector

CarScope AI is an end-to-end machine learning pipeline and interactive web application designed to determine fair market valuations and detect pricing and structural anomalies within the Indian used car market.

It utilizes a robust **Ensemble Anomaly Detection Framework** combining **Supervised Target Residuals (XGBoost)** and **Unsupervised Distance/Density Models (Isolation Forest & Local Outlier Factor)** to flag suspicious vehicle listings in real time.

---

## Features

### 🔹 End-to-End ML Pipeline

Fully automated data ingestion, temporal feature engineering, model training, anomaly detection, and evaluation scoring.

### 🔹 Multi-Algorithm Consensus Detection

Flags suspicious listings only when multiple mathematical perspectives (Global Isolation, Local Density, and Price Deviation) agree.

### 🔹 Interactive 3D PCA Visualization

Visualizes how different anomaly detection algorithms separate normal vehicle listings from anomalous outliers in a reduced 3D feature space.

### 🔹 Real-Time Listing Evaluation

A modern dark-themed Streamlit application allowing users to input vehicle specifications and instantly receive an **Anomaly Index Rating** with actionable insights.

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
│   ├── feature_engineering.py  # Data preprocessing and feature creation
│   ├── train_models.py         # XGBoost, Isolation Forest and LOF training
│   ├── pca_visualization.py    # PCA computation and visualization generation
│   └── inference.py            # Real-time anomaly inference logic
│
├── data/                       # Raw datasets and processed outputs
├── models/                     # Serialized trained models (.pkl files)
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
* Train all anomaly detection models
* Generate PCA coordinates
* Save trained models
* Run validation diagnostics

```bash
python run_pipeline.py
```

> **Note:** This automatically populates the `models/`, `data/`, and `plots/` directories with production-ready artifacts.

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

Features are processed through:

* Target Encoding for high-cardinality variables (Brand, Model, Variant, State)
* Standard Scaling for numerical features

Listings with significant deviations between actual and predicted prices are flagged as suspicious.

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

## Ensemble Anomaly Framework

CarScope AI combines three complementary perspectives:

| Component                 | Purpose                             |
| ------------------------- | ----------------------------------- |
| XGBoost Residual Analysis | Detects abnormal pricing            |
| Isolation Forest          | Detects global structural anomalies |
| Local Outlier Factor      | Detects local density anomalies     |

A listing receives a higher anomaly score when multiple detectors agree, significantly reducing false positives and improving fraud detection reliability.

---

## Outputs Generated

After running the pipeline, the following artifacts are generated:

* Trained XGBoost price prediction model
* Trained Isolation Forest model
* Trained Local Outlier Factor model
* PCA-transformed feature space
* Anomaly labels and scores
* Interactive Streamlit dashboard
* Diagnostic visualizations

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Plotly
* Matplotlib
* Streamlit
* Joblib

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
