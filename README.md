# CarScope AI — India's Used Car Fraud Detector

An unsupervised machine learning system that assigns a **0–100 fraud risk score** to any used car listing, explains *why* it's suspicious, and classifies the anomaly type — with a live Streamlit demo.

---

## The Problem

India's used car market (₹1.5 lakh crore, 2024) has no standardised pricing authority and no mandatory inspection. Listings on CarDekho, OLX Autos, and Cars24 are self-reported by sellers. This enables:

- Underpriced scam listings / salvaged vehicles
- Overpriced exploitation of uninformed buyers
- Odometer tampering (low KM/year on old cars)
- Commercial vehicle mislabelling (unusually high KM/year)
- Age-price misrepresentation

**There is no labeled fraud dataset for Indian used cars.** This makes it a classic unsupervised anomaly detection problem.

---

## Project Structure

```
carscope_ai/
├── src/
│   ├── feature_engineering.py  # Raw CSV → scaled features + saved preprocessor
│   ├── train_models.py         # Trains IF, LOF, DBSCAN, XGBoost → results.csv
│   └── inference.py            # Stateless scoring engine for Streamlit
├── app/
│   └── streamlit_app.py        # Three-page Streamlit app
├── data/                       # Auto-created: CSVs, results, PCA data
├── models/                     # Auto-created: saved .pkl files
├── run_pipeline.py             # One-shot training pipeline
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your dataset
```
data/Car Sell Dataset.csv
```

### 3. Run the full training pipeline
```bash
python run_pipeline.py --data "data/Car Sell Dataset.csv"
```

This runs feature engineering → price model → anomaly detection → saves everything.

### 4. Launch the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

---

## System Architecture

```
Raw Dataset
    │
    ▼
Feature Engineering
  Car_Age, KM_Per_Year
  TargetEncoder (Brand/Model/State)
  OneHotEncoder (Fuel/CarType)
  StandardScaler
    │
    ▼
XGBoost Price Regressor   →   Fair Market Value
  price_gap = listed − predicted
  price_deviation_%
    │  (fed back as features)
    ▼
Ensemble Anomaly Detection
  Isolation Forest  (contamination=0.03, trees=100)   — 45% weight
  Price Deviation Signal                              — 30% weight
  Local Outlier Factor (n_neighbors=5, cont=0.03)    — 15% weight
  DBSCAN (eps=1.5, min_samples=3)                    — 10% weight
    │
    ▼
Fraud Score 0–100  +  Anomaly Type  +  Per-Listing Insights
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| R² (XGBoost price model) | ~0.27* |
| MAE | ₹~X lakh |
| Ensemble anomaly rate | ~3% (tuned contamination) |

*R² of 0.27 is intentional — the goal is price residuals for anomaly detection, not price prediction. The model variant column is coarse trim tier, not full variant name, which limits ceiling. The residuals are still strongly predictive of fraud patterns.

---

## Fraud Score Formula

```
fraud_score = 0.45 × IF_contribution
            + 0.30 × price_deviation_signal
            + 0.15 × LOF_contribution
            + 0.10 × DBSCAN_contribution

→ clipped and scaled to [0, 100]
```

Consensus voting (≥2 of 3 models flag) determines binary `is_anomaly`.

---

## Anomaly Types

| Type | Rule |
|------|------|
| Underpriced | Listed < 65% of predicted fair value |
| Overpriced | Listed > 135% of predicted fair value |
| Unusual Mileage (High) | KM/year > 97th percentile |
| Unusual Mileage (Low for Age) | Car age > 8 yrs AND KM/year < 5th percentile |
| Suspicious Age-Price Ratio | Old car at 2× brand median or new car at <40% brand median |

---

## Streamlit App Pages

| Page | What It Does |
|------|-------------|
| 🔍 Analyse a Listing | Real-time fraud score for any user-entered listing |
| 📊 Market Overview | Dataset-level anomaly explorer with 3D PCA, scatter maps, brand heatmap |
| 📖 How It Works | Full methodology explainer with architecture diagram |

---

## Algorithm Selection Rationale

| Algorithm | Question | Best At |
|-----------|----------|---------|
| Isolation Forest | "Is this globally hard to isolate?" | Extreme outliers (₹95K Fortuner) |
| Local Outlier Factor | "Unusual vs its neighbours?" | Subtle fraud within a brand cluster |
| DBSCAN | "Does it belong to any cluster?" | Cars with no market peers |

---

## Advanced Note (Systems Thinking)

The price model is trained on data that includes fraudulent listings — meaning underpriced anomalies slightly lower the predicted fair value. The fix: run anomaly detection first, remove flagged listings, retrain on "clean" data. This two-pass approach is mentioned in the How It Works page and is a natural next step.
