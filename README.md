# Machine Learning — GitHub Bot Detection

> **Course:** Machine Learning
> **Professor:** Prof. Dr. Lule AHMEDI
> **Assistant:** Dr. Sc. Mërgim H. HOTI
> **University:** University of Prishtina "Hasan Prishtina"
> **Faculty:** Faculty of Electrical and Computer Engineering (FIEK)
> **Study Level:** Master — Semester II | Academic Year: 2025/26

---

## Contributors

| Name |
|------|
| `Dredhza Braina` |
| `Suhejla Hoxha` |
| `Ylljete Kicaj` |

---

## Project Overview

This project builds a **complete end-to-end Machine Learning pipeline** for a GitHub activity-log dataset. The goal is to predict whether an actor is a **bot or a human** (`actor_is_bot` — binary classification).

| Phase | Focus | Key Output |
|-------|-------|------------|
| **Phase I** | Data Preparation | Clean, ML-ready dataset (`processed_github.csv`) |
| **Phase II** | Model Training & Analysis | 6 classifiers + 4 clustering algorithms |
| **Phase III** | Retraining & Deployment | Optimised Random Forest saved for production |

---

## Dataset Details

| Attribute | Value |
|-----------|-------|
| **Name** | GitHub Activity Log |
| **Original format** | JSON (flattened to CSV) |
| **Rows** | 10,000 |
| **Columns** | 590 |
| **Source** | GitHub Audit Log API |
| **Target variable** | `actor_is_bot` (0 = human, 1 = bot) |
| **Time period** | Jan 10, 2026 — Mar 11, 2026 (59 days) |
| **Class imbalance** | 84.95% human / 15.05% bot (5.6:1 ratio) |

---

## Repository Structure

```
ML_GR10-main/
├── unprocessed dataset/
│   └── github.csv                    ← raw GitHub audit log
├── processed dataset/
│   └── processed_github.csv          ← Phase I output (ML-ready)
├── phase3_outputs/                   ← NEW — Phase III outputs
│   ├── retrained_model.joblib        ← production-ready Random Forest
│   ├── scaler.joblib                 ← fitted StandardScaler
│   ├── feature_names.json            ← ordered feature list
│   ├── phase3_metrics_summary.csv
│   └── *.png  (10 diagnostic plots)
├── eda_plots/                        ← Phase I  (17 plots)
├── phase2_plots/                     ← Phase II (40+ plots)
├── src/
│   ├── main.py                       ← Phase I  entry point
│   ├── main_phase2.py                ← Phase II entry point
│   ├── phase3_retraining.py          ← Phase III entry point  ← NEW
│   ├── supervised_learning.py
│   ├── unsupervised_learning.py
│   ├── advanced_preprocessing.py
│   ├── class_balancing.py
│   ├── cleaning.py
│   ├── data_collection.py
│   ├── data_quality.py
│   ├── eda_analyzer.py
│   ├── integration.py
│   ├── outlier_detection.py
│   └── requirements.txt
└── README.md
```

---

## How to Run

```bash
# Install dependencies
pip install -r src/requirements.txt

# Phase I — Data Preparation
python src/main.py

# Phase II — Model Training & Analysis
python src/main_phase2.py

# Phase III — Retraining & Deployment
python src/phase3_retraining.py
```

## Requirements

```
pandas>=2.0 | numpy>=1.24 | scikit-learn>=1.3 | scipy>=1.11
matplotlib>=3.7 | seaborn>=0.12
```

---

---

# What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence where computers **learn patterns from data** instead of being explicitly programmed with rules. Given enough labelled examples, an ML model discovers statistical structure and makes predictions on new, unseen inputs.

**Three main types:**
- **Supervised Learning** — learns from labelled examples (our main approach)
- **Unsupervised Learning** — finds hidden structure without labels (clustering in Phase II)
- **Reinforcement Learning** — learns by trial and error with rewards

**Why ML for bot detection?**
Traditional rule-based systems (e.g., "block if >100 requests/minute") are easy to evade. ML finds subtle, multi-dimensional patterns across behavioural features that bots cannot easily disguise.

---

---

# Phase I — Data Preparation

---

## Step 1 — Data Loading & Overview

```
Shape         : 10,000 rows × 590 columns
Unique actions: 481
Time range    : Jan 10, 2026 → Mar 11, 2026 (59 days)
```

---

## Step 2 — Data Quality Analysis

| Metric | Value |
|--------|-------|
| Full-row duplicates | 0 |
| `actor_is_bot` filled | 9,155 / 10,000 (91.55%) |
| Overall missing rate | 94.44% |
| Unique action types | 481 |

---

## Step 3 — Missing Values Handling

1. **Drop columns with >50% NaN** → 565 columns dropped, **25 remain**
2. **Fill categorical columns with `"Unknown"`**
3. **Fill numeric columns with column median**

```
Before : 5,572,204 NaN in 575 columns
After  : 118 NaN   (only created_at)
Columns: 590 → 25
```

---

## Step 4 — Feature Engineering (10 derived features)

| Feature | Why |
|---------|-----|
| `event_hour` | Humans peak in business hours; bots are uniform 24/7 |
| `event_dayofweek` / `is_weekend` | Humans less active on weekends |
| `actor_event_count` | Bots generate far more events per actor |
| `actor_event_velocity` | Events per day — bots are abnormally fast |
| `is_integration_event` | 1 if event originates from an automated integration |
| `actor_active_days` | Bots tend to be continuously active |

---

## Step 5 — PCA Dimensionality Reduction

```
Input  : 48 numeric features
Output : 15 principal components
Variance retained: 96.7%
```

---

## Step 6 — Class Imbalance: SMOTE & ADASYN

| Method | Before | After |
|--------|--------|-------|
| **SMOTE** | Bot: 1,214 | Bot: 6,786 |
| **ADASYN** | Bot: 1,214 | Bot: 7,284 |

Both implemented **from scratch** (no imblearn dependency).

---

## Step 7 — Outlier Detection (5 combined methods)

| Method | Outliers Found |
|--------|---------------|
| IQR | 7,307 |
| Z-Score | 2,742 |
| Isolation Forest | 399 |
| Local Outlier Factor | 395 |
| Rare Categories | 8,000 |

Confirmed outliers (≥2 methods): **2,699** — flagged via `is_outlier`, not removed.

---

## Phase I — Final Dataset

| Metric | Value |
|--------|-------|
| Rows | 8,000 |
| Columns | 20 |
| NaN values | **0** |
| Target distribution | 84.8% human / 15.2% bot |
| Output file | `processed dataset/processed_github.csv` |

---

---

# Phase II — Model Training & Analysis

---

## Why Random Forest?

**Random Forest** is an ensemble of many Decision Trees trained on random data subsets. Selected because:

- Handles class imbalance natively via `class_weight='balanced'`
- Resistant to overfitting — averaging many trees reduces variance
- Provides feature importance rankings
- No scaling required — tree splits are threshold-based
- Excellent for high-dimensional, sparse security data

### How it works (step-by-step)

1. **Bootstrap sampling**: Create N random subsets of training data
2. **Train N trees**: Each tree considers only a random feature subset at each split
3. **Predict**: Majority vote across all trees → final prediction
4. **Result**: More accurate and more stable than any single tree

---

## Part A — Supervised Learning Results

| Model | Accuracy | F1 | ROC-AUC | CV F1 |
|-------|----------|----|---------|-------|
| Logistic Regression | 0.9213 | 0.7941 | 0.9642 | 0.7655 ± 0.014 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.000 |
| **Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000 ± 0.000** |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.000 |
| SVM (RBF) | 1.0000 | 1.0000 | 1.0000 | 0.9985 ± 0.002 |
| K-Nearest Neighbours | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.000 |

> **On perfect scores:** Phase I's engineered features — especially `is_outlier`, `actor_event_velocity`, and `is_integration_event` — create near-perfect class separation. Logistic Regression (linear boundary) scores F1=0.794, confirming the problem is non-trivial without the right features.

### Top 5 Most Important Features

| Rank | Feature | Why it matters |
|------|---------|----------------|
| 1 | `is_outlier` | Bots are flagged by 5 combined anomaly detectors |
| 2 | `actor_event_count` | Bots generate far more events |
| 3 | `actor_event_velocity` | Bots act faster (events/day) |
| 4 | `is_integration_event` | Many bots are registered integrations |
| 5 | `event_hour` | Bots active 24/7; humans cluster in business hours |

### Model Comparison

| Criterion | Decision Tree | **Random Forest** | Gradient Boosting | SVM |
|-----------|:---:|:---:|:---:|:---:|
| Accuracy | Perfect | Perfect | Perfect | Perfect |
| Overfitting risk | High | **Low** | Medium | Low |
| Feature importance | Yes | **Yes** | Yes | No |
| Training speed | Fast | **Fast** | Slow | Slow |
| Production readiness | ✗ | **✓** | ✓ | ✗ |

**Recommended for deployment: Random Forest**

---

## Part B — Unsupervised Learning (Clustering)

| Algorithm | Config | Silhouette ↑ | ARI ↑ | Purity ↑ |
|-----------|--------|:---:|:---:|:---:|
| K-Means | k=3 | 0.224 | −0.029 | 0.848 |
| **DBSCAN** | eps=0.3 | **0.997** | 0.002 | **1.000** |
| Agglomerative | k=3, Ward | 0.224 | −0.029 | 0.848 |
| GMM | 8 components | 0.087 | 0.019 | 0.872 |

**Key finding:** No algorithm cleanly recovers the bot/human partition (all ARI ≈ 0) — supervised labels are necessary. DBSCAN identified **60 noise points** = genuine anomalies for security review.

---

---

# Phase III — Retraining & Deployment

---

## Retraining Strategies

### Strategy A — SMOTE Retraining

Physically creates synthetic bot examples by interpolating between existing minority samples:

```
new_sample = anchor + λ × (neighbour − anchor),   λ ∈ [0, 1]
```

**Hyperparameter changes:**
- `n_estimators`: 100 → **200** (more stable predictions)
- `max_depth`: None → **20** (controls overfit on synthetic data)
- `min_samples_leaf`: 1 → **2** (avoids single-synthetic-point leaves)

### Strategy B — Decision Threshold Optimisation

Default threshold = 0.5. The optimal threshold maximising F1 was found by sweeping 200 values between 0.01 and 0.99.

**Optimal threshold: 0.035** — the model is extremely confident, so even a 3.5% bot-probability triggers detection.

---

## Phase III — Results Summary

| Strategy | Accuracy | Precision | Recall | F1 | ROC-AUC |
|----------|----------|-----------|--------|----|---------|
| Baseline RF (Phase II) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RF + SMOTE (Strategy A) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RF + Threshold=0.03 (Strategy B) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

All three strategies confirm robustness. In real-world deployment with unseen bot patterns, SMOTE + threshold tuning provide the safety margin needed.

---

## How to Use the Saved Model

```python
import joblib, json, pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load saved artefacts
model         = joblib.load("phase3_outputs/retrained_model.joblib")
scaler        = joblib.load("phase3_outputs/scaler.joblib")
feature_names = json.load(open("phase3_outputs/feature_names.json"))

# Preprocess new events (same steps as training)
new_data = pd.read_csv("new_github_events.csv")
le = LabelEncoder()
for col in new_data.select_dtypes(include="object").columns:
    new_data[col] = le.fit_transform(new_data[col].astype(str))

X_new    = new_data[feature_names]
X_scaled = scaler.transform(X_new)

# Predict
predictions   = model.predict(X_scaled)          # 0=human, 1=bot
probabilities = model.predict_proba(X_scaled)[:, 1]
print("Bot probability:", probabilities)
```

---

## How to Improve Further

| Improvement | Method |
|-------------|--------|
| More diverse training data | Collect events over a longer time window |
| Real-time behavioural features | Sliding-window velocity (last 1h, 6h, 24h) |
| Better encoding of high-cardinality columns | Target encoding for `actor`, `action` |
| Stronger gradient boosting | XGBoost / LightGBM |
| Temporal validation | Test on future time period, not random split |
| Explainability | SHAP values per prediction |
| Online learning | Incremental model updates as new events arrive |

---

---

# Key Conclusions

1. **Feature engineering is the most important step** — `is_outlier`, `actor_event_velocity`, and `is_integration_event` are what make the problem solvable with near-perfect accuracy.

2. **Random Forest is the optimal algorithm** — it reduces variance through bagging, provides feature importance, and is production-ready.

3. **Perfect scores require temporal validation** — all 8,000 rows span 59 days; in production, the model should be validated on events from a later time period.

4. **Clustering is complementary, not a substitute** — unsupervised methods revealed behavioural subgroups and 60 anomalies, but cannot replace supervised labels for classification.

5. **SMOTE + threshold tuning provide production safety margins** — both retraining strategies are documented, reproducible, and ready for deployment.

---

**Recommended deployment model:** `phase3_outputs/retrained_model.joblib`
**Security-first threshold:** 0.035 (maximise recall, catch every bot)
**Balanced threshold:** 0.500 (reduce false alarms)
**Retraining schedule:** Monthly as new GitHub event data becomes available

---

*Full pipeline: Phase I → Phase II → Phase III*
*Reproducible: `python src/main.py` → `python src/main_phase2.py` → `python src/phase3_retraining.py`*
