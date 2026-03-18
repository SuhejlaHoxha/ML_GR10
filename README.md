# 🔍 GitHub Activity Log — Machine Learning Pipeline

> **Course:** Machine Learning  
> **Professor:** Prof. Dr. Lule AHMEDI  
> **Assistant:** Dr. Sc. Mërgim H. HOTI  
> **University:** University of Prishtina "Hasan Prishtina"  
> **Faculty:** Faculty of Electrical and Computer Engineering (FIEK)  
> **Study Level:** Master — Semester II | Academic Year: 2025/26

---

## 👥 Contributors

| Name 
|------|
| `Dredhza Braina` |
| `Suhejla Hoxha` | 
| `Ylljete Kicaj` | 

---

## 📌 Project Overview

This project builds a complete **Phase I Data Preparation pipeline** for a GitHub activity-log dataset. The goal is to clean, transform, and prepare the data for training a binary classifier that predicts whether an actor is a **bot or a human** (`actor_is_bot`).

The entire pipeline is written in modular Python (no Jupyter notebooks) as a `src/` package.

---

## 🗃️ Dataset Details

| Attribute | Value |
|-----------|-------|
| **Name** | GitHub Activity Log |
| **Original format** | JSON (flattened to CSV) |
| **Rows** | 10,000 |
| **Columns** | 590 |
| **Size** | ~45 MB in memory |
| **Source** | GitHub Audit Log API |
| **Target variable** | `actor_is_bot` (0 = human, 1 = bot) |
| **Time period** | Jan 10, 2026 — Mar 11, 2026 (59 days) |
| **Unique event types** | 481 |

**Key columns:** `@timestamp`, `_document_id`, `action`, `actor`, `actor_is_bot`, `operation_type`, `org`, `repo`, `visibility`, `actor_ip`

**Notable characteristics:**
- Only **4.36%** of all cells are filled — extremely sparse (flattened JSON)
- Class imbalance: **84.95% human** vs **15.05% bot**
- 572 columns have 100% missing values (event-specific fields)

---

## 📂 Repository Structure

```
github-ml-pipeline/
├── unprocessed dataset/
│   └── github.csv
├── processed dataset/
│   └── processed_github.csv
├── eda_plots/
├── src/
│   ├── main.py
│   ├── data_collection.py
│   ├── data_quality.py
│   ├── cleaning.py
│   ├── integration.py
│   ├── advanced_preprocessing.py
│   ├── eda_analyzer.py
│   ├── class_balancing.py
│   ├── outlier_detection.py
│   └── requirements.txt
└── README.md
```

---

---

# 📋 Phase I — Data Preparation

---

## Step 1 — Data Loading & Overview

The dataset was loaded with `pandas.read_csv(low_memory=False)` to handle mixed-type columns. Shape, preview, and column types were printed immediately after loading.

```
Shape            : 10,000 rows × 590 columns
Completeness     : 4.36% (95.64% of cells are NaN)
Unique actions   : 481
```

Type-definition rules applied: `@timestamp` converted from epoch-ms to datetime; `action`, `operation_type`, `visibility` cast to `category`; identifier columns kept as `string`.

---

## Step 2 — Data Quality Analysis

A structured quality report was generated covering duplicates, missing rates, target completeness, and temporal range.

| Metric | Value |
|--------|-------|
| Full-row duplicates | 0 |
| Duplicate event IDs (`_document_id`) | 9,519 |
| `actor_is_bot` filled | 9,155 / 10,000 (91.55%) |
| Event time span | 59 days |
| Overall missing rate | 95.64% |

> The 9,519 duplicate event IDs indicate the same events were recorded by multiple webhook receivers — deduplication by `_document_id` is essential.

---

## Step 3 — Missing Values Analysis

587 out of 590 columns contain at least one NaN. The top columns are 100% empty (e.g. `config`, `permissions_added`, `vulnerability_alert_rule_*`) — these are event-specific fields that simply don't apply to most event types.

```
Total missing cells : 5,642,789 across 587 columns
actor_is_bot missing: 845 rows (8.45%)
created_at missing  : 118 rows (1.18%)
```

---

## Step 4 — Missing Values Handling

A three-stage strategy was applied:

1. **Drop columns with >50% NaN** → 572 columns dropped, 18 remain
2. **Fill categorical columns with `"Unknown"`** → preserves the category as a valid label
3. **Fill numeric columns with the column median** → robust to outliers unlike the mean

```
Before : 5,642,789 NaN in 587 columns
After  : 118 NaN (only created_at)
Columns: 590 → 18
```

---

## Step 5 — Data Cleaning

1. **Semantic deduplication** → 9,519 repeated observations removed, leaving **481 unique events**
2. **Column names standardised** → dots and spaces replaced with underscores, all lowercase

**Deduplication key:** `action + actor + actor_id + actor_is_bot + org + repo + operation_type + visibility`

**Why semantic deduplication, not `_document_id`:**
Investigation revealed that all 10,000 timestamps are unique — so the rows are not byte-for-byte identical. However, every combination of meaningful attributes repeats ~20 times, with only the timestamp shifting by milliseconds. The dataset is a synthetic audit log where each of the 481 distinct event types was replicated ~20 times. Deduplicating on `_document_id` alone was wrong (it is a session/batch key, not a per-event ID). Deduplicating on the semantic key correctly reduces the dataset to 481 genuinely distinct events, avoiding model training on near-identical rows that would inflate performance metrics.

```
Before cleaning : 10,000 rows × 18 columns
After cleaning  :    481 rows × 18 columns  (481 unique semantic events)
```

---

## Step 6 — Dataset Integration

Only one dataset (`github.csv`) is available — **merging is not required**. This is explicitly stated in the pipeline output. If a second source existed (e.g. an actor-profile CSV), it would be integrated via:

```python
merged = df.merge(actor_profiles, left_on="actor", right_on="username", how="left")
```

**Aggregations produced:**

| Aggregation | Result |
|-------------|--------|
| Events per actor | Top actor: "Unknown" with 22 events |
| Events per action type | 481 types, ~1 event each |
| Events per organisation | 96 events from unknown org |
| Hourly event volume | Visible activity peaks in afternoon hours |

---

## Step 7 — Sampling

An **80% stratified sample** was drawn using `actor_is_bot` as the stratification key to preserve the class ratio.

```
Before : 481 rows
After  : 384 rows
```

Stratified sampling was chosen over random sampling to guarantee that the bot/human proportion is maintained in the working subset.

---

## Step 8 — Feature Engineering & Transformation

### New Features Created

| Feature | Source | Why |
|---------|--------|-----|
| `event_hour`, `event_dayofweek`, `event_month`, `event_year`, `is_weekend` | `@timestamp` | Bots operate uniformly; humans show time-of-day patterns |
| `actor_event_count` | `actor` | Total activity per actor |
| `actor_event_velocity` | `actor` + `@timestamp` | Events per active day — bots show abnormally high velocity |
| `is_programmatic` | `programmatic_access_type` | API/token access is a strong bot indicator |

### Encoding

5 low-cardinality categorical columns (≤200 unique values) were label-encoded. 10 high-cardinality columns (e.g. `actor`, `action`) were skipped — to be handled with target encoding or embeddings in Phase II.

### Discretisation

`event_hour` was discretised into four named buckets: Night / Morning / Afternoon / Evening. Binary flags `is_high_activity_actor` and `is_weekend_event` were created via threshold binarisation.

### Scaling

- `StandardScaler` applied to all numeric columns (required for PCA, SVM, Logistic Regression)
- `RobustScaler` applied to skew-sensitive columns (`actor_event_count`, `actor_event_velocity`)
- `log1p` and `sqrt` transforms applied to reduce skewness

---

## Step 9 — Dimensionality Reduction

### PCA

```
Input  : 31 numeric columns
Output : 9 principal components
Explained variance retained: 97.9%
```

| Component | Variance | Cumulative |
|-----------|----------|------------|
| PC1 | 25.18% | 25.18% |
| PC2 | 22.61% | 47.79% |
| PC3 | 12.64% | 60.43% |
| PC4–PC9 | remaining | 97.9% total |

### Univariate Selection (f_classif)

Top 20 features were selected by ANOVA F-test against the target variable.

---

## Step 10 — Exploratory Data Analysis

The `EDAAnalyzer` class produced 16+ plots saved to `eda_plots/`.

**Summary statistics (event_hour):** Mean = 14.26, Std = 4.99, Skewness = −1.03 → activity peaks in the afternoon, consistent with human behaviour.

**14.6%** of events occurred on weekends — lower than expected for bots if they were truly uniform.

![Events by Hour](eda_plots/events_by_hour.png)

![Correlation Heatmap](eda_plots/correlation_heatmap.png)

**Top 3 correlations (non-trivial):**
- `actor_event_count` ↔ `actor_active_days`: strong positive (expected)
- `is_weekend` ↔ `is_weekend_event`: identical after binarisation
- `event_month` ↔ `event_year`: constant (all 2026 data)

![PCA Plot](eda_plots/pca_plot.png)

---

## Step 11 — Class Imbalance Detection

| Class | Count | % |
|-------|-------|---|
| 0 — Human | 6,796 | 84.95% |
| 1 — Bot | 1,204 | 15.05% |

**Imbalance ratio: 5.6 : 1**

A naive classifier predicting "always human" would achieve 85% accuracy without learning anything. SMOTE and ADASYN are applied to address this before model training.

![Class Distribution](eda_plots/class_distribution.png)

---

## Step 12 — SMOTE & ADASYN

Both algorithms were implemented **from scratch** (no `imblearn` dependency).

**SMOTE** generates synthetic minority samples by interpolating between a minority instance and one of its k nearest neighbours: `new = anchor + λ × (neighbour − anchor)`.

**ADASYN** adapts the sampling density — instances near the decision boundary (surrounded by majority neighbours) receive proportionally more synthetic samples.

| | Before | After |
|-|--------|-------|
| **SMOTE** | {Human: 6,796 · Bot: 1,204} | {Human: 6,796 · Bot: 6,796} |
| **ADASYN** | {Human: 6,796 · Bot: 1,204} | {Human: 6,796 · Bot: 6,794} |

![Resampling Comparison](eda_plots/resampling_comparison.png)

Both results will be compared during Phase II model evaluation.

---

## Step 13 — Outlier Detection

Five methods were applied and combined into a composite score:

| Method | Outliers Found | Type |
|--------|---------------|------|
| IQR | 258 | Univariate |
| Z-Score | 124 | Univariate |
| Isolation Forest | 20 | Multivariate |
| Local Outlier Factor | 20 | Multivariate |
| Rare Categories | 384 | Categorical |

**Composite outlier types:**

| Type | Count | % |
|------|-------|---|
| mild (1 method) | 272 | 70.8% |
| strong (2 methods) | 51 | 13.3% |
| extreme (3+ methods) | 61 | 15.9% |

After requiring **≥2 methods to agree**, 60 outliers were confirmed (false-positive rate reduced from 100% to 15.6%). These are flagged via the `is_outlier` column in the final dataset.

---

## Step 14 — Final Dataset

**Selected features:**
`action`, `actor`, `actor_id`, `operation_type`, `visibility`, `repo`, `repo_id`, `org`, `org_id`, `user_agent`, `event_hour`, `event_dayofweek`, `event_month`, `event_year`, `is_weekend`, `actor_event_count`, `actor_event_velocity`, `is_outlier`

| Metric | Value |
|--------|-------|
| Rows | 384 |
| Columns | 17 |
| NaN values | 0 |
| Output | `processed dataset/processed_github.csv` |

---

## ▶️ How to Run

```bash
# Install dependencies
pip install -r src/requirements.txt

# Place the dataset
mkdir -p "unprocessed dataset"
cp github.csv "unprocessed dataset/github.csv"

# Run the pipeline
python src/main.py
```

---

## 🔧 Requirements

```
pandas>=2.0 | numpy>=1.24 | scikit-learn>=1.3 | scipy>=1.11 | matplotlib>=3.7 | seaborn>=0.12
```

---

## 📊 Phase I Summary

| Step | Input | Output |
|------|-------|--------|
| Loading | Raw CSV | 10,000 × 590 DataFrame |
| Quality | DataFrame | 9,519 semantic duplicates detected |
| NaN Analysis | 587 NaN columns | Top columns identified |
| NaN Handling | 590 columns | 18 columns, 0 NaN |
| Cleaning | 10,000 rows | 481 unique events (semantic dedup) |
| Integration | 1 source | Aggregations produced |
| Sampling | 481 rows | 384 rows (80%, stratified) |
| Feature Eng. | 17 columns | 26 columns (+9 new features) |
| Scaling | Raw numerics | Standardised + log/sqrt |
| PCA | 31 columns | 9 components (97.9% variance) |
| EDA | DataFrame | 16+ charts |
| Imbalance | 5.6:1 ratio | SMOTE/ADASYN applied |
| Outliers | 384 rows | 60 confirmed outliers |
| **Final output** | — | **384 × 17, zero NaN, ML-ready** |

---

*Machine Learning — Master FIEK, University of Prishtina — 2025/26*
