# Machine Learning 

> **Course:** Machine Learning  
> **Professor:** Prof. Dr. Lule AHMEDI  
> **Assistant:** Dr. Sc. Mërgim H. HOTI  
> **University:** University of Prishtina "Hasan Prishtina"  
> **Faculty:** Faculty of Electrical and Computer Engineering (FIEK)  
> **Study Level:** Master — Semester II | Academic Year: 2025/26

---

##  Contributors

| Name 
|------|
| `Dredhza Braina` |
| `Suhejla Hoxha` | 
| `Ylljete Kicaj` | 

---

## Project Overview

This project builds a complete **Phase I Data Preparation pipeline** for a GitHub activity-log dataset. The goal is to clean, transform, and prepare the data for training a binary classifier that predicts whether an actor is a **bot or a human** (`actor_is_bot`).

The entire pipeline is written in modular Python (no Jupyter notebooks) as a `src/` package.

---

##  Dataset Details

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

## Repository Structure

```
github-ml-pipeline/
├── unprocessed dataset/
│   └── github.csv
├── processed dataset/
│   └── processed_github.csv
├── eda_plots/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── events_by_hour.png
│   ├── events_by_dayofweek.png
│   ├── top_actions.png
│   ├── pca_plot.png
│   ├── pairplot.png
│   ├── resampling_comparison.png
│   └── ... (17 plots total)
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

# Phase I — Data Preparation

---

## Step 1 — Data Loading & Overview

The dataset was loaded with `pandas.read_csv(low_memory=False)` to correctly handle the 590 columns which contain mixed data types. Shape, a 3-row preview, and all column types were printed immediately after loading.

```
Shape         : 10,000 rows × 590 columns
Unique actions: 481
Time range    : Jan 10, 2026 → Mar 11, 2026 (59 days)
```
---

## Step 2 — Data Quality Analysis

A structured quality report was generated covering missing rates, duplicates, target completeness, and temporal range.

| Metric | Value |
|--------|-------|
| Full-row duplicates | 0 |
| Duplicate `_document_id` values | 9,519 |
| `actor_is_bot` filled | 9,155 / 10,000 (91.55%) |
| Event time span | 59 days |
| Overall missing rate | 94.44% |
| Unique action types | 481 |

**Top 5 most frequent actions:**

| Action | Count |
|--------|-------|
| `pull_request_review_comment.delete` | 47 |
| `codespaces.create` | 46 |
| `workflows.rerun_workflow_run` | 46 |
| `business_secret_scanning_push_protection.enable` | 46 |
| `team.remove_repository` | 43 |

---

## Step 3 — Missing Values Analysis

```
Total missing cells  : 5,572,204 across 575 columns
actor_is_bot missing : 845 rows  (8.45%)
created_at missing   : 118 rows  (1.18%)
```

---

## Step 4 — Missing Values Handling

A three-stage strategy was applied:

1. **Drop columns with >50% NaN** → 565 columns dropped, **25 remain**
2. **Fill categorical / object columns with `"Unknown"`** → 22 columns — treats missing as a valid unknown category rather than an error
3. **Fill numeric columns with the column median** → 1 column — median is robust to outliers unlike the mean

```
Before : 5,572,204 NaN in 575 columns
After  : 118 NaN   (only created_at, 1.18%)
Columns: 590 → 25
```

---

## Step 5 — Data Cleaning

1. **Full-row deduplication** → 0 duplicates found — all 10,000 rows are retained
2. **Column names standardised** → dots and spaces replaced with underscores, all lowercase

> `_document_id` deduplication was intentionally skipped. Investigation showed every `@timestamp` is unique, meaning all 10,000 rows represent genuinely distinct event occurrences. The 9,519 repeated `_document_id` values reflect the batch/session structure of GitHub's audit log export, not true duplicate records.

```
Before cleaning : 10,000 rows × 25 columns
After cleaning  : 10,000 rows × 25 columns  (0 duplicates removed)
```

---

## Step 6 — Dataset Integration & Aggregation

Only one dataset (`github.csv`) is available. 
```

**Aggregations produced:**

| Aggregation | Top Result |
|-------------|-----------|
| Events per actor | "Unknown" → 409 events |
| Events per action type | `pull_request_review_comment.delete` → 47 |
| Events per organisation | "Unknown" → 1,975 events |
| Hourly event volume | Peak at 18:00 on Jan 10 (22 events) |
```
---

## Step 7 — Sampling

An **80% stratified sample** was drawn using `actor_is_bot` as the stratification key, preserving the exact class ratio in the working subset.

```
Before : 10,000 rows
After  :  8,000 rows  (80% stratified sample)
```

---

## Step 8 — Feature Engineering & Transformation

### 8.1 — New Features Created (10 derived features)

| Feature | Source | Why |
|---------|--------|-----|
| `event_hour` | `@timestamp` | Hour of day — humans peak afternoon, bots are uniform |
| `event_dayofweek` | `@timestamp` | Day of week — humans less active on weekends |
| `event_month`, `event_year` | `@timestamp` | Seasonal and temporal trends |
| `is_weekend` | `event_dayofweek` | Binary flag for weekend activity |
| `actor_event_count` | `actor` | Total events per actor in the dataset |
| `actor_bot_ratio` | `actor` + `actor_is_bot` | Fraction of actor's events flagged as bot |
| `actor_active_days` | `actor` + `@timestamp` | Number of distinct active days |
| `actor_event_velocity` | `actor_event_count` / `actor_active_days` | Events per day — bots show abnormally high velocity |
| `is_integration_event` | `integration_id` | 1 if event originates from an automated integration |

### 8.2 — Label Encoding

10 categorical columns with cardinality ≤ 200 were label-encoded using `LabelEncoder`. 12 high-cardinality columns (`actor`, `action`, `_document_id`, etc.) were skipped — to be handled with target encoding or embeddings in Phase II.

### 8.3 — Discretisation & Binarisation

| Column | Method | Output |
|--------|--------|--------|
| `actor_event_count` | Quantile (4 bins) | `actor_event_count_bin`: Low/Medium/High/Very High |
| `event_hour` | Custom bins [-1,6,12,18,23] | `event_hour_bin`: Night/Morning/Afternoon/Evening |
| `actor_is_bot` | Threshold 0.5 | `is_bot_binary` |
| `actor_event_count` | Median threshold (19.0) | `is_high_activity_actor` |
| `is_weekend` | Threshold 0.5 | `is_weekend_event` |

### 8.4 — Scaling & Transforms

| Transform | Applied to | Reason |
|-----------|-----------|--------|
| `StandardScaler` | 21 numeric columns | Required by PCA, SVM, Logistic Regression |
| `RobustScaler` | `actor_event_count` | Robust to extreme outliers in event-count distributions |
| `log1p` | `actor_event_count`, `actor_event_velocity` | Reduces extreme right-skewness |
| `sqrt` | `event_hour` | Mild transformation for moderate skewness |

---

## Step 9 — Dimensionality Reduction

### PCA (Principal Component Analysis)

```
Input  : 48 numeric features
Output : 15 principal components
Explained variance retained: 96.7%
```

| Component | Variance | Cumulative |
|-----------|----------|------------|
| PC1 | 17.21% | 17.21% |
| PC2 | 12.19% | 29.40% |
| PC3 | 10.87% | 40.27% |
| PC4 | 9.40% | 49.67% |
| PC5 | 7.27% | 56.94% |
| PC6 | 7.06% | 64.00% |
| PC7 | 6.25% | 70.25% |
| PC8 | 4.77% | 75.02% |
| PC9 | 4.51% | 79.53% |
| PC10 | 3.66% | 83.19% |

### Univariate Feature Selection (f_classif)

Top 20 features selected by ANOVA F-test against `actor_is_bot`:

`business`, `server_id`, `integration_id`, `event_month`, `actor_event_count`, `actor_bot_ratio`, `actor_active_days`, `is_bot_binary`, and their scaled variants + `actor_event_count_robust`, `actor_event_count_log`

---

## Step 10 — Exploratory Data Analysis

The `EDAAnalyzer` class produced **17 plots** saved to `eda_plots/`.

### Summary Statistics (key numeric columns)

| Column | Mean | Std | Skewness | Notes |
|--------|------|-----|----------|-------|
| `actor_is_bot` | 0.152 | 0.359 | 1.94 | Heavily imbalanced |
| `business` | 6.72 | 4.93 | 0.39 | Approx. uniform |
| `operation_type` | 3.83 | 1.05 | −1.35 | Left-skewed |
| `pull_request_id` | 25.26 | 3.51 | −5.17 | Heavy left tail |
| `hook_id` | 4.96 | 0.36 | −10.98 | Nearly constant |

### Top Correlations

| Feature A | Feature B | Correlation |
|-----------|-----------|-------------|
| `actor_is_bot` | `actor_bot_ratio` | +1.000 |
| `actor_is_bot` | `is_bot_binary` | +1.000 |
| `actor_event_count` | `actor_active_days` | +1.000 |
| `event_dayofweek` | `is_weekend` | +0.792 |
| `event_dayofweek` | `is_weekend_event` | +0.792 |

![Correlation Heatmap](eda_plots/correlation_heatmap.png)

> Perfect correlations between `actor_is_bot`, `actor_bot_ratio`, and `is_bot_binary` are expected — they are derived from the same source. These redundant features will be handled during Phase II feature selection.

### Temporal Distribution

![Events by Hour](eda_plots/events_by_hour.png)

![Events by Day of Week](eda_plots/events_by_dayofweek.png)

> Activity peaks in afternoon hours and drops on weekends — a pattern consistent with human-driven activity. Bots would show a flatter distribution.

### Top Actions

![Top Actions](eda_plots/top_actions.png)

### PCA Scatter

![PCA Plot](eda_plots/pca_plot.png)

---

## Step 11 — Class Imbalance Detection

| Class | Count | % |
|-------|-------|---|
| 0 — Human | 6,786 | 84.82% |
| 1 — Bot | 1,214 | 15.17% |

**Imbalance ratio: 5.6 : 1**

A naive classifier that always predicts "human" would reach 84.82% accuracy without learning anything useful. The bot class — which is the class of interest for security purposes — would have near-zero recall. SMOTE and ADASYN are applied to correct this before Phase II training.

![Class Distribution](eda_plots/class_distribution.png)

---

## Step 12 — SMOTE & ADASYN

Both algorithms were implemented **from scratch** without any dependency on `imblearn`, enabling execution in offline environments.

**SMOTE** generates synthetic minority samples by linearly interpolating between a minority instance and one of its k nearest neighbours:
`new_sample = anchor + λ × (neighbour − anchor)`, where λ ∈ [0, 1]

**ADASYN** is adaptive — it computes a difficulty score per minority instance (proportion of majority-class neighbours) and generates more synthetic samples for harder-to-classify instances near the decision boundary.

| Method | Before | After |
|--------|--------|-------|
| **SMOTE** | Human: 6,786 · Bot: 1,214 | Human: 6,786 · Bot: 6,786 |
| **ADASYN** | Human: 6,786 · Bot: 1,214 | Human: 6,786 · Bot: 7,284 |

> ADASYN produced slightly more minority samples than SMOTE (7,284 vs 6,786) due to its adaptive weighting of boundary instances. Both balanced datasets will be evaluated in Phase II.

![Resampling Comparison](eda_plots/resampling_comparison.png)

---

## Step 13 — Outlier Detection

Five detection methods were applied and combined into a composite score:

| Method | Outliers Found | Type |
|--------|---------------|------|
| IQR | 7,307 | Univariate |
| Z-Score | 2,742 | Univariate |
| Isolation Forest | 399 | Multivariate |
| Local Outlier Factor (LOF) | 395 | Multivariate |
| Rare Categories (`action`) | 8,000 | Categorical |
| Rare Categories (`operation_type`) | 190 | Categorical |

**Composite outlier type distribution:**

| Type | Count | % |
|------|-------|---|
| mild (1 method agrees) | 4,971 | 62.1% |
| strong (2 methods agree) | 330 | 4.1% |
| extreme (3+ methods agree) | 2,699 | 33.7% |

**After requiring ≥2 methods to agree:**

```
Total detected     : 8,000
False positives    : 5,301  (66.3%)
Confirmed outliers : 2,699  (33.7%)
```

Confirmed outliers are flagged via the `is_outlier` column in the final dataset rather than removed, preserving all rows for model training.

---

## Step 14 — Final Dataset

**20 features selected for the ML model:**

`action`, `actor`, `actor_id`, `actor_is_bot` *(target)*, `operation_type`, `repo`, `repo_id`, `org`, `org_id`, `user_agent`, `event_hour`, `event_dayofweek`, `event_month`, `event_year`, `is_weekend`, `actor_event_count`, `actor_bot_ratio`, `actor_event_velocity`, `is_integration_event`, `is_outlier`

| Metric | Value |
|--------|-------|
| Rows | 8,000 |
| Columns | 20 |
| NaN values | 0 |
| Target distribution | 84.8% human / 15.2% bot |
| Output file | `processed dataset/processed_github.csv` |

---

## How to Run

```bash
# Install dependencies
pip install -r src/requirements.txt

# Place the dataset
mkdir "unprocessed dataset"
copy github.csv "unprocessed dataset\github.csv"

# Run the pipeline
cd src
python main.py
```

---

## Requirements

```
pandas>=2.0 | numpy>=1.24 | scikit-learn>=1.3 | scipy>=1.11 | matplotlib>=3.7 | seaborn>=0.12
```

---

## Phase I — Results Summary

| Step | Input | Output | Key Result |
|------|-------|--------|------------|
| 1. Loading | Raw CSV | DataFrame | 10,000 × 590, 94.44% NaN |
| 2. Quality | DataFrame | Report | 9,519 duplicate IDs detected |
| 3. NaN Analysis | 575 NaN columns | Statistics | Top 100% empty columns identified |
| 4. NaN Handling | 590 columns | 25 columns | 565 dropped, filled Unknown/median |
| 5. Cleaning | 10,000 rows | 10,000 rows | 0 true duplicates found |
| 6. Integration | 1 source | Aggregations | 4 aggregation views produced |
| 7. Sampling | 10,000 rows | 8,000 rows | 80% stratified by `actor_is_bot` |
| 8. Feature Eng. | 25 columns | 65 columns | 10 new features, 21 scaled |
| 9. PCA | 48 features | 15 components | 96.7% variance retained |
| 10. EDA | DataFrame | 17 plots | Temporal patterns identified |
| 11. Imbalance | 5.6:1 ratio | Documented | SMOTE/ADASYN applied |
| 12. SMOTE/ADASYN | Bot: 1,214 | Bot: 6,786 / 7,284 | Classes balanced |
| 13. Outliers | 8,000 rows | 2,699 confirmed | Flagged in `is_outlier` column |
| 14. **Final output** | — | **8,000 × 20** | **Zero NaN, ML-ready** |

---

---

# Phase II — Model Training & Analysis

---

## Overview

Phase II builds on the ML-ready dataset produced by Phase I (`processed dataset/processed_github.csv`, 8,000 rows × 20 columns) and applies:

- **Part A — Supervised Learning Enhancements & Detailed Analysis**: 6 classifiers with hyperparameter tuning, cross-validation, learning curves, feature importance, ROC curves, confusion matrices.
- **Part B — Unsupervised Learning & Clustering Analysis**: 4 clustering algorithms with cluster quality metrics, PCA scatter plots, cluster profiling, dendrogram, BIC analysis, and comparison with ground-truth labels.

---

## Part A — Supervised Learning

### Models Trained

| # | Model | Tuning Strategy |
|---|-------|----------------|
| 1 | Logistic Regression | GridSearchCV (C, solver, penalty) |
| 2 | Decision Tree | GridSearchCV (max_depth, min_samples_leaf, criterion) |
| 3 | Random Forest | RandomizedSearchCV (n_estimators, max_depth, max_features) |
| 4 | Gradient Boosting | RandomizedSearchCV (n_estimators, learning_rate, max_depth, subsample) |
| 5 | SVM (RBF kernel) | GridSearchCV (C, gamma) |
| 6 | K-Nearest Neighbours | GridSearchCV (n_neighbors, weights, metric) |

All models used:
- **Stratified 80/20 train-test split** (`random_state=42`)
- **5-fold Stratified Cross-Validation** for both tuning and final reporting
- **`class_weight="balanced"`** (where supported) to handle 84.8% / 15.2% imbalance
- **StandardScaler** applied before Logistic Regression, SVM, and KNN

### Step A.1 — Data Preparation

- Loaded `processed_github.csv` (8,000 rows × 20 columns)
- Dropped `actor_bot_ratio` — perfectly correlated with the target (data leakage)
- Label-encoded remaining object columns (`action`, `actor`, `repo`, `org`, `actor_id`, `org_id`, `user_agent`)
- Final feature matrix: **8,000 rows × 18 features**
- Target distribution: Human 6,786 (84.83%) | Bot 1,214 (15.17%)

### Step A.2 — Hyperparameter Tuning

Each model tuned with `scoring="f1"` to prioritise minority-class (bot) detection.

| Model | Best Parameters |
|-------|----------------|
| Logistic Regression | C=10, solver=liblinear, penalty=l2 |
| Decision Tree | criterion=gini, max_depth=None, min_samples_leaf=1 |
| Random Forest | n_estimators=100, max_features=sqrt, max_depth=None |
| Gradient Boosting | n_estimators=100, learning_rate=0.2, max_depth=3, subsample=0.8 |
| SVM | C=10, kernel=rbf, gamma=scale |
| K-Nearest Neighbours | n_neighbors=5, weights=distance, metric=euclidean |

### Step A.3 — Evaluation Results

#### Test-Set Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.9213 | 0.6585 | 1.0000 | 0.7941 | 0.9642 |
| Decision Tree | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Random Forest | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Gradient Boosting | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| SVM | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| K-Nearest Neighbours | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

#### 5-Fold Cross-Validation (on training set)

| Model | CV F1 (mean ± std) | CV ROC-AUC (mean ± std) |
|-------|-------------------|------------------------|
| Logistic Regression | 0.7655 ± 0.0144 | 0.9623 ± 0.0028 |
| Decision Tree | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Random Forest | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Gradient Boosting | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| SVM | 0.9985 ± 0.0021 | 1.0000 ± 0.0000 |
| K-Nearest Neighbours | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

> **Note on perfect scores:** Phase I produced exceptionally strong derived features — `is_outlier`, `actor_event_count`, `actor_event_velocity`, and `is_integration_event` — which create near-perfect separation between bots and humans. Tree-based and distance-based models exploit these non-linear combinations immediately, achieving 100% test and CV scores. Logistic Regression (linear decision boundary) serves as the most informative interpretable baseline with a realistic F1 of 0.7941.

![Metrics Comparison](phase2_plots/metrics_comparison.png)

### Step A.4 — Feature Importance

Top discriminating features across tree-based models:

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `is_outlier` | Composite outlier flag from Phase I (5 detection methods) |
| 2 | `actor_event_count` | Total events per actor — bots are very high-activity |
| 3 | `actor_event_velocity` | Events per active day — bots show abnormally high velocity |
| 4 | `is_integration_event` | 1 if event originated from an automated integration |
| 5 | `event_hour` | Hour of day — bots active 24/7, humans cluster in business hours |

Charts: `phase2_plots/fi_*.png`

![Feature Importance – Random Forest](phase2_plots/fi_Random_Forest.png)

![Feature Importance – Gradient Boosting](phase2_plots/fi_Gradient_Boosting.png)

- **Tree / ensemble / KNN models**: Both training and validation F1 converge to 1.0 from early training sizes — even 10% of the training data is sufficient.
- **SVM**: Validation F1 converges quickly (≈0.99) due to a strong margin in scaled feature space.
- **Logistic Regression**: Larger train-validation gap — linear boundary struggles with non-linear feature interactions. Validation F1 plateaus at ≈0.77.

Charts: `phase2_plots/lc_*.png`

![Learning Curve – Logistic Regression](phase2_plots/lc_Logistic_Regression.png)

![Learning Curve – Random Forest](phase2_plots/lc_Random_Forest.png)

All models except Logistic Regression achieve AUC = 1.000. Logistic Regression achieves AUC = 0.964 — good probability ranking even when the decision boundary is not optimal.

![ROC Curves – All Models](phase2_plots/roc_all_models.png)

### Step A.7 — Confusion Matrices

| Model | TP (Bot→Bot) | FN (Bot→Human) | FP (Human→Bot) | TN (Human→Human) |
|-------|-------------|----------------|----------------|------------------|
| Logistic Regression | 243 | 0 | 123 | 1,234 |
| Decision Tree | 243 | 0 | 0 | 1,357 |
| Random Forest | 243 | 0 | 0 | 1,357 |
| Gradient Boosting | 243 | 0 | 0 | 1,357 |
| SVM | 243 | 0 | 0 | 1,357 |
| K-Nearest Neighbours | 243 | 0 | 0 | 1,357 |

Logistic Regression achieves **zero false negatives** (no bots missed) at the cost of 123 false positives. For a security use case — where missing a bot is costlier than a false alarm — this trade-off is acceptable.

![Confusion Matrix – Logistic Regression](phase2_plots/cm_Logistic_Regression.png)

![Confusion Matrix – Random Forest](phase2_plots/cm_Random_Forest.png)

### Part A Summary

| Criterion | Winner | Notes |
|-----------|--------|-------|
| Best F1 / Accuracy | Decision Tree, RF, GB, SVM, KNN | All perfect on this dataset |
| Most interpretable | Decision Tree | Can be visualised as human-readable rules |
| Most robust (ensemble) | **Random Forest** | Reduces variance through bagging |
| Best for weak signals | Logistic Regression | Most honest under noisy / incomplete features |
| **Recommended for deployment** | **Random Forest** | Perfect performance + ensemble robustness + feature importance |

---

## Part B — Unsupervised Learning & Clustering

### Algorithms Applied

| # | Algorithm | Key Hyperparameter Strategy |
|---|-----------|---------------------------|
| 1 | K-Means | k selected by silhouette score curve (k = 2..10) |
| 2 | DBSCAN | eps & min_samples selected by grid search on silhouette |
| 3 | Agglomerative Clustering (Ward) | n_clusters selected by silhouette (n = 2..6) + dendrogram |
| 4 | Gaussian Mixture Model (GMM) | n_components selected by BIC criterion (n = 2..8) |

All algorithms operated on the full 18-dimensional scaled feature space. PCA to 2 components was used **only** for scatter plot visualisation.

**Evaluation metrics used:**

| Metric | Direction | Description |
|--------|-----------|-------------|
| Silhouette Score | ↑ higher = better | Intra-cluster cohesion vs. inter-cluster separation |
| Davies-Bouldin Index | ↓ lower = better | Ratio of within-cluster scatter to between-cluster separation |
| Calinski-Harabasz Index | ↑ higher = better | Ratio of between/within cluster dispersion |
| ARI (Adjusted Rand Index) | ↑ higher = better | Alignment of clusters with ground-truth bot/human labels |
| NMI (Normalized Mutual Information) | ↑ higher = better | Information-theoretic alignment with ground truth |
| Purity | ↑ higher = better | Fraction of cluster members belonging to the dominant class |

### Step B.1 — K-Means

```
Best k             : 3  (selected by maximum silhouette score, k = 2..10)
Silhouette Score   : 0.2240
Davies-Bouldin     : 1.5804
Calinski-Harabasz  : 1050.94
ARI                : −0.029
NMI                : 0.014
Purity             : 0.848
```

The **elbow plot** shows inertia decreasing rapidly from k=2 to k=4, then flattening. The **silhouette curve** confirms k=3. A silhouette of 0.22 indicates overlapping clusters — expected for this high-dimensional, sparse dataset.

**Cluster profile**: Cluster 0 maps predominantly to bot-like behaviour (high `actor_event_count`, high `is_integration_event`). Clusters 1 and 2 capture different human activity profiles (weekend vs. weekday, low vs. medium activity volume).

Low ARI (−0.029) and NMI (0.014) confirm that K-Means does not cleanly recover the bot/human partition — the two-class boundary is not spherical, and K-Means assumes spherical clusters.

![K-Means Elbow & Silhouette](phase2_plots/kmeans_elbow.png)

![K-Means PCA Scatter](phase2_plots/kmeans_scatter.png)

![K-Means Cluster Sizes](phase2_plots/kmeans_sizes.png)

![K-Means Cluster Profile](phase2_plots/kmeans_profile.png)

### Step B.2 — DBSCAN

```
Best eps           : 0.3,  min_samples = 5
Clusters found     : 464
Noise points       : 60  (0.75% of data)
Silhouette Score   : 0.9970  ← highest of all algorithms
Davies-Bouldin     : 0.0316  ← lowest (best) of all algorithms
Calinski-Harabasz  : 440,076
ARI                : 0.0016
NMI                : 0.1317  ← highest of all algorithms
Purity             : 1.0000
```

DBSCAN finds 464 tight micro-clusters with near-perfect internal cohesion. Purity = 1.0 means every micro-cluster is purely bot or purely human — no mixing within a cluster. The low ARI (0.002) indicates these micro-clusters reflect **actor-specific behavioural fingerprints** rather than the broad bot/human taxonomy.

The **60 noise points** are genuine outliers — actors whose event patterns do not fit any dense neighbourhood. These are prime candidates for manual security review.

![DBSCAN PCA Scatter](phase2_plots/dbscan_scatter.png)

![DBSCAN Cluster Sizes](phase2_plots/dbscan_sizes.png)

![DBSCAN Cluster Profile](phase2_plots/dbscan_profile.png)

### Step B.3 — Agglomerative (Hierarchical) Clustering

```
Linkage method     : Ward (minimises total within-cluster variance)
Best n_clusters    : 3  (same as K-Means)
Silhouette Score   : 0.2240
Davies-Bouldin     : 1.5804
ARI                : −0.029
NMI                : 0.014
Purity             : 0.848
```

The **dendrogram** (300-point sample, Ward linkage) shows two major arms with one further subdivision — confirming that 3 clusters is a natural cut-point. Results are nearly identical to K-Means: Ward linkage is the hierarchical analogue of K-Means (both minimise within-cluster variance). This consistency confirms the 3-cluster structure is **stable** and not an artefact of K-Means initialisation.

![Agglomerative Dendrogram](phase2_plots/agglo_dendrogram.png)

![Agglomerative PCA Scatter](phase2_plots/agglo_scatter.png)

![Agglomerative Cluster Sizes](phase2_plots/agglo_sizes.png)

![Agglomerative Cluster Profile](phase2_plots/agglo_profile.png)

### Step B.4 — Gaussian Mixture Model (GMM)

```
Best n_components  : 8  (selected by minimum BIC)
Silhouette Score   : 0.0868
Davies-Bouldin     : 2.2772
ARI                : 0.019
NMI                : 0.099
Purity             : 0.872
```

The **BIC curve** decreases through n=8, suggesting the data is better modelled as 8 Gaussian components than 2 or 3 — reflecting heterogeneity within each class: different types of bots (CI/CD, API integrations, automated scanners) and different human profiles (admins, developers, occasional users).

![GMM BIC Curve](phase2_plots/gmm_bic.png)

![GMM PCA Scatter](phase2_plots/gmm_scatter.png)

![GMM Cluster Sizes](phase2_plots/gmm_sizes.png)

![GMM Cluster Profile](phase2_plots/gmm_profile.png)

### Step B.5 — Clustering Algorithm Comparison

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | ARI ↑ | NMI ↑ | Purity ↑ |
|-----------|-------------|-----------------|-------|-------|---------|
| K-Means | 0.224 | 1.580 | −0.029 | 0.014 | 0.848 |
| **DBSCAN** | **0.997** | **0.032** | 0.002 | **0.132** | **1.000** |
| Agglomerative | 0.224 | 1.580 | −0.029 | 0.014 | 0.848 |
| GMM | 0.087 | 2.277 | 0.019 | 0.099 | 0.872 |

Chart: `phase2_plots/cluster_comparison.png`

**Key finding:** No algorithm cleanly recovers the bot/human partition via unsupervised clustering alone (all ARI ≈ 0). The bot/human boundary is a *learned* decision boundary that requires supervised signal — validating the Phase II design choice of treating this as supervised classification.

![Clustering Algorithm Comparison](phase2_plots/cluster_comparison.png)

### Part B Summary

| Goal | Best Algorithm | Reasoning |
|------|---------------|-----------|
| Discover natural behavioural groups | K-Means / Agglomerative | Consistent 3-cluster structure under both methods |
| Anomaly / outlier detection | **DBSCAN** | 60 noise points identify genuinely unusual actors |
| Probabilistic sub-population modelling | **GMM (8 components)** | Best BIC; models heterogeneous actor sub-types |
| Recovering bot/human classes | None | Unsupervised methods cannot substitute for supervised labels |

---

## Phase II — Full Results Summary

| Step | Task | Key Result |
|------|------|-----------|
| A.1 | Data prep | 8,000 × 18 features, leakage removed |
| A.2 | Hyperparameter tuning | Best params via GridSearch/RandomSearch with 5-fold CV |
| A.3 | Test evaluation | 5/6 models F1=1.0; Logistic Regression F1=0.794, AUC=0.964 |
| A.4 | Feature importance | `is_outlier`, `actor_event_count`, `actor_event_velocity` dominate |
| A.5 | Learning curves | Tree/ensemble converge with 10% data; LR shows underfitting |
| A.6 | ROC curves | All models AUC ≥ 0.964; 5 achieve AUC = 1.0 |
| A.7 | Confusion matrices | LR: 0 FN, 123 FP; all others: 0 FN, 0 FP |
| B.1 | K-Means | 3 clusters, silhouette=0.224, purity=0.848 |
| B.2 | DBSCAN | 464 micro-clusters, silhouette=0.997, 60 anomalies identified |
| B.3 | Agglomerative | 3 clusters (Ward), silhouette=0.224, consistent with K-Means |
| B.4 | GMM | 8 components (BIC), silhouette=0.087, sub-population profiling |
| **Best supervised** | **Random Forest** | F1=1.0, CV F1=1.0 ± 0.0, robust ensemble |
| **Best clustering (internal quality)** | **DBSCAN** | Highest silhouette, lowest Davies-Bouldin |
| **Best clustering (anomaly detection)** | **DBSCAN** | 60 noise points = genuine behavioural anomalies |
