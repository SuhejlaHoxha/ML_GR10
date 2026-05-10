import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report, precision_recall_curve,
    average_precision_score
)

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
TARGET_COL  = "actor_is_bot"
LEAKAGE_COLS = ["actor_bot_ratio"]   

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH    = os.path.join(ROOT_DIR, "processed dataset", "processed_github.csv")
OUT_DIR     = os.path.join(ROOT_DIR, "phase3_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def load_data(csv_path: str):
    print("\n[1/8] Loading dataset …")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    df = df.fillna(df.median(numeric_only=True))

    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    print(f"    Dataset shape   : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"    Class balance   : Human={int((y==0).sum()):,}  Bot={int((y==1).sum()):,}")
    print(f"    Imbalance ratio : {(y==0).sum()/(y==1).sum():.1f}:1")
    return X, y, list(X.columns)

def smote_oversample(X: np.ndarray, y: np.ndarray,
                     k: int = 5, random_state: int = RANDOM_SEED) -> tuple:

    rng = np.random.RandomState(random_state)
    minority_mask = y == 1
    X_min = X[minority_mask]
    X_maj = X[~minority_mask]

    n_to_generate = len(X_maj) - len(X_min)
    if n_to_generate <= 0:
        return X, y

    synthetic = []
    for _ in range(n_to_generate):
        anchor_idx = rng.randint(0, len(X_min))
        anchor     = X_min[anchor_idx]

        dists = np.linalg.norm(X_min - anchor, axis=1)
        dists[anchor_idx] = np.inf        
        nn_indices = np.argsort(dists)[:k]
        neighbour  = X_min[rng.choice(nn_indices)]

        lam     = rng.uniform(0, 1)
        new_pt  = anchor + lam * (neighbour - anchor)
        synthetic.append(new_pt)

    X_syn   = np.vstack(synthetic)
    y_syn   = np.ones(len(X_syn), dtype=int)

    X_bal   = np.vstack([X, X_syn])
    y_bal   = np.hstack([y, y_syn])
    return X_bal, y_bal


# ══════════════════════════════════════════════════════════════════════
# 3. BASELINE MODEL (Phase II reference)
# ══════════════════════════════════════════════════════════════════════

def train_baseline(X_train, X_test, y_train, y_test):
    """
    Reproduce the Phase II best Random Forest with its best parameters.
    Used as the comparison baseline for Phase III retraining.
    """
    print("\n[2/8] Training baseline Random Forest (Phase II reference) …")
    rf_baseline = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_baseline.fit(X_train, y_train)
    return rf_baseline


# ══════════════════════════════════════════════════════════════════════
# 4. RETRAINING STRATEGY A — SMOTE BALANCED
# ══════════════════════════════════════════════════════════════════════

def train_smote_model(X_train, X_test, y_train, y_test):
    """
    Retrain Random Forest on SMOTE-balanced training data.

    Why SMOTE here?
    ---------------
    Phase II used class_weight='balanced' as a soft correction.
    SMOTE physically creates new minority samples, giving the model
    more diverse bot examples to learn decision boundaries from.
    This reduces bias towards the majority class without weighting tricks.
    """
    print("\n[3/8] Retraining with SMOTE balancing …")
    X_bal, y_bal = smote_oversample(X_train, y_train)
    print(f"    After SMOTE — Human: {int((y_bal==0).sum()):,}  "
          f"Bot: {int((y_bal==1).sum()):,}")

    rf_smote = RandomForestClassifier(
        n_estimators=200,       # more trees → more stable
        max_features="sqrt",
        max_depth=20,           # slightly constrained to reduce overfit on synthetic data
        min_samples_leaf=2,     # prevents leaf nodes with only 1 synthetic sample
        class_weight=None,      # SMOTE handles balance; no extra weighting needed
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_smote.fit(X_bal, y_bal)
    return rf_smote, X_bal, y_bal